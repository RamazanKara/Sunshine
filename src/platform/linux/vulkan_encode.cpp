/**
 * @file src/platform/linux/vulkan_encode.cpp
 * @brief Vulkan-native encoder: DMA-BUF -> Vulkan compute (RGB->YUV) -> Vulkan Video encode.
 *        No EGL/GL dependency — all GPU work stays in a single Vulkan queue.
 */
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <drm_fourcc.h>
#include <mutex>
#include <sys/stat.h>
#if defined(__FreeBSD__)
  #include <sys/types.h>
#else
  #include <sys/sysmacros.h>
#endif
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

#include "graphics.h"
#if defined(__linux__)
  #include "src/amf/amf_native.h"
#endif
#include "src/config.h"
#include "src/logging.h"
#include "src/video.h"
#include "src/video_colorspace.h"
#include "vulkan_encode.h"

#if defined(__linux__)
  #include <AMF/core/VulkanAMF.h>
#endif

// SPIR-V data generated at build time
static const std::vector<uint32_t> rgb2yuv_comp_spv_data
#include "shaders/rgb2yuv.spv.inc"
  ;
static const size_t rgb2yuv_comp_spv_size = rgb2yuv_comp_spv_data.size() * sizeof(uint32_t);

using namespace std::literals;

namespace vk {

  // Match a DRI render node path to a Vulkan device index via VK_EXT_physical_device_drm.
  // Returns the index as a string (e.g. "1"), or empty string if no match.
  static std::string find_vulkan_index_for_render_node(const char *render_path) {
    struct stat node_stat;
    if (stat(render_path, &node_stat) < 0) {
      return {};
    }

    auto target_major = major(node_stat.st_rdev);
    auto target_minor = minor(node_stat.st_rdev);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;

    static const std::array<const char *, 1> instance_exts = {VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME};
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = instance_exts.size();
    ci.ppEnabledExtensionNames = instance_exts.data();
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) {
      // Retry without the extension for loaders that don't support it
      ci.enabledExtensionCount = 0;
      ci.ppEnabledExtensionNames = nullptr;
      if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) {
        return {};
      }
    }

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(inst, &count, devs.data());

    std::string result;
    for (uint32_t i = 0; i < count; i++) {
      VkPhysicalDeviceDrmPropertiesEXT drm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
      VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      props2.pNext = &drm;
      vkGetPhysicalDeviceProperties2(devs[i], &props2);
      if (drm.hasRender && drm.renderMajor == (int64_t) target_major && drm.renderMinor == (int64_t) target_minor) {
        result = std::to_string(i);
        break;
      }
    }
    vkDestroyInstance(inst, nullptr);
    return result;
  }

  static int create_vulkan_hwdevice(AVBufferRef **hw_device_buf, const std::string &required_device_extensions = {}) {
    AVDictionary *options = nullptr;
    if (!required_device_extensions.empty()) {
      av_dict_set(&options, "device_extensions", required_device_extensions.c_str(), 0);
    }
    auto release_options = util::fail_guard([&]() {
      av_dict_free(&options);
    });

    // Resolve render device path to Vulkan device index
    if (auto render_path = platf::resolve_render_device(); render_path[0] == '/') {
      if (auto idx = find_vulkan_index_for_render_node(render_path.c_str()); !idx.empty() && av_hwdevice_ctx_create(hw_device_buf, AV_HWDEVICE_TYPE_VULKAN, idx.c_str(), options, 0) >= 0) {
        return 0;
      }
    } else {
      // Non-path: treat as device name substring or numeric index
      if (av_hwdevice_ctx_create(hw_device_buf, AV_HWDEVICE_TYPE_VULKAN, render_path.c_str(), options, 0) >= 0) {
        return 0;
      }
    }
    // Final fallback: let FFmpeg pick default
    if (av_hwdevice_ctx_create(hw_device_buf, AV_HWDEVICE_TYPE_VULKAN, nullptr, options, 0) >= 0) {
      return 0;
    }
    return -1;
  }

  /**
   * @brief Vulkan shader constants used by the conversion pass.
   */
  struct PushConstants {
    std::array<float, 4> color_vec_y;  ///< Color vec y.
    std::array<float, 4> color_vec_u;  ///< Color vec u.
    std::array<float, 4> color_vec_v;  ///< Color vec v.
    std::array<float, 2> range_y;  ///< Range y.
    std::array<float, 2> range_uv;  ///< Range uv.
    std::array<int32_t, 2> src_offset;  ///< Src offset.
    std::array<int32_t, 2> src_size;  ///< Src size.
    std::array<int32_t, 2> dst_offset;  ///< Dst offset.
    std::array<int32_t, 2> dst_size;  ///< Dst size.
    std::array<int32_t, 2> dst_full_size;  ///< Dst full size.
    std::array<int32_t, 2> cursor_pos;  ///< Cursor pos.
    std::array<int32_t, 2> cursor_size;  ///< Cursor size.
    int32_t y_invert;  ///< Y invert.
  };

// Helper to check VkResult
/**
 * @def VK_CHECK(expr)
 * @brief Macro for VK CHECK.
 */
#define VK_CHECK(expr) \
  do { \
    VkResult _r = (expr); \
    if (_r != VK_SUCCESS) { \
      BOOST_LOG(error) << #expr << " failed: " << _r; \
      return -1; \
    } \
  } while (0)
/**
 * @def VK_CHECK_BOOL(expr)
 * @brief Macro for VK CHECK BOOL.
 */
#define VK_CHECK_BOOL(expr) \
  do { \
    VkResult _r = (expr); \
    if (_r != VK_SUCCESS) { \
      BOOST_LOG(error) << #expr << " failed: " << _r; \
      return false; \
    } \
  } while (0)

  /**
   * @brief Vulkan encode device that keeps converted frames in GPU memory.
   */
  class vk_vram_t: public platf::avcodec_encode_device_t {
  public:
    ~vk_vram_t() override {
      cleanup_pipeline();
    }

    /**
     * @brief Initialize Vulkan encode device and conversion resources.
     *
     * @param in_width In width.
     * @param in_height In height.
     * @param in_offset_x In offset x.
     * @param in_offset_y In offset y.
     * @return 0 on success; nonzero or negative platform status on failure.
     */
    int init(int in_width, int in_height, int in_offset_x = 0, int in_offset_y = 0) {
      width = in_width;
      height = in_height;
      offset_x = in_offset_x;
      offset_y = in_offset_y;
      this->data = (void *) &init_hw_device;
      return 0;
    }

    /**
     * @brief Initialize codec options.
     *
     * @param ctx Native context object used by the operation or callback.
     * @param options Request options or socket options to apply.
     */
    void init_codec_options(AVCodecContext *ctx, AVDictionary **options) override {
      // When VBR mode is selected (rc_mode=4), don't pin rc_min_rate to the target bitrate.
      // Having rc_min_rate == rc_max_rate == bit_rate in VBR mode prevents the encoder from
      // undershooting on simple frames, which builds up headroom that causes large overshoots
      // on complex frames.
      if (config::video.vk.rc_mode == 4) {
        ctx->rc_min_rate = 0;
      }
    }

    /**
     * @brief Attach frame resources used by the next conversion or encode operation.
     *
     * @param new_frame Frame to attach.
     * @param hw_frames_ctx_buf Hardware frames context buffer.
     * @return Status from updating frame.
     */
    int set_frame(AVFrame *new_frame, AVBufferRef *hw_frames_ctx_buf) override {
      this->hwframe.reset(new_frame);
      if (initialize_vulkan(hw_frames_ctx_buf) != 0) {
        return -1;
      }
      return set_target_frame(new_frame);
    }

    /**
     * @brief Initialize the converter for a caller-owned Vulkan frame pool.
     *
     * @param hw_frames_ctx_buf Vulkan hardware-frames context.
     * @return Zero on success.
     */
    int initialize_for_native_amf(AVBufferRef *hw_frames_ctx_buf) {
      return initialize_vulkan(hw_frames_ctx_buf);
    }

    /**
     * @brief Select a caller-owned Vulkan conversion target.
     *
     * @param new_frame Allocated Vulkan frame.
     * @param completion_semaphore Binary semaphore signaled after conversion.
     * @param wait_for_completion True when AMF left the semaphore signaled.
     * @return Zero when the target is usable.
     */
    int set_native_amf_target(AVFrame *new_frame, VkSemaphore completion_semaphore, bool wait_for_completion) {
      external_signal_semaphore = completion_semaphore;
      external_wait_semaphore = wait_for_completion ? completion_semaphore : VK_NULL_HANDLE;
      const auto result = set_target_frame(new_frame);
      if (result != 0) {
        cancel_native_amf_target();
      }
      return result;
    }

    /**
     * @brief Clear synchronization state for a conversion that was not submitted.
     */
    void cancel_native_amf_target() noexcept {
      external_signal_semaphore = VK_NULL_HANDLE;
      external_wait_semaphore = VK_NULL_HANDLE;
    }

    /**
     * @brief Copy one Vulkan YUV frame into another and signal AMF readiness.
     *
     * @param destination Destination Vulkan frame.
     * @param source Source Vulkan frame.
     * @param completion_semaphore Binary semaphore signaled after the copy.
     * @return Zero on success.
     */
    int copy_native_amf_frame(AVFrame *destination, AVFrame *source, VkSemaphore completion_semaphore) {
      return copy_frame(destination, source, completion_semaphore);
    }

    /**
     * @brief Release cached image views before a caller destroys its frame pool.
     */
    void release_native_amf_targets() {
      if (!vk_dev.dev) {
        return;
      }
      vkDeviceWaitIdle(vk_dev.dev);
      for (auto &[image, cached_target] : targets) {
        (void) image;
        if (cached_target.y_view) {
          vkDestroyImageView(vk_dev.dev, cached_target.y_view, nullptr);
        }
        if (cached_target.uv_view) {
          vkDestroyImageView(vk_dev.dev, cached_target.uv_view, nullptr);
        }
      }
      targets.clear();
      target = nullptr;
      frame = nullptr;
    }

  private:
    int initialize_vulkan(AVBufferRef *hw_frames_ctx_buf) {
      if (vulkan_initialized) {
        return hw_frames_ctx == hw_frames_ctx_buf ? 0 : -1;
      }
      if (!hw_frames_ctx_buf || !hw_frames_ctx_buf->data) {
        return -1;
      }

      hw_frames_ctx = hw_frames_ctx_buf;

      auto *frames_ctx = (AVHWFramesContext *) hw_frames_ctx_buf->data;
      auto *dev_ctx = (AVHWDeviceContext *) frames_ctx->device_ref->data;
      vk_dev.ctx = (AVVulkanDeviceContext *) dev_ctx->hwctx;
      vk_dev.dev = vk_dev.ctx->act_dev;
      vk_dev.phys_dev = vk_dev.ctx->phys_dev;
      is_10bit = (frames_ctx->sw_format == AV_PIX_FMT_P010);

      {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(vk_dev.phys_dev, &p);
        BOOST_LOG(info) << "Vulkan encode using GPU: " << p.deviceName;
      }

      // Find a compute-capable queue family from FFmpeg's context
      vk_dev.compute_qf = -1;
      for (int i = 0; i < vk_dev.ctx->nb_qf; i++) {
        if (vk_dev.ctx->qf[i].flags & VK_QUEUE_COMPUTE_BIT) {
          vk_dev.compute_qf = vk_dev.ctx->qf[i].idx;
          break;
        }
      }
      if (vk_dev.compute_qf < 0) {
        BOOST_LOG(error) << "No compute queue family in Vulkan device"sv;
        return -1;
      }

      vkGetDeviceQueue(vk_dev.dev, vk_dev.compute_qf, 0, &vk_dev.compute_queue);

      // Load extension functions
      vk_dev.getMemoryFdProperties = (PFN_vkGetMemoryFdPropertiesKHR)
        vkGetDeviceProcAddr(vk_dev.dev, "vkGetMemoryFdPropertiesKHR");

      if (!create_compute_pipeline()) {
        return -1;
      }
      if (!create_command_resources()) {
        return -1;
      }

      vulkan_initialized = true;
      return 0;
    }

    int set_target_frame(AVFrame *new_frame) {
      if (!vulkan_initialized || !new_frame) {
        return -1;
      }
      frame = new_frame;
      target = nullptr;
      descriptors_dirty = true;
      return 0;
    }

  public:
    /**
     * @brief Apply the configured colorspace metadata to the active frame.
     */
    void apply_colorspace() override {
      auto *colors = video::color_vectors_from_colorspace(colorspace, true);
      if (colors) {
        memcpy(push.color_vec_y.data(), colors->color_vec_y, sizeof(push.color_vec_y));
        memcpy(push.color_vec_u.data(), colors->color_vec_u, sizeof(push.color_vec_u));
        memcpy(push.color_vec_v.data(), colors->color_vec_v, sizeof(push.color_vec_v));
        memcpy(push.range_y.data(), colors->range_y, sizeof(push.range_y));
        memcpy(push.range_uv.data(), colors->range_uv, sizeof(push.range_uv));
      }
    }

    /**
     * @brief Configure FFmpeg Vulkan hardware frames for video encode input.
     *
     * @param frames FFmpeg hardware frames context to initialize.
     */
    void init_hwframes(AVHWFramesContext *frames) override {
      frames->initial_pool_size = 4;
      auto *vk_frames = (AVVulkanFramesContext *) frames->hwctx;
      vk_frames->tiling = VK_IMAGE_TILING_OPTIMAL;
      vk_frames->usage = (VkImageUsageFlagBits) (VK_IMAGE_USAGE_STORAGE_BIT |
                                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                 VK_IMAGE_USAGE_SAMPLED_BIT |
                                                 VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR);
    }

    /**
     * @brief Convert a captured frame into a Vulkan hardware frame.
     *
     * @param img Image or frame object to read from or populate.
     * @return Conversion status.
     */
    int convert(platf::img_t &img) override {
      auto &descriptor = (egl::img_descriptor_t &) img;

      // Get encoder target frame
      if (!frame->buf[0]) {
        if (av_hwframe_get_buffer(hw_frames_ctx, frame, 0) < 0) {
          BOOST_LOG(error) << "Failed to get Vulkan frame"sv;
          return -1;
        }
      }

      // Import new DMA-BUF as VkImage when capture sequence changes
      if (descriptor.sequence == 0) {
        // Probe/fallback frames have no DMA-BUF. Preserve the historical
        // undefined-content behavior, but still fulfill the synchronization
        // contract required by a native AMF consumer.
        return external_signal_semaphore ? signal_target_ready() : 0;
      }

      if (descriptor.sequence > sequence) {
        sequence = descriptor.sequence;
        if (!import_dmabuf(descriptor.sd)) {
          BOOST_LOG(error) << "Failed to import DMA-BUF"sv;
          return -1;
        }
        descriptors_dirty = true;
      }

      if (src.image == VK_NULL_HANDLE) {
        return -1;
      }

      // Setup Y/UV image views for the encoder target (once)
      if (!target) {
        if (!create_target_views()) {
          return -1;
        }
        descriptors_dirty = true;
      }

      // Update descriptor set only when source or target changed
      if (descriptors_dirty) {
        update_descriptors();
        descriptors_dirty = false;
      }

      if (descriptor.data && descriptor.serial != cursor_serial) {
        cursor_serial = descriptor.serial;
        if (!create_cursor_image(descriptor.src_w, descriptor.src_h, descriptor.data)) {
          return -1;
        }
        update_descriptors();
        descriptors_dirty = false;
      }

      // Preserve aspect ratio: fit src into dst, center with black bars.
      // UV plane is subsampled 2x, so keep effective size and offset even.
      float scalar = std::min((float) frame->width / width, (float) frame->height / height);
      int32_t eff_w = std::min<int32_t>(((int32_t) (width * scalar)) & ~1, frame->width & ~1);
      int32_t eff_h = std::min<int32_t>(((int32_t) (height * scalar)) & ~1, frame->height & ~1);
      int32_t dst_off_x = ((frame->width - eff_w) / 2) & ~1;
      int32_t dst_off_y = ((frame->height - eff_h) / 2) & ~1;
      eff_w = std::min(eff_w, (frame->width - dst_off_x) & ~1);
      eff_h = std::min(eff_h, (frame->height - dst_off_y) & ~1);

      // Fill push constants
      push.src_offset[0] = offset_x;
      push.src_offset[1] = offset_y;
      push.src_size[0] = width;
      push.src_size[1] = height;
      push.dst_offset[0] = dst_off_x;
      push.dst_offset[1] = dst_off_y;
      push.dst_size[0] = eff_w;
      push.dst_size[1] = eff_h;
      push.dst_full_size[0] = frame->width;
      push.dst_full_size[1] = frame->height;
      push.y_invert = descriptor.y_invert ? 1 : 0;

      if (descriptor.data) {
        float scale_x = (float) eff_w / width;
        float scale_y = (float) eff_h / height;
        push.cursor_pos[0] = (int32_t) ((descriptor.x - offset_x) * scale_x) + dst_off_x;
        push.cursor_pos[1] = (int32_t) ((descriptor.y - offset_y) * scale_y) + dst_off_y;
        push.cursor_size[0] = (int32_t) (descriptor.width * scale_x);
        push.cursor_size[1] = (int32_t) (descriptor.height * scale_y);
      } else {
        push.cursor_size[0] = 0;
      }

      // Record and submit compute dispatch
      return dispatch_compute();
    }

  private:
    bool create_compute_pipeline() {
      // Shader module
      VkShaderModuleCreateInfo shader_ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      shader_ci.codeSize = rgb2yuv_comp_spv_size;
      shader_ci.pCode = rgb2yuv_comp_spv_data.data();
      VK_CHECK_BOOL(vkCreateShaderModule(vk_dev.dev, &shader_ci, nullptr, &compute.shader_module));

      // Descriptor set layout: binding 0=sampler, 1=Y storage, 2=UV storage, 3=cursor sampler
      std::array<VkDescriptorSetLayoutBinding, 4> bindings = {};
      bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
      bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
      bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
      bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

      VkDescriptorSetLayoutCreateInfo ds_layout_ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      ds_layout_ci.bindingCount = bindings.size();
      ds_layout_ci.pBindings = bindings.data();
      VK_CHECK_BOOL(vkCreateDescriptorSetLayout(vk_dev.dev, &ds_layout_ci, nullptr, &compute.ds_layout));

      // Push constant range
      VkPushConstantRange pc_range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};

      VkPipelineLayoutCreateInfo pl_ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pl_ci.setLayoutCount = 1;
      pl_ci.pSetLayouts = &compute.ds_layout;
      pl_ci.pushConstantRangeCount = 1;
      pl_ci.pPushConstantRanges = &pc_range;
      VK_CHECK_BOOL(vkCreatePipelineLayout(vk_dev.dev, &pl_ci, nullptr, &compute.pipeline_layout));

      // Compute pipeline
      VkComputePipelineCreateInfo comp_ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      comp_ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      comp_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      comp_ci.stage.module = compute.shader_module;
      comp_ci.stage.pName = "main";
      comp_ci.layout = compute.pipeline_layout;
      VK_CHECK_BOOL(vkCreateComputePipelines(vk_dev.dev, VK_NULL_HANDLE, 1, &comp_ci, nullptr, &compute.pipeline));

      // Descriptor pool
      std::array<VkDescriptorPoolSize, 2> pool_sizes = {{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
      }};
      VkDescriptorPoolCreateInfo pool_ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pool_ci.maxSets = 1;
      pool_ci.poolSizeCount = pool_sizes.size();
      pool_ci.pPoolSizes = pool_sizes.data();
      VK_CHECK_BOOL(vkCreateDescriptorPool(vk_dev.dev, &pool_ci, nullptr, &compute.desc_pool));

      VkDescriptorSetAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      alloc_info.descriptorPool = compute.desc_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &compute.ds_layout;
      VK_CHECK_BOOL(vkAllocateDescriptorSets(vk_dev.dev, &alloc_info, &compute.desc_set));

      // Sampler for source image
      VkSamplerCreateInfo sampler_ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
      sampler_ci.magFilter = VK_FILTER_LINEAR;
      sampler_ci.minFilter = VK_FILTER_LINEAR;
      sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      VK_CHECK_BOOL(vkCreateSampler(vk_dev.dev, &sampler_ci, nullptr, &compute.sampler));

      if (!create_cursor_image(1, 1, nullptr)) {
        return false;
      }

      return true;
    }

    bool create_command_resources() {
      VkCommandPoolCreateInfo pool_ci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      pool_ci.queueFamilyIndex = vk_dev.compute_qf;
      pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      VK_CHECK_BOOL(vkCreateCommandPool(vk_dev.dev, &pool_ci, nullptr, &cmd.pool));

      VkCommandBufferAllocateInfo alloc_ci = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      alloc_ci.commandPool = cmd.pool;
      alloc_ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      alloc_ci.commandBufferCount = CMD_RING_SIZE;
      VK_CHECK_BOOL(vkAllocateCommandBuffers(vk_dev.dev, &alloc_ci, cmd.ring.data()));

      VkFenceCreateInfo fence_ci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
      for (auto &fence : cmd.fences) {
        VK_CHECK_BOOL(vkCreateFence(vk_dev.dev, &fence_ci, nullptr, &fence));
      }

      return true;
    }

    struct drm_format_info {
      VkFormat format;
      VkComponentMapping swizzle;
    };

    static drm_format_info drm_fourcc_to_vk_format(uint32_t fourcc) {
      static constexpr VkComponentMapping identity = {
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
      };
      static constexpr VkComponentMapping bgr_swap = {
        VK_COMPONENT_SWIZZLE_B,
        VK_COMPONENT_SWIZZLE_G,
        VK_COMPONENT_SWIZZLE_R,
        VK_COMPONENT_SWIZZLE_A,
      };

      switch (fourcc) {
        case DRM_FORMAT_XRGB8888:
        case DRM_FORMAT_ARGB8888:
          return {VK_FORMAT_B8G8R8A8_UNORM, identity};
        case DRM_FORMAT_XBGR8888:
        case DRM_FORMAT_ABGR8888:
          return {VK_FORMAT_R8G8B8A8_UNORM, identity};
        case DRM_FORMAT_XRGB2101010:
        case DRM_FORMAT_ARGB2101010:
          return {VK_FORMAT_A2R10G10B10_UNORM_PACK32, identity};
        case DRM_FORMAT_XBGR2101010:
        case DRM_FORMAT_ABGR2101010:
          return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, identity};
        case DRM_FORMAT_XBGR16161616:
        case DRM_FORMAT_ABGR16161616:
          return {VK_FORMAT_R16G16B16A16_UNORM, identity};
        case DRM_FORMAT_XRGB16161616:
        case DRM_FORMAT_ARGB16161616:
          return {VK_FORMAT_R16G16B16A16_UNORM, bgr_swap};
        case DRM_FORMAT_XBGR16161616F:
        case DRM_FORMAT_ABGR16161616F:
          return {VK_FORMAT_R16G16B16A16_SFLOAT, identity};
        case DRM_FORMAT_XRGB16161616F:
        case DRM_FORMAT_ARGB16161616F:
          return {VK_FORMAT_R16G16B16A16_SFLOAT, bgr_swap};
        default:
          BOOST_LOG(warning) << "Unknown DRM fourcc 0x" << std::hex << fourcc << std::dec << ", assuming B8G8R8A8";
          return {VK_FORMAT_B8G8R8A8_UNORM, identity};
      }
    }

    /**
     * @brief Query the driver-expected plane count for a format+modifier pair.
     * @return Expected plane count, or 0 if unknown.
     */
    int query_modifier_plane_count(VkFormat format, uint64_t modifier) {
      VkDrmFormatModifierPropertiesListEXT mod_list = {VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
      VkFormatProperties2 fmt_props2 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
      fmt_props2.pNext = &mod_list;
      vkGetPhysicalDeviceFormatProperties2(vk_dev.phys_dev, format, &fmt_props2);
      std::vector<VkDrmFormatModifierPropertiesEXT> mod_props(mod_list.drmFormatModifierCount);
      mod_list.pDrmFormatModifierProperties = mod_props.data();
      vkGetPhysicalDeviceFormatProperties2(vk_dev.phys_dev, format, &fmt_props2);
      for (const auto &mp : mod_props) {
        if (mp.drmFormatModifier == modifier) {
          return mp.drmFormatModifierPlaneCount;
        }
      }
      return 0;
    }

    bool import_dmabuf(const egl::surface_descriptor_t &sd) {
      destroy_src_image();

      int fd = dup(sd.fds[0]);
      if (fd < 0) {
        return false;
      }

      // Query memory requirements for this DMA-BUF
      VkMemoryFdPropertiesKHR fd_props = {VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
      if (vk_dev.getMemoryFdProperties) {
        vk_dev.getMemoryFdProperties(vk_dev.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fd_props);
      }

      // Create VkImage for the DMA-BUF
      VkExternalMemoryImageCreateInfo ext_ci = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
      ext_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

      std::array<VkSubresourceLayout, 4> drm_layouts = {};
      VkImageDrmFormatModifierExplicitCreateInfoEXT drm_ci = {
        VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT
      };
      VkImageTiling tiling;

      auto [vk_format, vk_swizzle] = drm_fourcc_to_vk_format(sd.fourcc);

      if (sd.modifier != DRM_FORMAT_MOD_INVALID) {
        int dmabuf_planes = 0;
        for (int i = 0; i < 4 && sd.fds[i] >= 0; ++i) {
          dmabuf_planes++;
        }

        // Query driver for the expected plane count for this format+modifier.
        // DMA-BUF exports may include extra metadata planes (e.g. AMD DCC).
        int expected = query_modifier_plane_count(vk_format, sd.modifier);
        int plane_count = (expected > 0 && expected <= dmabuf_planes) ? expected : dmabuf_planes;

        for (int i = 0; i < plane_count; ++i) {
          drm_layouts[i].offset = sd.offsets[i];
          drm_layouts[i].rowPitch = sd.pitches[i];
        }
        drm_ci.drmFormatModifier = sd.modifier;
        drm_ci.drmFormatModifierPlaneCount = plane_count;
        drm_ci.pPlaneLayouts = drm_layouts.data();
        ext_ci.pNext = &drm_ci;
        tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
      } else {
        tiling = VK_IMAGE_TILING_LINEAR;
      }

      VkImageCreateInfo img_ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      img_ci.pNext = &ext_ci;
      img_ci.imageType = VK_IMAGE_TYPE_2D;
      img_ci.format = vk_format;
      img_ci.extent = {(uint32_t) sd.width, (uint32_t) sd.height, 1};
      img_ci.mipLevels = 1;
      img_ci.arrayLayers = 1;
      img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
      img_ci.tiling = tiling;
      img_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
      img_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      auto res = vkCreateImage(vk_dev.dev, &img_ci, nullptr, &src.image);
      if (res != VK_SUCCESS) {
        close(fd);
        BOOST_LOG(error) << "vkCreateImage for DMA-BUF failed: " << res
                         << " (modifier=0x" << std::hex << sd.modifier << std::dec
                         << ", pitch=" << sd.pitches[0] << ", offset=" << sd.offsets[0] << ")";
        return false;
      }

      // Bind imported DMA-BUF memory
      VkMemoryRequirements mem_req;
      vkGetImageMemoryRequirements(vk_dev.dev, src.image, &mem_req);

      VkImportMemoryFdInfoKHR import_fd = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
      import_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
      import_fd.fd = fd;  // Vulkan takes ownership

      VkMemoryAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      alloc_info.pNext = &import_fd;
      alloc_info.allocationSize = mem_req.size;
      alloc_info.memoryTypeIndex = find_memory_type(
        fd_props.memoryTypeBits ? fd_props.memoryTypeBits : mem_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
      );

      VkDeviceMemory src_mem = VK_NULL_HANDLE;
      res = vkAllocateMemory(vk_dev.dev, &alloc_info, nullptr, &src_mem);
      if (res != VK_SUCCESS) {
        BOOST_LOG(error) << "vkAllocateMemory for DMA-BUF failed: " << res;
        vkDestroyImage(vk_dev.dev, src.image, nullptr);
        src.image = VK_NULL_HANDLE;
        return false;
      }

      vkBindImageMemory(vk_dev.dev, src.image, src_mem, 0);

      // Create image view
      VkImageViewCreateInfo view_ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view_ci.image = src.image;
      view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_ci.format = vk_format;
      view_ci.components = vk_swizzle;
      view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      VK_CHECK_BOOL(vkCreateImageView(vk_dev.dev, &view_ci, nullptr, &src.view));

      src.mem = src_mem;
      return true;
    }

    bool create_cursor_image(int w, int h, const uint8_t *pixels) {
      destroy_cursor_image();

      VkImageCreateInfo img_ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      img_ci.imageType = VK_IMAGE_TYPE_2D;
      img_ci.format = VK_FORMAT_B8G8R8A8_UNORM;
      img_ci.extent = {(uint32_t) w, (uint32_t) h, 1};
      img_ci.mipLevels = 1;
      img_ci.arrayLayers = 1;
      img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
      img_ci.tiling = VK_IMAGE_TILING_LINEAR;
      img_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
      img_ci.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
      VK_CHECK_BOOL(vkCreateImage(vk_dev.dev, &img_ci, nullptr, &cursor.image));

      VkMemoryRequirements mem_req;
      vkGetImageMemoryRequirements(vk_dev.dev, cursor.image, &mem_req);
      VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      alloc.allocationSize = mem_req.size;
      alloc.memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      VK_CHECK_BOOL(vkAllocateMemory(vk_dev.dev, &alloc, nullptr, &cursor.mem));
      VK_CHECK_BOOL(vkBindImageMemory(vk_dev.dev, cursor.image, cursor.mem, 0));

      if (pixels) {
        void *mapped;
        VK_CHECK_BOOL(vkMapMemory(vk_dev.dev, cursor.mem, 0, VK_WHOLE_SIZE, 0, &mapped));
        VkImageSubresource subres = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
        VkSubresourceLayout layout;
        vkGetImageSubresourceLayout(vk_dev.dev, cursor.image, &subres, &layout);
        for (int y = 0; y < h; y++) {
          memcpy((uint8_t *) mapped + layout.offset + y * layout.rowPitch, pixels + y * w * 4, w * 4);
        }
        vkUnmapMemory(vk_dev.dev, cursor.mem);
      }

      VkImageViewCreateInfo view_ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view_ci.image = cursor.image;
      view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_ci.format = VK_FORMAT_B8G8R8A8_UNORM;
      view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      VK_CHECK_BOOL(vkCreateImageView(vk_dev.dev, &view_ci, nullptr, &cursor.view));

      cursor.needs_transition = true;
      descriptors_dirty = true;
      return true;
    }

    void destroy_cursor_image() {
      if (cursor.view) {
        vkDestroyImageView(vk_dev.dev, cursor.view, nullptr);
        cursor.view = VK_NULL_HANDLE;
      }
      if (cursor.image) {
        vkDestroyImage(vk_dev.dev, cursor.image, nullptr);
        cursor.image = VK_NULL_HANDLE;
      }
      if (cursor.mem) {
        vkFreeMemory(vk_dev.dev, cursor.mem, nullptr);
        cursor.mem = VK_NULL_HANDLE;
      }
    }

    bool create_target_views() {
      auto *vk_frame = (AVVkFrame *) frame->data[0];
      if (!vk_frame || !vk_frame->img[0]) {
        return false;
      }

      auto [target_it, inserted] = targets.try_emplace(vk_frame->img[0]);
      target = &target_it->second;
      if (!inserted) {
        return true;
      }

      auto y_fmt = is_10bit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
      auto uv_fmt = is_10bit ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R8G8_UNORM;

      // Detect multiplane vs multi-image layout
      int num_imgs = 0;
      for (int i = 0; i < AV_NUM_DATA_POINTERS && vk_frame->img[i]; i++) {
        num_imgs++;
      }

      if (num_imgs == 1) {
        // Single multiplane image — create plane views
        VkImageViewCreateInfo view_ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_ci.image = vk_frame->img[0];
        view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;

        // Y plane
        view_ci.format = y_fmt;
        view_ci.subresourceRange = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 1, 0, 1};
        VK_CHECK_BOOL(vkCreateImageView(vk_dev.dev, &view_ci, nullptr, &target->y_view));

        // UV plane
        view_ci.format = uv_fmt;
        view_ci.subresourceRange = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 1, 0, 1};
        VK_CHECK_BOOL(vkCreateImageView(vk_dev.dev, &view_ci, nullptr, &target->uv_view));
      } else {
        // Separate images per plane
        VkImageViewCreateInfo view_ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        view_ci.image = vk_frame->img[0];
        view_ci.format = y_fmt;
        VK_CHECK_BOOL(vkCreateImageView(vk_dev.dev, &view_ci, nullptr, &target->y_view));

        view_ci.image = vk_frame->img[1];
        view_ci.format = uv_fmt;
        VK_CHECK_BOOL(vkCreateImageView(vk_dev.dev, &view_ci, nullptr, &target->uv_view));
      }
      return true;
    }

    void update_descriptors() {
      VkDescriptorImageInfo src_info = {compute.sampler, src.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkDescriptorImageInfo y_info = {VK_NULL_HANDLE, target->y_view, VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo uv_info = {VK_NULL_HANDLE, target->uv_view, VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo cursor_info = {compute.sampler, cursor.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

      std::array<VkWriteDescriptorSet, 4> writes = {};
      writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compute.desc_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &src_info, nullptr, nullptr};
      writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compute.desc_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &y_info, nullptr, nullptr};
      writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compute.desc_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &uv_info, nullptr, nullptr};
      writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compute.desc_set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cursor_info, nullptr, nullptr};
      vkUpdateDescriptorSets(vk_dev.dev, writes.size(), writes.data(), 0, nullptr);
    }

    int dispatch_compute() {
      auto *vk_frame = (AVVkFrame *) frame->data[0];
      int num_imgs = 0;
      for (int i = 0; i < AV_NUM_DATA_POINTERS && vk_frame->img[i]; i++) {
        num_imgs++;
      }

      // Bound command-buffer reuse to actual GPU completion. Three elapsed
      // frames are not a synchronization primitive when a queue stalls.
      const auto command_index = cmd.ring_idx;
      auto cmd_buf = cmd.ring[command_index];
      auto cmd_fence = cmd.fences[command_index];
      cmd.ring_idx = (cmd.ring_idx + 1) % CMD_RING_SIZE;
      auto fence_result = vkWaitForFences(vk_dev.dev, 1, &cmd_fence, VK_TRUE, 1000000000ULL);
      if (fence_result != VK_SUCCESS) {
        BOOST_LOG(error) << "Vulkan conversion command-buffer wait failed: " << fence_result;
        return -1;
      }
      VK_CHECK(vkResetFences(vk_dev.dev, 1, &cmd_fence));
      VK_CHECK(vkResetCommandBuffer(cmd_buf, 0));

      VkCommandBufferBeginInfo begin_ci = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin_ci.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      VK_CHECK(vkBeginCommandBuffer(cmd_buf, &begin_ci));

      // Transition source image to SHADER_READ_ONLY
      VkImageMemoryBarrier src_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      src_barrier.srcAccessMask = 0;
      src_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      src_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      src_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      src_barrier.image = src.image;
      src_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      src_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
      src_barrier.dstQueueFamilyIndex = vk_dev.compute_qf;

      vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &src_barrier);

      // Transition cursor image if needed
      if (cursor.needs_transition) {
        VkImageMemoryBarrier cursor_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        cursor_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        cursor_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        cursor_barrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
        cursor_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        cursor_barrier.image = cursor.image;
        cursor_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        cursor_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        cursor_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &cursor_barrier);
        cursor.needs_transition = false;
      }

      // Transition target planes to GENERAL for storage writes
      std::array<VkImageMemoryBarrier, 2> dst_barriers = {};
      int num_dst_barriers = (num_imgs == 1) ? 1 : 2;
      for (int i = 0; i < num_dst_barriers; i++) {
        dst_barriers[i] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        dst_barriers[i].srcAccessMask = target->initialized ? VK_ACCESS_MEMORY_READ_BIT : 0;
        dst_barriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        dst_barriers[i].oldLayout = target->initialized ? vk_frame->layout[i] : VK_IMAGE_LAYOUT_UNDEFINED;
        dst_barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        dst_barriers[i].image = vk_frame->img[num_imgs == 1 ? 0 : i];
        dst_barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        dst_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dst_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      }

      vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, num_dst_barriers, dst_barriers.data());

      // Bind pipeline and dispatch
      vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
      vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline_layout, 0, 1, &compute.desc_set, 0, nullptr);
      vkCmdPushConstants(cmd_buf, compute.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);

      uint32_t gx = (frame->width + 15) / 16;
      uint32_t gy = (frame->height + 15) / 16;
      vkCmdDispatch(cmd_buf, gx, gy, 1);

      VK_CHECK(vkEndCommandBuffer(cmd_buf));

      // Submit with timeline semaphore signaling for FFmpeg
      VkTimelineSemaphoreSubmitInfo timeline_info = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      std::array<VkSemaphore, AV_NUM_DATA_POINTERS + 1> wait_sems = {};
      std::array<VkSemaphore, AV_NUM_DATA_POINTERS + 1> signal_sems = {};
      std::array<uint64_t, AV_NUM_DATA_POINTERS + 1> wait_vals = {};
      std::array<uint64_t, AV_NUM_DATA_POINTERS + 1> signal_vals = {};
      std::array<VkPipelineStageFlags, AV_NUM_DATA_POINTERS + 1> wait_stages = {};
      int wait_count = 0;
      int signal_count = 0;

      for (int i = 0; i < AV_NUM_DATA_POINTERS && vk_frame->sem[i]; i++) {
        wait_sems[wait_count] = vk_frame->sem[i];
        wait_vals[wait_count] = vk_frame->sem_value[i];
        wait_stages[wait_count] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        ++wait_count;

        signal_sems[signal_count] = vk_frame->sem[i];
        signal_vals[signal_count] = vk_frame->sem_value[i] + 1;
        vk_frame->sem_value[i]++;
        ++signal_count;
      }

      if (external_wait_semaphore) {
        wait_sems[wait_count] = external_wait_semaphore;
        wait_vals[wait_count] = 0;
        wait_stages[wait_count] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        ++wait_count;
      }

      if (external_signal_semaphore) {
        signal_sems[signal_count] = external_signal_semaphore;
        signal_vals[signal_count] = 0;
        ++signal_count;
      }

      timeline_info.waitSemaphoreValueCount = wait_count;
      timeline_info.pWaitSemaphoreValues = wait_vals.data();
      timeline_info.signalSemaphoreValueCount = signal_count;
      timeline_info.pSignalSemaphoreValues = signal_vals.data();

      VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.pNext = &timeline_info;
      submit.waitSemaphoreCount = wait_count;
      submit.pWaitSemaphores = wait_sems.data();
      submit.pWaitDstStageMask = wait_stages.data();
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &cmd_buf;
      submit.signalSemaphoreCount = signal_count;
      submit.pSignalSemaphores = signal_sems.data();

      auto res = vkQueueSubmit(vk_dev.compute_queue, 1, &submit, cmd_fence);

      if (res != VK_SUCCESS) {
        BOOST_LOG(error) << "vkQueueSubmit failed: " << res;
        return -1;
      }
      external_signal_semaphore = VK_NULL_HANDLE;
      external_wait_semaphore = VK_NULL_HANDLE;

      // Update frame layouts for FFmpeg
      for (int i = 0; i < AV_NUM_DATA_POINTERS && vk_frame->img[i]; i++) {
        vk_frame->layout[i] = VK_IMAGE_LAYOUT_GENERAL;
        vk_frame->access[i] = VK_ACCESS_SHADER_WRITE_BIT;
        vk_frame->queue_family[i] = vk_dev.compute_qf;
      }

      target->initialized = true;

      return 0;
    }

    int copy_frame(AVFrame *destination, AVFrame *source, VkSemaphore completion_semaphore) {
      if (!vulkan_initialized || !destination || !source || !destination->data[0] || !source->data[0]) {
        return -1;
      }
      auto *dst = reinterpret_cast<AVVkFrame *>(destination->data[0]);
      auto *src_frame = reinterpret_cast<AVVkFrame *>(source->data[0]);
      if (!dst->img[0] || dst->img[1] || !src_frame->img[0] || src_frame->img[1]) {
        BOOST_LOG(error) << "Native AMF repeat copy requires single multiplane Vulkan images"sv;
        return -1;
      }

      const auto command_index = cmd.ring_idx;
      auto command_buffer = cmd.ring[command_index];
      auto command_fence = cmd.fences[command_index];
      cmd.ring_idx = (cmd.ring_idx + 1) % CMD_RING_SIZE;
      auto result = vkWaitForFences(vk_dev.dev, 1, &command_fence, VK_TRUE, 1000000000ULL);
      if (result != VK_SUCCESS || vkResetFences(vk_dev.dev, 1, &command_fence) != VK_SUCCESS ||
          vkResetCommandBuffer(command_buffer, 0) != VK_SUCCESS) {
        BOOST_LOG(error) << "Native AMF repeat-copy command buffer is unavailable"sv;
        return -1;
      }

      VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
        return -1;
      }

      std::array<VkImageMemoryBarrier, 2> before = {};
      before[0] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      before[0].srcAccessMask = src_frame->access[0];
      before[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      before[0].oldLayout = src_frame->layout[0];
      before[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      before[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      before[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      before[0].image = src_frame->img[0];
      before[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      before[1] = before[0];
      before[1].srcAccessMask = dst->access[0];
      before[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      before[1].oldLayout = dst->layout[0];
      before[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      before[1].image = dst->img[0];
      vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        before.size(),
        before.data()
      );

      VkImageCopy region = {};
      region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.extent = {
        static_cast<uint32_t>(destination->width),
        static_cast<uint32_t>(destination->height),
        1,
      };
      vkCmdCopyImage(
        command_buffer,
        src_frame->img[0],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst->img[0],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
      );

      std::array<VkImageMemoryBarrier, 2> after = before;
      after[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      after[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      after[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      after[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
      after[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      after[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      after[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      after[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
      vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        after.size(),
        after.data()
      );
      if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        return -1;
      }

      std::array<VkSemaphore, 2> wait_semaphores = {src_frame->sem[0], dst->sem[0]};
      std::array<uint64_t, 2> wait_values = {src_frame->sem_value[0], dst->sem_value[0]};
      std::array<VkPipelineStageFlags, 2> wait_stages = {
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
      };
      std::array<VkSemaphore, 3> signal_semaphores = {
        src_frame->sem[0],
        dst->sem[0],
        completion_semaphore,
      };
      std::array<uint64_t, 3> signal_values = {
        ++src_frame->sem_value[0],
        ++dst->sem_value[0],
        0,
      };
      VkTimelineSemaphoreSubmitInfo timeline = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      timeline.waitSemaphoreValueCount = wait_values.size();
      timeline.pWaitSemaphoreValues = wait_values.data();
      timeline.signalSemaphoreValueCount = signal_values.size();
      timeline.pSignalSemaphoreValues = signal_values.data();
      VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.pNext = &timeline;
      submit.waitSemaphoreCount = wait_semaphores.size();
      submit.pWaitSemaphores = wait_semaphores.data();
      submit.pWaitDstStageMask = wait_stages.data();
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command_buffer;
      submit.signalSemaphoreCount = signal_semaphores.size();
      submit.pSignalSemaphores = signal_semaphores.data();
      result = vkQueueSubmit(vk_dev.compute_queue, 1, &submit, command_fence);
      if (result != VK_SUCCESS) {
        BOOST_LOG(error) << "Native AMF repeat-copy submission failed: " << result;
        return -1;
      }

      src_frame->layout[0] = VK_IMAGE_LAYOUT_GENERAL;
      src_frame->access[0] = VK_ACCESS_MEMORY_READ_BIT;
      src_frame->queue_family[0] = vk_dev.compute_qf;
      dst->layout[0] = VK_IMAGE_LAYOUT_GENERAL;
      dst->access[0] = VK_ACCESS_MEMORY_READ_BIT;
      dst->queue_family[0] = vk_dev.compute_qf;
      targets[src_frame->img[0]].initialized = true;
      targets[dst->img[0]].initialized = true;
      return 0;
    }

    int signal_target_ready() {
      auto *vk_frame = frame && frame->data[0] ? reinterpret_cast<AVVkFrame *>(frame->data[0]) : nullptr;
      if (!vk_frame || !vk_frame->img[0] || !vk_frame->sem[0] || !external_signal_semaphore) {
        return -1;
      }

      const auto command_index = cmd.ring_idx;
      auto command_buffer = cmd.ring[command_index];
      auto command_fence = cmd.fences[command_index];
      cmd.ring_idx = (cmd.ring_idx + 1) % CMD_RING_SIZE;
      auto result = vkWaitForFences(vk_dev.dev, 1, &command_fence, VK_TRUE, 1000000000ULL);
      if (result != VK_SUCCESS || vkResetFences(vk_dev.dev, 1, &command_fence) != VK_SUCCESS ||
          vkResetCommandBuffer(command_buffer, 0) != VK_SUCCESS) {
        return -1;
      }
      VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS ||
          vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        return -1;
      }

      std::array<VkSemaphore, 2> wait_semaphores = {vk_frame->sem[0], external_wait_semaphore};
      std::array<uint64_t, 2> wait_values = {vk_frame->sem_value[0], 0};
      std::array<VkPipelineStageFlags, 2> wait_stages = {
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      };
      const uint32_t wait_count = external_wait_semaphore ? 2 : 1;
      std::array<VkSemaphore, 2> signal_semaphores = {vk_frame->sem[0], external_signal_semaphore};
      std::array<uint64_t, 2> signal_values = {++vk_frame->sem_value[0], 0};
      VkTimelineSemaphoreSubmitInfo timeline = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      timeline.waitSemaphoreValueCount = wait_count;
      timeline.pWaitSemaphoreValues = wait_values.data();
      timeline.signalSemaphoreValueCount = signal_values.size();
      timeline.pSignalSemaphoreValues = signal_values.data();
      VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.pNext = &timeline;
      submit.waitSemaphoreCount = wait_count;
      submit.pWaitSemaphores = wait_semaphores.data();
      submit.pWaitDstStageMask = wait_stages.data();
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command_buffer;
      submit.signalSemaphoreCount = signal_semaphores.size();
      submit.pSignalSemaphores = signal_semaphores.data();
      result = vkQueueSubmit(vk_dev.compute_queue, 1, &submit, command_fence);
      external_wait_semaphore = VK_NULL_HANDLE;
      external_signal_semaphore = VK_NULL_HANDLE;
      return result == VK_SUCCESS ? 0 : -1;
    }

    uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
      VkPhysicalDeviceMemoryProperties mem_props;
      vkGetPhysicalDeviceMemoryProperties(vk_dev.phys_dev, &mem_props);
      for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props) {
          return i;
        }
      }
      // Fallback: any matching type bit
      for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if (type_bits & (1 << i)) {
          return i;
        }
      }
      return 0;
    }

    void destroy_src_image() {
      if (src.image) {
        // Defer destruction — the GPU may still be using this image.
        // By the time we wrap around (4 frames later), it's guaranteed done.
        auto &slot = defer_ring[defer_idx];
        if (slot.view) {
          vkDestroyImageView(vk_dev.dev, slot.view, nullptr);
        }
        if (slot.image) {
          vkDestroyImage(vk_dev.dev, slot.image, nullptr);
        }
        if (slot.mem) {
          vkFreeMemory(vk_dev.dev, slot.mem, nullptr);
        }
        slot = src;
        defer_idx = (defer_idx + 1) % DEFER_RING_SIZE;
      }
      src = {};
    }

    void cleanup_pipeline() {
      if (!vk_dev.dev) {
        return;
      }
      vkDeviceWaitIdle(vk_dev.dev);
      destroy_src_image();
      // Flush deferred destroys
      for (auto &slot : defer_ring) {
        if (slot.view) {
          vkDestroyImageView(vk_dev.dev, slot.view, nullptr);
        }
        if (slot.image) {
          vkDestroyImage(vk_dev.dev, slot.image, nullptr);
        }
        if (slot.mem) {
          vkFreeMemory(vk_dev.dev, slot.mem, nullptr);
        }
        slot = {};
      }
      for (auto &[image, cached_target] : targets) {
        (void) image;
        if (cached_target.y_view) {
          vkDestroyImageView(vk_dev.dev, cached_target.y_view, nullptr);
        }
        if (cached_target.uv_view) {
          vkDestroyImageView(vk_dev.dev, cached_target.uv_view, nullptr);
        }
      }
      targets.clear();
      target = nullptr;
      destroy_cursor_image();
      if (cmd.pool) {
        vkDestroyCommandPool(vk_dev.dev, cmd.pool, nullptr);
      }
      for (auto &fence : cmd.fences) {
        if (fence) {
          vkDestroyFence(vk_dev.dev, fence, nullptr);
          fence = VK_NULL_HANDLE;
        }
      }
      if (compute.sampler) {
        vkDestroySampler(vk_dev.dev, compute.sampler, nullptr);
      }
      if (compute.desc_pool) {
        vkDestroyDescriptorPool(vk_dev.dev, compute.desc_pool, nullptr);
      }
      if (compute.pipeline) {
        vkDestroyPipeline(vk_dev.dev, compute.pipeline, nullptr);
      }
      if (compute.pipeline_layout) {
        vkDestroyPipelineLayout(vk_dev.dev, compute.pipeline_layout, nullptr);
      }
      if (compute.ds_layout) {
        vkDestroyDescriptorSetLayout(vk_dev.dev, compute.ds_layout, nullptr);
      }
      if (compute.shader_module) {
        vkDestroyShaderModule(vk_dev.dev, compute.shader_module, nullptr);
      }
    }

    static int init_hw_device(platf::avcodec_encode_device_t *, AVBufferRef **hw_device_buf) {
      return create_vulkan_hwdevice(hw_device_buf);
    }

    // Dimensions
    int width = 0;
    int height = 0;
    int offset_x = 0;
    int offset_y = 0;
    bool is_10bit = false;
    bool vulkan_initialized = false;
    AVBufferRef *hw_frames_ctx = nullptr;
    frame_t hwframe;
    std::uint64_t sequence = 0;

    // Vulkan device (from FFmpeg)
    struct vk_device_t {
      VkDevice dev = VK_NULL_HANDLE;
      VkPhysicalDevice phys_dev = VK_NULL_HANDLE;
      AVVulkanDeviceContext *ctx = nullptr;
      int compute_qf = -1;
      VkQueue compute_queue = VK_NULL_HANDLE;
      PFN_vkGetMemoryFdPropertiesKHR getMemoryFdProperties = nullptr;
    };

    vk_device_t vk_dev = {};

    // Compute pipeline
    struct compute_pipeline_t {
      VkShaderModule shader_module = VK_NULL_HANDLE;
      VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
      VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
      VkPipeline pipeline = VK_NULL_HANDLE;
      VkDescriptorPool desc_pool = VK_NULL_HANDLE;
      VkDescriptorSet desc_set = VK_NULL_HANDLE;
      VkSampler sampler = VK_NULL_HANDLE;
    };

    compute_pipeline_t compute = {};

    // Command submission — bounded reuse verified by one fence per ring slot.
    static constexpr int CMD_RING_SIZE = 3;

    struct cmd_submission_t {
      VkCommandPool pool = VK_NULL_HANDLE;
      std::array<VkCommandBuffer, CMD_RING_SIZE> ring = {};
      std::array<VkFence, CMD_RING_SIZE> fences = {};
      int ring_idx = 0;
    };

    cmd_submission_t cmd = {};

    // Source DMA-BUF image with deferred destruction
    struct src_image_t {
      VkImage image = VK_NULL_HANDLE;
      VkDeviceMemory mem = VK_NULL_HANDLE;
      VkImageView view = VK_NULL_HANDLE;
    };

    src_image_t src = {};
    static constexpr int DEFER_RING_SIZE = 4;
    std::array<src_image_t, DEFER_RING_SIZE> defer_ring = {};
    int defer_idx = 0;

    // Target NV12 plane views
    struct target_state_t {
      VkImageView y_view = VK_NULL_HANDLE;
      VkImageView uv_view = VK_NULL_HANDLE;
      bool initialized = false;
    };

    std::unordered_map<VkImage, target_state_t> targets;
    target_state_t *target = nullptr;
    VkSemaphore external_signal_semaphore = VK_NULL_HANDLE;
    VkSemaphore external_wait_semaphore = VK_NULL_HANDLE;

    bool descriptors_dirty = false;

    // Cursor image
    struct {
      VkImage image = VK_NULL_HANDLE;
      VkDeviceMemory mem = VK_NULL_HANDLE;
      VkImageView view = VK_NULL_HANDLE;
      bool needs_transition = false;
    } cursor = {};

    unsigned long cursor_serial = 0;

    // Push constants (color matrix)
    PushConstants push = {};
  };

#if defined(__linux__)
  /**
   * @brief Linux native-AMF adapter sharing Sunshine's Vulkan conversion device.
   */
  class amf_vulkan_t final: public ::amf::amf_native {
  public:
    /**
     * @brief Construct a Vulkan AMF adapter for one capture region.
     *
     * @param capture_width Captured image width in pixels.
     * @param capture_height Captured image height in pixels.
     * @param capture_offset_x Horizontal capture offset in pixels.
     * @param capture_offset_y Vertical capture offset in pixels.
     */
    amf_vulkan_t(int capture_width, int capture_height, int capture_offset_x, int capture_offset_y):
        source_width(capture_width),
        source_height(capture_height),
        source_offset_x(capture_offset_x),
        source_offset_y(capture_offset_y),
        converter(std::make_unique<vk_vram_t>()) {
      converter->init(capture_width, capture_height, capture_offset_x, capture_offset_y);
    }

    /**
     * @brief Destroy AMF before releasing Vulkan frames and the shared device.
     */
    ~amf_vulkan_t() override {
      destroy_encoder();
      converter.reset();
      if (frames_ref) {
        av_buffer_unref(&frames_ref);
      }
      if (device_ref) {
        av_buffer_unref(&device_ref);
      }
    }

    /**
     * @brief Convert one DMA-BUF capture into the reserved AMF surface.
     *
     * @param image Captured KMS or PipeWire descriptor.
     * @return Zero on success.
     */
    int convert(platf::img_t &image) {
      ::amf::AMFContext1::AMFVulkanLocker vulkan_lock(shared_context);
      auto *target_frame = static_cast<AVFrame *>(acquire_input_surface_for_render());
      if (!target_frame) {
        return -1;
      }
      const auto slot = find_slot(target_frame);
      bool wait_for_completion = false;
      if (slot) {
        std::lock_guard lock(semaphore_state_mutex);
        wait_for_completion = ::amf::lifecycle::prepare_native_surface_producer(semaphore_submitted[*slot]);
      }
      if (!slot || converter->set_native_amf_target(
                     target_frame,
                     completion_semaphores[*slot],
                     wait_for_completion
                   ) != 0) {
        if (slot) {
          std::lock_guard lock(semaphore_state_mutex);
          ::amf::lifecycle::cancel_native_surface_producer(
            semaphore_submitted[*slot],
            wait_for_completion
          );
        }
        cancel_input_surface_for_render();
        return -1;
      }
      const auto result = converter->convert(image);
      if (result != 0) {
        converter->cancel_native_amf_target();
        std::lock_guard lock(semaphore_state_mutex);
        ::amf::lifecycle::cancel_native_surface_producer(
          semaphore_submitted[*slot],
          wait_for_completion
        );
        cancel_input_surface_for_render();
      }
      return result;
    }

    /**
     * @brief Apply the stream color transform to the Vulkan compute pass.
     *
     * @param colorspace Stream color matrix and range configuration.
     */
    void apply_colorspace(const video::sunshine_colorspace_t &colorspace) {
      converter->colorspace = colorspace;
      converter->apply_colorspace();
    }

  protected:
    bool load_amf_runtime(::amf::AMFFactory *&output_factory) override {
      if (runtime_module && output_factory) {
        return true;
      }
      runtime_module = dlopen(AMF_DLL_NAMEA, RTLD_NOW | RTLD_LOCAL);
      if (!runtime_module) {
        BOOST_LOG(info) << "AMF: Linux runtime " << AMF_DLL_NAMEA << " is unavailable: " << dlerror();
        return false;
      }
      auto query_version = reinterpret_cast<AMFQueryVersion_Fn>(dlsym(runtime_module, AMF_QUERY_VERSION_FUNCTION_NAME));
      auto initialize = reinterpret_cast<AMFInit_Fn>(dlsym(runtime_module, AMF_INIT_FUNCTION_NAME));
      if (!query_version || !initialize) {
        BOOST_LOG(error) << "AMF: Linux runtime is missing required entry points"sv;
        unload_amf_runtime();
        return false;
      }
      amf_uint64 version = 0;
      if (query_version(&version) != AMF_OK || initialize(AMF_FULL_VERSION, &output_factory) != AMF_OK || !output_factory) {
        BOOST_LOG(error) << "AMF: Linux runtime initialization failed"sv;
        unload_amf_runtime();
        return false;
      }
      BOOST_LOG(info) << "AMF Linux runtime version: "
                      << AMF_GET_MAJOR_VERSION(version) << '.'
                      << AMF_GET_MINOR_VERSION(version) << '.'
                      << AMF_GET_SUBMINOR_VERSION(version) << '.'
                      << AMF_GET_BUILD_VERSION(version);
      return true;
    }

    void unload_amf_runtime() noexcept override {
      vulkan_context_initialized = false;
      shared_context = nullptr;
      if (runtime_module) {
        dlclose(runtime_module);
        runtime_module = nullptr;
      }
    }

    AMF_RESULT initialize_platform_context(::amf::AMFContext *native_context) override {
      if (!native_context) {
        return AMF_INVALID_ARG;
      }
      ::amf::AMFContext1Ptr context1(native_context);
      if (!context1) {
        return AMF_NO_INTERFACE;
      }
      shared_context = context1;

      amf_size extension_count = 0;
      if (context1->GetVulkanDeviceExtensions(&extension_count, nullptr) != AMF_OK) {
        return AMF_VULKAN_FAILED;
      }
      std::vector<const char *> required_extensions(extension_count);
      if (extension_count > 0 &&
          context1->GetVulkanDeviceExtensions(&extension_count, required_extensions.data()) != AMF_OK) {
        return AMF_VULKAN_FAILED;
      }
      std::string extension_option;
      for (const auto *extension : required_extensions) {
        if (!extension || !*extension) {
          continue;
        }
        if (!extension_option.empty()) {
          extension_option += '+';
        }
        extension_option += extension;
      }
      if (create_vulkan_hwdevice(&device_ref, extension_option) != 0 || !device_ref) {
        BOOST_LOG(error) << "AMF: failed to create a Vulkan device with AMD's required extensions"sv;
        return AMF_NO_DEVICE;
      }

      auto *device_context = reinterpret_cast<AVHWDeviceContext *>(device_ref->data);
      auto *vulkan_context = reinterpret_cast<AVVulkanDeviceContext *>(device_context->hwctx);
      vulkan_device = {};
      vulkan_device.cbSizeof = sizeof(vulkan_device);
      vulkan_device.hInstance = vulkan_context->inst;
      vulkan_device.hPhysicalDevice = vulkan_context->phys_dev;
      vulkan_device.hDevice = vulkan_context->act_dev;
      const auto result = context1->InitVulkan(&vulkan_device);
      vulkan_context_initialized = result == AMF_OK;
      return result;
    }

    bool configure_platform_surfaces(platf::pix_fmt_e buffer_format, int bit_depth, int width, int height) override {
      if (!device_ref || width <= 0 || height <= 0) {
        return false;
      }
      encoded_width = width;
      encoded_height = height;
      ten_bit = buffer_format == platf::pix_fmt_e::p010 ||
                (buffer_format != platf::pix_fmt_e::nv12 && bit_depth == 10);
      target_format = ten_bit ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 :
                                VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
      frames_ref = av_hwframe_ctx_alloc(device_ref);
      if (!frames_ref) {
        return false;
      }
      auto *frames = reinterpret_cast<AVHWFramesContext *>(frames_ref->data);
      frames->format = AV_PIX_FMT_VULKAN;
      frames->sw_format = ten_bit ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;
      frames->width = width;
      frames->height = height;
      frames->initial_pool_size = 0;
      auto *vulkan_frames = reinterpret_cast<AVVulkanFramesContext *>(frames->hwctx);
      vulkan_frames->tiling = VK_IMAGE_TILING_OPTIMAL;
      vulkan_frames->usage = static_cast<VkImageUsageFlagBits>(
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT
      );
      vulkan_frames->flags = AV_VK_FRAME_FLAG_NONE;
      vulkan_frames->format[0] = target_format;
      ::amf::AMFContext1::AMFVulkanLocker vulkan_lock(shared_context);
      if (av_hwframe_ctx_init(frames_ref) < 0 || converter->initialize_for_native_amf(frames_ref) != 0) {
        BOOST_LOG(error) << "AMF: failed to initialize the shared Vulkan frame pool"sv;
        return false;
      }
      return true;
    }

    bool allocate_platform_surface(std::size_t slot_index) override {
      if (!frames_ref || slot_index >= input_frames.size()) {
        return false;
      }
      if (input_frames[slot_index]) {
        return true;
      }
      ::amf::AMFContext1::AMFVulkanLocker vulkan_lock(shared_context);
      frame_t frame {av_frame_alloc()};
      if (!frame || av_hwframe_get_buffer(frames_ref, frame.get(), 0) < 0) {
        return false;
      }
      auto *vulkan_frame = reinterpret_cast<AVVkFrame *>(frame->data[0]);
      if (!vulkan_frame || !vulkan_frame->img[0] || vulkan_frame->img[1] || vulkan_frame->mem[1]) {
        BOOST_LOG(error) << "AMF: Vulkan frame pool did not provide one contiguous multiplane image"sv;
        return false;
      }
      VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      if (vkCreateSemaphore(vulkan_device.hDevice, &semaphore_info, nullptr, &completion_semaphores[slot_index]) != VK_SUCCESS) {
        return false;
      }
      input_frames[slot_index] = std::move(frame);
      return true;
    }

    void clear_platform_surfaces() noexcept override {
      std::unique_ptr<::amf::AMFContext1::AMFVulkanLocker> vulkan_lock;
      if (vulkan_context_initialized && shared_context) {
        vulkan_lock = std::make_unique<::amf::AMFContext1::AMFVulkanLocker>(shared_context);
      }
      if (vulkan_device.hDevice) {
        vkDeviceWaitIdle(vulkan_device.hDevice);
      }
      if (converter) {
        converter->release_native_amf_targets();
      }
      for (std::size_t index = 0; index < input_frames.size(); ++index) {
        input_frames[index].reset();
        if (completion_semaphores[index] && vulkan_device.hDevice) {
          vkDestroySemaphore(vulkan_device.hDevice, completion_semaphores[index], nullptr);
        }
        completion_semaphores[index] = VK_NULL_HANDLE;
        native_surfaces[index] = {};
      }
      {
        std::lock_guard lock(semaphore_state_mutex);
        semaphore_submitted.fill(false);
        surface_release_reconciled.fill(false);
        source_signal_consumed_before_release.fill(false);
      }
      if (frames_ref) {
        av_buffer_unref(&frames_ref);
      }
    }

    void *platform_surface(std::size_t slot_index) noexcept override {
      return slot_index < input_frames.size() ? input_frames[slot_index].get() : nullptr;
    }

    bool copy_platform_surface(std::size_t destination_slot, std::size_t source_slot) override {
      if (destination_slot >= input_frames.size() || source_slot >= input_frames.size() ||
          !input_frames[destination_slot] || !input_frames[source_slot]) {
        return false;
      }
      ::amf::AMFContext1::AMFVulkanLocker vulkan_lock(shared_context);
      // A released AMF semaphore may be signaled. Consume it before the copy
      // producer signals the destination again. The source wait also serializes
      // this read when AMF still owns the most recently rendered surface.
      bool consume_destination_signal = false;
      {
        std::lock_guard lock(semaphore_state_mutex);
        consume_destination_signal =
          ::amf::lifecycle::consume_native_surface_signal(semaphore_submitted[destination_slot]);
      }
      if (consume_destination_signal) {
        if (!consume_binary_semaphore(completion_semaphores[destination_slot])) {
          std::lock_guard lock(semaphore_state_mutex);
          semaphore_submitted[destination_slot] = true;
          return false;
        }
      }
      bool consume_source_signal = false;
      {
        std::lock_guard lock(semaphore_state_mutex);
        consume_source_signal =
          ::amf::lifecycle::consume_native_surface_signal(semaphore_submitted[source_slot]);
      }
      if (consume_source_signal) {
        if (!consume_binary_semaphore(completion_semaphores[source_slot])) {
          std::lock_guard lock(semaphore_state_mutex);
          semaphore_submitted[source_slot] = true;
          return false;
        }
        std::lock_guard lock(semaphore_state_mutex);
        // The queue wait above consumes the signal even when AMF's release
        // callback raced ahead and had already reported it as submitted.
        ::amf::lifecycle::mark_native_surface_signal_consumed(
          semaphore_submitted[source_slot],
          surface_release_reconciled[source_slot],
          source_signal_consumed_before_release[source_slot]
        );
      }
      const auto result = converter->copy_native_amf_frame(
        input_frames[destination_slot].get(),
        input_frames[source_slot].get(),
        completion_semaphores[destination_slot]
      );
      {
        std::lock_guard lock(semaphore_state_mutex);
        semaphore_submitted[destination_slot] = result == 0;
      }
      return result == 0;
    }

    AMF_RESULT create_platform_amf_surface(
      ::amf::AMFContext *native_context,
      std::size_t slot_index,
      ::amf::AMFSurface **output_surface,
      ::amf::AMFSurfaceObserver *observer
    ) override {
      if (!native_context || !output_surface || slot_index >= input_frames.size() || !input_frames[slot_index]) {
        return AMF_INVALID_ARG;
      }
      auto *frame = input_frames[slot_index].get();
      auto *vulkan_frame = reinterpret_cast<AVVkFrame *>(frame->data[0]);
      auto &surface = native_surfaces[slot_index];
      surface = {};
      surface.cbSizeof = sizeof(surface);
      surface.hImage = vulkan_frame->img[0];
      surface.hMemory = vulkan_frame->mem[0];
      surface.iSize = static_cast<amf_int64>(vulkan_frame->size[0]);
      surface.eFormat = target_format;
      surface.iWidth = encoded_width;
      surface.iHeight = encoded_height;
      surface.eCurrentLayout = vulkan_frame->layout[0];
      surface.eUsage = ::amf::AMF_SURFACE_USAGE_DEFAULT;
      surface.eAccess = ::amf::AMF_MEMORY_CPU_LOCAL;
      surface.Sync.cbSizeof = sizeof(surface.Sync);
      surface.Sync.hSemaphore = completion_semaphores[slot_index];
      {
        std::lock_guard lock(semaphore_state_mutex);
        surface.Sync.bSubmitted = semaphore_submitted[slot_index];
        ::amf::lifecycle::begin_native_surface_consumer(
          surface_release_reconciled[slot_index],
          source_signal_consumed_before_release[slot_index]
        );
      }
      ::amf::AMFContext1Ptr context1(native_context);
      return context1 ? context1->CreateSurfaceFromVulkanNative(&surface, output_surface, observer) : AMF_NO_INTERFACE;
    }

    void on_platform_surface_released(
      std::size_t slot_index,
      ::amf::AMFSurface *surface
    ) noexcept override {
      if (slot_index >= input_frames.size() || !input_frames[slot_index] || !surface) {
        return;
      }
      auto *plane = surface->GetPlaneAt(0);
      auto *view = plane ? reinterpret_cast<::amf::AMFVulkanView *>(plane->GetNative()) : nullptr;
      auto *released = view ? view->pSurface : nullptr;
      auto *vulkan_frame = reinterpret_cast<AVVkFrame *>(input_frames[slot_index]->data[0]);
      const bool consumer_submitted = released ?
                                        released->Sync.bSubmitted :
                                        native_surfaces[slot_index].Sync.bSubmitted;
      if (released) {
        vulkan_frame->layout[0] = static_cast<VkImageLayout>(released->eCurrentLayout);
        vulkan_frame->access[0] = VK_ACCESS_MEMORY_READ_BIT;
      }
      {
        std::lock_guard lock(semaphore_state_mutex);
        // A repeat-copy may already have queued a wait that consumes AMF's
        // returned signal. Do not resurrect that signal from a racing callback.
        ::amf::lifecycle::reconcile_native_surface_release(
          semaphore_submitted[slot_index],
          surface_release_reconciled[slot_index],
          source_signal_consumed_before_release[slot_index],
          consumer_submitted
        );
      }
    }

    bool platform_device_failed(const char *operation) override {
      if (!vulkan_device.hDevice) {
        BOOST_LOG(error) << "AMF: Vulkan device missing during " << operation;
        return true;
      }
      return false;
    }

  private:
    std::optional<std::size_t> find_slot(const AVFrame *frame) const {
      for (std::size_t index = 0; index < input_frames.size(); ++index) {
        if (input_frames[index].get() == frame) {
          return index;
        }
      }
      return std::nullopt;
    }

    bool consume_binary_semaphore(VkSemaphore semaphore) {
      if (!semaphore || !vulkan_device.hDevice) {
        return false;
      }
      auto *device_context = reinterpret_cast<AVHWDeviceContext *>(device_ref->data);
      auto *vulkan_context = reinterpret_cast<AVVulkanDeviceContext *>(device_context->hwctx);
      VkQueue queue = VK_NULL_HANDLE;
      for (int index = 0; index < vulkan_context->nb_qf; ++index) {
        if (vulkan_context->qf[index].flags & VK_QUEUE_COMPUTE_BIT) {
          vkGetDeviceQueue(vulkan_device.hDevice, vulkan_context->qf[index].idx, 0, &queue);
          break;
        }
      }
      if (!queue) {
        return false;
      }
      VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.waitSemaphoreCount = 1;
      submit.pWaitSemaphores = &semaphore;
      submit.pWaitDstStageMask = &stage;
      return vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
    }

    int source_width = 0;
    int source_height = 0;
    int source_offset_x = 0;
    int source_offset_y = 0;
    int encoded_width = 0;
    int encoded_height = 0;
    bool ten_bit = false;
    bool vulkan_context_initialized = false;
    VkFormat target_format = VK_FORMAT_UNDEFINED;
    void *runtime_module = nullptr;
    AVBufferRef *device_ref = nullptr;
    AVBufferRef *frames_ref = nullptr;
    ::amf::AMFVulkanDevice vulkan_device {};
    ::amf::AMFContext1Ptr shared_context;
    std::unique_ptr<vk_vram_t> converter;
    std::array<frame_t, ::amf::lifecycle::maximum_input_surface_count> input_frames;
    std::array<VkSemaphore, ::amf::lifecycle::maximum_input_surface_count> completion_semaphores {};
    std::mutex semaphore_state_mutex;
    std::array<bool, ::amf::lifecycle::maximum_input_surface_count> semaphore_submitted {};
    std::array<bool, ::amf::lifecycle::maximum_input_surface_count> surface_release_reconciled {};
    std::array<bool, ::amf::lifecycle::maximum_input_surface_count> source_signal_consumed_before_release {};
    std::array<::amf::AMFVulkanSurface, ::amf::lifecycle::maximum_input_surface_count> native_surfaces {};
  };

  namespace {
    std::atomic<int> active_native_amf_encoders {0};
  }  // namespace

  /**
   * @brief Platform encode device joining Vulkan capture conversion to native AMF.
   */
  class vulkan_amf_encode_device_t final: public platf::amf_encode_device_t {
  public:
    /**
     * @brief Construct the shared Vulkan/AMF device.
     *
     * @param width Captured image width in pixels.
     * @param height Captured image height in pixels.
     * @param offset_x Horizontal capture offset in pixels.
     * @param offset_y Vertical capture offset in pixels.
     * @param format Sunshine pixel format to convert into.
     */
    vulkan_amf_encode_device_t(
      int width,
      int height,
      int offset_x,
      int offset_y,
      platf::pix_fmt_e format
    ):
        input_format(format),
        native_encoder(std::make_unique<amf_vulkan_t>(width, height, offset_x, offset_y)) {
      amf = native_encoder.get();
    }

    /**
     * @brief Release native AMF and active-session bookkeeping.
     */
    ~vulkan_amf_encode_device_t() override {
      amf = nullptr;
      native_encoder.reset();
      if (registered) {
        const auto remaining = active_native_amf_encoders.fetch_sub(1, std::memory_order_acq_rel) - 1;
        BOOST_LOG(info) << "AMF: Linux native encoder session closed (active=" << remaining << ')';
      }
    }

    bool init_encoder(
      const video::config_t &client_config,
      const video::sunshine_colorspace_t &stream_colorspace
    ) override {
      if (!native_encoder) {
        return false;
      }
      auto native_config = ::amf::make_amf_config(client_config);
      const auto active = active_native_amf_encoders.fetch_add(1, std::memory_order_acq_rel) + 1;
      registered = true;
      if (active > 1 &&
          ((native_config.lowlatency_mode && *native_config.lowlatency_mode) ||
           (native_config.high_motion_quality_boost_enable && *native_config.high_motion_quality_boost_enable))) {
        BOOST_LOG(error) << "AMF: unsafe low-latency/high-motion override requested with concurrent native sessions"sv;
        active_native_amf_encoders.fetch_sub(1, std::memory_order_acq_rel);
        registered = false;
        return false;
      }
      BOOST_LOG(info) << "AMF: creating Linux Vulkan-native encoder session "
                      << client_config.width << 'x' << client_config.height << '@'
                      << client_config.framerate << " codec=" << client_config.videoFormat
                      << " bitrate=" << client_config.bitrate << "kbps (active=" << active << ')';
      if (!native_encoder->create_encoder(native_config, client_config, stream_colorspace, input_format)) {
        active_native_amf_encoders.fetch_sub(1, std::memory_order_acq_rel);
        registered = false;
        return false;
      }
      native_encoder->apply_colorspace(stream_colorspace);
      return true;
    }

    int convert(platf::img_t &image) override {
      return native_encoder ? native_encoder->convert(image) : -1;
    }

  private:
    platf::pix_fmt_e input_format = platf::pix_fmt_e::unknown;
    std::unique_ptr<amf_vulkan_t> native_encoder;
    bool registered = false;
  };
#endif

  // Free functions

  int vulkan_init_avcodec_hardware_input_buffer(platf::avcodec_encode_device_t *, AVBufferRef **hw_device_buf) {
    return create_vulkan_hwdevice(hw_device_buf);
  }

  bool validate() {
    if (!avcodec_find_encoder_by_name("h264_vulkan") && !avcodec_find_encoder_by_name("hevc_vulkan")) {
      return false;
    }
    AVBufferRef *dev = nullptr;
    if (create_vulkan_hwdevice(&dev) < 0) {
      return false;
    }
    av_buffer_unref(&dev);
    return true;
  }

  std::unique_ptr<platf::avcodec_encode_device_t> make_avcodec_encode_device_vram(int w, int h, int offset_x, int offset_y) {
    auto dev = std::make_unique<vk_vram_t>();
    if (dev->init(w, h, offset_x, offset_y) < 0) {
      return nullptr;
    }
    return dev;
  }

#if defined(__linux__)
  std::unique_ptr<platf::amf_encode_device_t>
    make_amf_encode_device_vram(int width, int height, int offset_x, int offset_y, platf::pix_fmt_e format) {
    if (format != platf::pix_fmt_e::nv12 && format != platf::pix_fmt_e::p010) {
      return nullptr;
    }
    return std::make_unique<vulkan_amf_encode_device_t>(width, height, offset_x, offset_y, format);
  }
#endif

  std::unique_ptr<platf::avcodec_encode_device_t> make_avcodec_encode_device_ram(int, int) {
    return nullptr;
  }

}  // namespace vk
