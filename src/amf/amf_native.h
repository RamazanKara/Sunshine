/**
 * @file src/amf/amf_native.h
 * @brief Declarations for the platform-neutral native AMF encoder core.
 */
#pragma once

#include "amf_encoder.h"
#include "amf_lifecycle.h"

#include <AMF/components/Component.h>
#include <AMF/core/Context.h>
#include <AMF/core/Data.h>
#include <AMF/core/Factory.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

namespace amf {

  /**
   * @brief AMF encoder core using platform-provided native input surfaces.
   *
   * Platform adapters own device creation, native surface allocation, copies,
   * synchronization, and AMF surface wrapping. Codec configuration, scheduling,
   * recovery, and output processing remain identical across platforms.
   */
  class amf_native: public amf_encoder {
  public:
    /**
     * @brief Construct the platform-neutral native AMF encoder state.
     */
    amf_native();

    /**
     * @brief Destroy the AMF encoder and release driver resources.
     */
    ~amf_native() override = default;

    bool
      create_encoder(const amf_config &config, const video::config_t &client_config, const video::sunshine_colorspace_t &colorspace, platf::pix_fmt_e buffer_format) override;

    void
      destroy_encoder() override;

    amf_encode_result
      encode_frame(uint64_t frame_index, bool force_idr) override;

    amf_encode_result
      drain_output(std::chrono::milliseconds timeout) override;

    bool
      begin_drain() override;

    bool
      invalidate_ref_frames(uint64_t first_frame, uint64_t last_frame) override;

    bool
      set_bitrate(int bitrate_kbps) override;

    bool
      set_hdr_metadata(const std::optional<amf_hdr_metadata> &metadata) override;

    void *
      get_input_texture() override;

    /**
     * @brief Reserve a free native texture for the next conversion.
     *
     * @return Texture that the display converter may render into, or nullptr on backpressure.
     */
    void *
      acquire_input_surface_for_render();

    /**
     * @brief Cancel a texture reservation after conversion fails.
     */
    void
      cancel_input_surface_for_render();

  protected:
    /**
     * @brief Load the platform AMF runtime and return its factory.
     *
     * @param output_factory Receives the initialized AMF factory.
     * @return True when the runtime and required entry points are available.
     */
    virtual bool
      load_amf_runtime(::amf::AMFFactory *&output_factory) = 0;

    /**
     * @brief Release the platform AMF runtime module.
     */
    virtual void
      unload_amf_runtime() noexcept = 0;

    /**
     * @brief Attach the platform graphics device to a new AMF context.
     *
     * @param native_context AMF context to initialize.
     * @return AMF operation result.
     */
    virtual AMF_RESULT
      initialize_platform_context(::amf::AMFContext *native_context) = 0;

    /**
     * @brief Configure platform input-surface allocation for a stream.
     *
     * @param buffer_format Sunshine input format.
     * @param bit_depth Negotiated bit depth.
     * @param width Encoded width in pixels.
     * @param height Encoded height in pixels.
     * @return True when the requested platform format is supported.
     */
    virtual bool
      configure_platform_surfaces(platf::pix_fmt_e buffer_format, int bit_depth, int width, int height) = 0;

    /**
     * @brief Lazily allocate one platform-native input surface.
     *
     * @param slot_index Surface-pool slot.
     * @return True when the slot owns a usable surface.
     */
    virtual bool
      allocate_platform_surface(std::size_t slot_index) = 0;

    /**
     * @brief Release every platform-native input surface.
     */
    virtual void
      clear_platform_surfaces() noexcept = 0;

    /**
     * @brief Return a platform-native render target for a surface slot.
     *
     * @param slot_index Surface-pool slot.
     * @return Opaque platform render target, or nullptr.
     */
    virtual void *
      platform_surface(std::size_t slot_index) noexcept = 0;

    /**
     * @brief Duplicate a previously rendered surface for a repeated frame.
     *
     * @param destination_slot Destination slot.
     * @param source_slot Source slot.
     * @return True when the GPU copy was queued successfully.
     */
    virtual bool
      copy_platform_surface(std::size_t destination_slot, std::size_t source_slot) = 0;

    /**
     * @brief Wrap a platform-native input surface for AMF submission.
     *
     * @param native_context Active AMF context.
     * @param slot_index Surface-pool slot.
     * @param output_surface Receives the AMF surface wrapper.
     * @param observer Observer that recycles the slot after AMF releases it.
     * @return AMF operation result.
     */
    virtual AMF_RESULT
      create_platform_amf_surface(::amf::AMFContext *native_context, std::size_t slot_index, ::amf::AMFSurface **output_surface, ::amf::AMFSurfaceObserver *observer) = 0;

    /**
     * @brief Reconcile platform synchronization after AMF releases a surface.
     *
     * @param slot_index Released surface-pool slot.
     * @param surface Surface whose native synchronization state AMF has updated.
     */
    virtual void
      on_platform_surface_released(std::size_t slot_index, ::amf::AMFSurface *surface) noexcept {
      (void) slot_index;
      (void) surface;
    }

    /**
     * @brief Check and log a platform graphics-device failure.
     *
     * @param operation Operation being diagnosed.
     * @return True when the platform device is unusable.
     */
    virtual bool
      platform_device_failed(const char *operation) = 0;

    /**
     * @brief Apply platform-specific output-thread scheduling policy.
     */
    virtual void
      configure_output_thread() noexcept {
    }

  private:
    /**
     * @brief Apply codec-independent and codec-specific AMF properties.
     *
     * @param config Native AMF settings.
     * @param client_config Negotiated stream configuration.
     * @param colorspace Output colorimetry.
     * @return True when all mandatory properties were accepted and verified.
     */
    bool
      configure_encoder(const amf_config &config, const video::config_t &client_config, const video::sunshine_colorspace_t &colorspace);

    /**
     * @brief Map Sunshine's input format to an AMF surface format.
     *
     * @param buffer_format Sunshine pixel format.
     * @param bit_depth Stream bit depth.
     * @return AMF surface format.
     */
    AMF_SURFACE_FORMAT
    get_amf_format(platf::pix_fmt_e buffer_format, int bit_depth);

    /**
     * @brief Select the AMF component identifier for the negotiated codec.
     *
     * @return AMF component identifier.
     */
    const wchar_t *
      get_codec_id();

    /**
     * @brief Copy one AMF output object into Sunshine-owned frame storage.
     *
     * @param output_data AMF output object.
     * @return Encoded frame and metadata.
     */
    amf_encoded_frame
      extract_encoded_frame(const ::amf::AMFDataPtr &output_data);

    /**
     * @brief Retrieve asynchronous AMF output until stopped or drained.
     *
     * @param stop_token Cooperative stop token.
     */
    void
      output_pump(std::stop_token stop_token) noexcept;

    /**
     * @brief Handle AMF releasing one input-surface slot.
     *
     * @param slot_index Released ring index.
     */
    void
      on_input_surface_released(std::size_t slot_index) noexcept;

    /**
     * @brief Lazily allocate direct-render surfaces up to the requested pool size.
     *
     * @param count Number of input-surface slots that must be available.
     * @return True when every requested slot is allocated.
     */
    bool
      ensure_input_surface_count(std::size_t count);

    ::amf::AMFFactory *factory = nullptr;
    ::amf::AMFContextPtr context;
    ::amf::AMFComponentPtr encoder;

    // The converter renders directly into a reserved pool surface. AMF may retain an
    // accepted native surface while encoding it, so a slot is recycled exclusively by
    // AMFSurfaceObserver. The pool starts at the queue/lookahead-aware working depth
    // and allocates additional slots lazily during a transient driver backlog.
    static constexpr std::size_t INPUT_SURFACE_RING_SIZE = lifecycle::maximum_input_surface_count;
    using input_surface_state_e = lifecycle::input_surface_state_e;

    struct input_surface_release_observer_t final: ::amf::AMFSurfaceObserver {
      amf_native *owner = nullptr;
      std::size_t slot_index = 0;

      void AMF_STD_CALL OnSurfaceDataRelease(::amf::AMFSurface *surface) override;
    };

    struct input_surface_slot_t: lifecycle::input_surface_state_t {};

    std::array<input_surface_slot_t, INPUT_SURFACE_RING_SIZE> input_surface_ring;
    std::array<input_surface_release_observer_t, INPUT_SURFACE_RING_SIZE> input_surface_release_observers;
    std::size_t active_input_surface_count = lifecycle::minimum_input_surface_count;  ///< Allocated pool prefix.
    std::size_t encoder_input_queue_size = lifecycle::default_amf_input_queue_size;  ///< Applied AMF queue depth.
    std::size_t next_input_surface_slot = 0;
    std::optional<std::size_t> prepared_input_surface_slot;
    std::optional<std::size_t> last_rendered_input_surface_slot;

    // Encoder state
    video::config_t current_config {};
    int video_format = 0;  // 0=H264, 1=HEVC, 2=AV1
    AMF_SURFACE_FORMAT surface_format = AMF_SURFACE_NV12;
    int encode_width = 0;
    int encode_height = 0;
    bool rfi_pending = false;
    uint64_t last_rfi_ltr_index = 0;
    int max_ltr_frames = 0;
    bool rfi_enabled = false;

    // Current LTR state for RFI.
    // Slot 0 is reserved as the IDR baseline (set on every IDR, never overwritten by
    // periodic marks) so RFI always has a known-good fallback even when every recent
    // periodic-marked frame was inside a packet-loss window. Slots 1..N-1 form a
    // sliding window of more recent anchors.
    static constexpr int MAX_LTR_SLOTS = 4;
    static constexpr uint64_t LTR_MARK_INTERVAL = 4;  // Mark LTR every N frames
    int effective_ltr_slots = 0;  // Clamped to min(max_ltr_frames, MAX_LTR_SLOTS)
    int current_ltr_slot = 0;  // Which LTR slot to mark next
    std::array<bool, MAX_LTR_SLOTS> ltr_slots_valid {};
    std::array<uint64_t, MAX_LTR_SLOTS> ltr_slot_frame_index {};  // Frame index when each LTR slot was marked

    // QueryOutput is owned by a dedicated pump. This keeps AMF driver completion work
    // off Sunshine's latency-critical encode thread and gives input-slot ownership one
    // synchronization boundary.
    std::mutex state_mutex;
    std::condition_variable state_cv;
    std::jthread output_thread;
    std::deque<amf_encoded_frame> completed_outputs;
    std::unordered_map<uint64_t, bool> frame_rfi_flags;
    uint64_t last_completed_frame_index = 0;
    uint64_t last_submitted_frame_index = 0;
    bool output_fatal = false;
    bool drain_requested = false;
    bool drain_complete = false;
    bool output_poll_requested = false;  ///< Whether the output thread has an active polling lease.
    std::size_t active_output_poll_waiters = 0;  ///< Encode calls waiting for bounded output progress.

    // Input ownership and output production are deliberately tracked separately.
    // AMF explicitly does not guarantee a one-to-one input/output relationship;
    // AMFSurfaceObserver owns the first count while QueryOutput owns the totals.
    std::size_t input_surfaces_in_flight = 0;  ///< Native textures still owned by AMF.
    uint64_t accepted_input_count = 0;  ///< Monotonic successful SubmitInput count.
    uint64_t completed_output_count = 0;  ///< Monotonic valid QueryOutput count.
    bool preanalysis_enabled = false;
    int preanalysis_lookahead_depth = 0;
    bool query_timeout_supported = false;  ///< Whether QueryOutput honors AMF's timeout property.
    bool user_configured_rate_control = false;
    bool enforce_hrd_enabled = false;

    // Statistics feedback state
    bool statistics_enabled = false;
    bool psnr_enabled = false;
    bool ssim_enabled = false;

    // Runtime fault watchdog: count consecutive failures so we can signal
    // a fatal error to the upper layer (triggering a real reinit) instead
    // of silently producing no output forever. Threshold is derived from
    // client framerate in create_encoder() so the watchdog fires after
    // roughly the same wall-clock time regardless of fps.
    int consecutive_submit_failures = 0;
    int consecutive_surface_failures = 0;
    int consecutive_query_failures = 0;
    int consecutive_output_failures = 0;  ///< Consecutive invalid output objects.
    // Consecutive submissions whose bounded coalescing target was not reached.
    int consecutive_catchup_misses = 0;
    uint64_t catchup_batch_count = 0;  ///< Multi-frame batches observed for rate-limited diagnostics.
    int max_consecutive_failures = 60;  // Set to ~1s of frames in create_encoder()
    std::chrono::steady_clock::time_point last_output_progress {};
    std::chrono::steady_clock::time_point submit_backpressure_started {};  ///< Start of the current retry exhaustion sequence.

    std::string last_error_string;
  };

}  // namespace amf
