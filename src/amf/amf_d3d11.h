/**
 * @file src/amf/amf_d3d11.h
 * @brief Declarations for the Windows D3D11 native AMF adapter.
 */
#pragma once

#include "amf_native.h"

#include <array>
#include <d3d11.h>
#include <wrl/client.h>

namespace amf {

  /**
   * @brief Native AMF encoder using D3D11 conversion targets.
   */
  class amf_d3d11 final: public amf_native {
  public:
    /**
     * @brief Construct a D3D11 AMF adapter.
     *
     * @param d3d_device D3D11 device used for conversion and encoding.
     */
    explicit amf_d3d11(ID3D11Device *d3d_device);

    /**
     * @brief Destroy AMF before releasing the D3D11 adapter.
     */
    ~amf_d3d11() override;

    /**
     * @brief Reserve a D3D11 texture for the next conversion.
     *
     * @return Reserved NV12/P010 texture, or nullptr on bounded backpressure.
     */
    ID3D11Texture2D *
      acquire_input_texture_for_render();

    /**
     * @brief Cancel a reserved D3D11 conversion target.
     */
    void
      cancel_input_texture_for_render();

  protected:
    bool
      load_amf_runtime(::amf::AMFFactory *&output_factory) override;

    void
      unload_amf_runtime() noexcept override;

    AMF_RESULT
    initialize_platform_context(::amf::AMFContext *native_context) override;

    bool
      configure_platform_surfaces(platf::pix_fmt_e buffer_format, int bit_depth, int width, int height) override;

    bool
      allocate_platform_surface(std::size_t slot_index) override;

    void
      clear_platform_surfaces() noexcept override;

    void *
      platform_surface(std::size_t slot_index) noexcept override;

    bool
      copy_platform_surface(std::size_t destination_slot, std::size_t source_slot) override;

    AMF_RESULT
    create_platform_amf_surface(::amf::AMFContext *native_context, std::size_t slot_index, ::amf::AMFSurface **output_surface, ::amf::AMFSurfaceObserver *observer) override;

    bool
      platform_device_failed(const char *operation) override;

    void
      configure_output_thread() noexcept override;

  private:
    ID3D11Device *device = nullptr;  ///< Non-owning conversion device.
    HMODULE runtime_module = nullptr;  ///< Dynamically loaded AMF runtime.
    D3D11_TEXTURE2D_DESC input_surface_desc {};  ///< Lazy pool texture description.
    std::array<Microsoft::WRL::ComPtr<ID3D11Texture2D>, lifecycle::maximum_input_surface_count> input_textures;
  };

  /**
   * @brief Create a D3D11 native AMF encoder.
   *
   * @param d3d_device D3D11 device used for conversion and encoding.
   * @return Encoder instance, or nullptr for an invalid device.
   */
  std::unique_ptr<amf_d3d11>
    create_amf_d3d11(ID3D11Device *d3d_device);

}  // namespace amf
