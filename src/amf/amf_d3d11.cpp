/**
 * @file src/amf/amf_d3d11.cpp
 * @brief Implementation of the Windows D3D11 native AMF adapter.
 */

#include "amf_d3d11.h"

#ifndef DOXYGEN
  #include "src/logging.h"
#endif
#include "src/utility.h"

#include <AMF/core/Factory.h>
#include <iomanip>

namespace amf {

  /// Function pointer type for the AMF runtime initialization entry point.
  using AMFInit_Fn = AMF_RESULT(AMF_CDECL_CALL *)(amf_uint64 version, ::amf::AMFFactory **factory);
  /// Function pointer type for the AMF runtime version-query entry point.
  using AMFQueryVersion_Fn = AMF_RESULT(AMF_CDECL_CALL *)(amf_uint64 *version);

  amf_d3d11::amf_d3d11(ID3D11Device *d3d_device):
      device(d3d_device) {
  }

  amf_d3d11::~amf_d3d11() {
    destroy_encoder();
  }

  bool
    amf_d3d11::load_amf_runtime(::amf::AMFFactory *&output_factory) {
    if (runtime_module && output_factory) {
      return true;
    }

    runtime_module = LoadLibraryA(AMF_DLL_NAMEA);
    if (!runtime_module) {
      BOOST_LOG(error) << "AMF: failed to load " << AMF_DLL_NAMEA;
      return false;
    }

    const auto query_version = reinterpret_cast<AMFQueryVersion_Fn>(GetProcAddress(runtime_module, AMF_QUERY_VERSION_FUNCTION_NAME));
    const auto initialize = reinterpret_cast<AMFInit_Fn>(GetProcAddress(runtime_module, AMF_INIT_FUNCTION_NAME));
    if (!query_version || !initialize) {
      BOOST_LOG(error) << "AMF: missing runtime entry points in " << AMF_DLL_NAMEA;
      unload_amf_runtime();
      return false;
    }

    amf_uint64 version = 0;
    if (query_version(&version) != AMF_OK) {
      BOOST_LOG(error) << "AMF: failed to query runtime version";
      unload_amf_runtime();
      return false;
    }
    BOOST_LOG(info) << "AMF runtime version: "
                    << AMF_GET_MAJOR_VERSION(version) << '.'
                    << AMF_GET_MINOR_VERSION(version) << '.'
                    << AMF_GET_SUBMINOR_VERSION(version) << '.'
                    << AMF_GET_BUILD_VERSION(version);

    if (initialize(AMF_FULL_VERSION, &output_factory) != AMF_OK || !output_factory) {
      BOOST_LOG(error) << "AMF: AMFInit failed";
      unload_amf_runtime();
      return false;
    }
    return true;
  }

  void
    amf_d3d11::unload_amf_runtime() noexcept {
    if (runtime_module) {
      FreeLibrary(runtime_module);
      runtime_module = nullptr;
    }
  }

  AMF_RESULT
  amf_d3d11::initialize_platform_context(::amf::AMFContext *native_context) {
    return native_context && device ? native_context->InitDX11(device, AMF_DX11_1) : AMF_INVALID_ARG;
  }

  bool
    amf_d3d11::configure_platform_surfaces(platf::pix_fmt_e buffer_format, int bit_depth, int width, int height) {
    DXGI_FORMAT format = DXGI_FORMAT_NV12;
    if (buffer_format == platf::pix_fmt_e::p010 ||
        (buffer_format != platf::pix_fmt_e::nv12 && bit_depth == 10)) {
      format = DXGI_FORMAT_P010;
    }

    input_surface_desc = {};
    input_surface_desc.Width = static_cast<UINT>(width);
    input_surface_desc.Height = static_cast<UINT>(height);
    input_surface_desc.MipLevels = 1;
    input_surface_desc.ArraySize = 1;
    input_surface_desc.Format = format;
    input_surface_desc.SampleDesc.Count = 1;
    input_surface_desc.Usage = D3D11_USAGE_DEFAULT;
    input_surface_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    return device && width > 0 && height > 0;
  }

  bool
    amf_d3d11::allocate_platform_surface(std::size_t slot_index) {
    if (!device || slot_index >= input_textures.size()) {
      return false;
    }
    auto &texture = input_textures[slot_index];
    if (texture) {
      return true;
    }

    const auto result = device->CreateTexture2D(&input_surface_desc, nullptr, texture.ReleaseAndGetAddressOf());
    if (FAILED(result)) {
      BOOST_LOG(error) << "AMF: failed to create D3D11 input texture " << slot_index
                       << ", HRESULT: 0x" << std::hex << result;
      return false;
    }

    static const GUID texture_array_index_guid = {0x28115527, 0xe7c3, 0x4b66, {0x99, 0xd3, 0x4f, 0x2a, 0xe6, 0xb4, 0x7f, 0xaf}};
    const int array_index = 0;
    texture->SetPrivateData(texture_array_index_guid, sizeof(array_index), &array_index);
    return true;
  }

  void
    amf_d3d11::clear_platform_surfaces() noexcept {
    for (auto &texture : input_textures) {
      texture.Reset();
    }
    input_surface_desc = {};
  }

  void *
    amf_d3d11::platform_surface(std::size_t slot_index) noexcept {
    return slot_index < input_textures.size() ? input_textures[slot_index].Get() : nullptr;
  }

  bool
    amf_d3d11::copy_platform_surface(std::size_t destination_slot, std::size_t source_slot) {
    if (!device || destination_slot >= input_textures.size() || source_slot >= input_textures.size() ||
        !input_textures[destination_slot] || !input_textures[source_slot]) {
      return false;
    }
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> immediate_context;
    device->GetImmediateContext(immediate_context.GetAddressOf());
    if (!immediate_context) {
      return false;
    }
    immediate_context->CopyResource(input_textures[destination_slot].Get(), input_textures[source_slot].Get());
    return !platform_device_failed("repeated input copy");
  }

  AMF_RESULT
  amf_d3d11::create_platform_amf_surface(::amf::AMFContext *native_context, std::size_t slot_index, ::amf::AMFSurface **output_surface, ::amf::AMFSurfaceObserver *observer) {
    if (!native_context || !output_surface || slot_index >= input_textures.size() || !input_textures[slot_index]) {
      return AMF_INVALID_ARG;
    }
    return native_context->CreateSurfaceFromDX11Native(input_textures[slot_index].Get(), output_surface, observer);
  }

  bool
    amf_d3d11::platform_device_failed(const char *operation) {
    if (!device) {
      BOOST_LOG(error) << "AMF: D3D11 device missing during " << operation;
      return true;
    }
    const auto removed_reason = device->GetDeviceRemovedReason();
    if (removed_reason == S_OK) {
      return false;
    }
    BOOST_LOG(error) << "AMF: D3D11 device lost during " << operation
                     << ", reason: 0x" << util::hex(removed_reason).to_string_view();
    return true;
  }

  void
    amf_d3d11::configure_output_thread() noexcept {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
  }

  ID3D11Texture2D *
    amf_d3d11::acquire_input_texture_for_render() {
    return static_cast<ID3D11Texture2D *>(acquire_input_surface_for_render());
  }

  void
    amf_d3d11::cancel_input_texture_for_render() {
    cancel_input_surface_for_render();
  }

  std::unique_ptr<amf_d3d11>
    create_amf_d3d11(ID3D11Device *d3d_device) {
    return d3d_device ? std::make_unique<amf_d3d11>(d3d_device) : nullptr;
  }

}  // namespace amf
