/**
 * @file src/amf/amf_config.cpp
 * @brief Shared translation from Sunshine settings to native AMF configuration.
 */

#include "amf_config.h"

#include "amf_lifecycle.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/video.h"

namespace amf {

  amf_config
    make_amf_config(const video::config_t &client_config) {
    amf_config result;
    if (client_config.videoFormat == 0) {
      result.usage = config::video.amd.amd_usage_h264;
      result.quality_preset = config::video.amd.amd_quality_h264;
      result.rc_mode = config::video.amd.amd_rc_h264;
    } else if (client_config.videoFormat == 1) {
      result.usage = config::video.amd.amd_usage_hevc;
      result.quality_preset = config::video.amd.amd_quality_hevc;
      result.rc_mode = config::video.amd.amd_rc_hevc;
    } else {
      result.usage = config::video.amd.amd_usage_av1;
      result.quality_preset = config::video.amd.amd_quality_av1;
      result.rc_mode = config::video.amd.amd_rc_av1;
    }

    result.vbaq = config::video.amd.amd_vbaq;
    result.enforce_hrd = config::video.amd.amd_enforce_hrd;
    result.qvbr_quality_level = config::video.amd.amd_qvbr_quality_level;
    result.h264_cabac = lifecycle::resolve_h264_cabac(config::video.amd.amd_coder);

    const auto preanalysis = lifecycle::resolve_preanalysis(
      result.rc_mode,
      config::video.amd.amd_preanalysis
    );
    result.preanalysis = preanalysis.enabled ? 1 : 0;
    if (preanalysis.enabled) {
      result.pa_lookahead_depth = preanalysis.lookahead_depth;
      if (preanalysis.enabled_for_rate_control &&
          (!config::video.amd.amd_preanalysis || !*config::video.amd.amd_preanalysis)) {
        BOOST_LOG(info) << "AMF: enabling native PreAnalysis required by the selected rate-control mode";
      }
    }

    result.max_ltr_frames = config::video.amd.amd_ltr_frames;
    if (config::video.amd.amd_input_queue_size > 0) {
      result.input_queue_size = config::video.amd.amd_input_queue_size;
    }

    auto tristate = [](const std::optional<int> &value) -> std::optional<bool> {
      return value ? std::optional<bool> {*value != 0} : std::nullopt;
    };
    result.multi_hw_instance_encode = tristate(config::video.amd.amd_smart_access_video);
    result.lowlatency_mode = tristate(config::video.amd.amd_lowlatency_mode);
    result.high_motion_quality_boost_enable = tristate(config::video.amd.amd_high_motion_quality_boost);
    result.av1_screen_content_tools = tristate(config::video.amd.amd_av1_screen_content);
    result.av1_encoding_latency_mode = config::video.amd.amd_av1_latency_mode;
    result.enable_statistics_feedback = false;
    return result;
  }

}  // namespace amf
