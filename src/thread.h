/**
 * @file src/thread.h
 * @brief Thread type selection helpers.
 */
#pragma once

// standard includes
#include <thread>

namespace util {
  /**
   * @brief Thread type that uses `std::jthread` when the standard library supports it.
   */
#if defined(__cpp_lib_jthread)
  using jthread_t = std::jthread;
#else
  using jthread_t = std::thread;  // NOSONAR(cpp:S6168) - std::jthread is not available with some libc++ toolchains.
#endif
}  // namespace util
