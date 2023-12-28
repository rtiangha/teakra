#pragma once

#include <array>
#include <deque>
#include "../src/common_types.h"

/// Samples per second which the 3DS's audio hardware natively outputs at
constexpr int native_sample_rate = 32728; // Hz

/// Samples per audio frame at native sample rate
constexpr int samples_per_frame = 160;

constexpr size_t FCRAM_PADDR = 0x20000000;
constexpr size_t FCRAM_N3DS_SIZE = 0x10000000;
constexpr size_t DSP_RAM_VADDR = 0x1FF00000;
constexpr size_t DSP_RAM_SIZE = 0x00080000;

/// The final output to the speakers is stereo. Preprocessing output in Source is also stereo.
using StereoFrame16 = std::array<std::array<s16, 2>, samples_per_frame>;

/// The DSP is quadraphonic internally.
using QuadFrame32 = std::array<std::array<s32, 4>, samples_per_frame>;

/// A variable length buffer of signed PCM16 stereo samples.
using StereoBuffer16 = std::deque<std::array<s16, 2>>;

constexpr std::size_t num_dsp_pipe = 8;
enum class DspPipe {
    Debug = 0,
    Dma = 1,
    Audio = 2,
    Binary = 3,
};

enum class DspState {
    Off,
    On,
    Sleeping,
};

enum class InterruptType : u32 { Zero = 0, One = 1, Pipe = 2, Count };
