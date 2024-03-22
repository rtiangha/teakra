#pragma once

#include <array>
#include <optional>
#include <vector>

#include "dsp.h"
#include "lle.h"

using Handle = u32;
using VAddr = u32;

struct SharedMem {
    u16* frame_counter;

    DSP::HLE::SourceConfiguration* source_configurations; // access through write()
    DSP::HLE::SourceStatus* source_statuses;              // access through read()
    DSP::HLE::AdpcmCoefficients* adpcm_coefficients;      // access through write()

    DSP::HLE::DspConfiguration* dsp_configuration; // access through write()
    DSP::HLE::DspStatus* dsp_status;               // access through read()

    DSP::HLE::FinalMixSamples* final_samples;                   // access through read()
    DSP::HLE::IntermediateMixSamples* intermediate_mix_samples; // access through write()

    DSP::HLE::Compressor* compressor; // access through write()

    DSP::HLE::DspDebug* dsp_debug; // access through read()
};

struct AudioState {
    explicit AudioState(std::vector<u8>&& dspfirm);
    ~AudioState();

    void initSharedMem(bool is_jit = true);

    Handle dsp_semaphore = 0;
    DspLle lle;

    std::array<std::array<u16*, 2>, 16> dsp_structs;
    std::array<SharedMem, 2> shared_mem;
    std::array<SharedMem, 2> shared_mem_interp;
    u16 frame_id = 4;

    const SharedMem& read(bool is_jit = true) const;
    const SharedMem& write(bool is_jit = true) const;
    void waitForSync();
    void notifyDsp();

    std::atomic_bool irq2{false};
};

std::vector<u8> loadDspFirmFromFile();
