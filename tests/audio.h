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

    volatile DSP::HLE::SourceConfiguration* source_configurations; // access through write()
    volatile DSP::HLE::SourceStatus* source_statuses; // access through read()
    volatile DSP::HLE::AdpcmCoefficients* adpcm_coefficients; // access through write()

    volatile DSP::HLE::DspConfiguration* dsp_configuration; // access through write()
    volatile DSP::HLE::DspStatus* dsp_status; // access through read()

    volatile DSP::HLE::FinalMixSamples* final_samples; // access through read()
    volatile DSP::HLE::IntermediateMixSamples* intermediate_mix_samples; // access through write()

    volatile DSP::HLE::Compressor* compressor; // access through write()

    volatile DSP::HLE::DspDebug* dsp_debug; // access through read()
};

struct AudioState {
    explicit AudioState(std::vector<u8>&& dspfirm);
    ~AudioState();

    void initSharedMem();

    Handle dsp_semaphore = 0;
    DspLle lle;

    std::array<std::array<u16*, 2>, 16> dsp_structs;
    std::array<SharedMem, 2> shared_mem;
    u16 frame_id = 4;

    const SharedMem& read() const;
    const SharedMem& write() const;
    void waitForSync();
    void notifyDsp();

    std::atomic_bool irq2{false};
};

std::vector<u8> loadDspFirmFromFile();
