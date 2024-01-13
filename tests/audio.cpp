#pragma clang optimize off
#include <array>
#include <cstdio>
#include <cstdlib>
#include <experimental/optional>
#include <vector>
#include <cstring>

#include "audio.h"
#include "dsp.h"

#define VERIFY(call)                       \
if (R_FAILED(call)) {                  \
        printf("failed at %s\n", #call);   \
        return nullopt;                    \
}

std::vector<u8> loadDspFirmFromFile() {
    FILE *f = fopen("dspaudio.cdc", "rb");

    if (!f) {
        printf("Couldn't find dspfirm\n");
        return {};
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<u8> dspfirm_binary(size);
    fread(dspfirm_binary.data(), dspfirm_binary.size(), 1, f);
    fclose(f);

    return dspfirm_binary;
}

AudioState::AudioState(std::vector<u8>&& dspfirm) {
    // interrupt type == 2 (pipe related)
    // pipe channel == 2 (audio pipe)
    // VERIFY(DSP_RegisterInterruptEvents(pipe2_irq, 2, 2));
    lle.SetInterruptHandler([this](InterruptType type, DspPipe pipe) {
        if (type == InterruptType::Pipe && pipe == DspPipe::Audio) {
            irq2 = true;
        }
        //std::printf("SetInterruptHandler type=%d pipe=%d\n", type, pipe);
    });

    lle.LoadComponent(dspfirm);

    //VERIFY(DSP_GetSemaphoreHandle(&dsp_semaphore));

    lle.teakra.MaskSemaphore(0x2000);
    //VERIFY(DSP_SetSemaphoreMask(0x2000));

    {
        // dsp_mode == 0 (request initialisation of DSP)
        constexpr u32 dsp_mode = 0;
        std::vector<u8> buffer(4, 0);
        buffer[0] = dsp_mode;
        lle.PipeWrite(DspPipe::Audio, buffer);
        // VERIFY(DSP_WriteProcessPipe(2, &dsp_mode, 4));
    }

    // Inform the DSP that we have data for her.
    lle.SetSemaphore(0x4000);
    // VERIFY(DSP_SetSemaphore(0x4000));

    // Wait for the DSP to tell us data is available.
    /*long long size = 0;
    do {
        lle.RunTeakraSlice();
        size = lle.GetPipeReadableSize(DspPipe::Audio);
        std::printf("Pipe readable size %lld\n", size);
    } while (size == 0);*/
    waitForSync();
    // VERIFY(svcWaitSynchronization(pipe2_irq, U64_MAX));
    // VERIFY(svcClearEvent(pipe2_irq));

    {
        //VERIFY(DSP_ReadPipeIfPossible(2, 0, &num_structs, 2, &len_read));
        //if (len_read != 2) {
        //   printf("Reading struct addrs header: Could only read %i bytes!\n", len_read);
        //    return nullopt;
        //}

        u16 num_structs = 0;
        const auto num_structs_buf = lle.PipeRead(DspPipe::Audio, 2);
        assert(num_structs_buf.size() == 2);
        std::memcpy(&num_structs, num_structs_buf.data(), sizeof(u16));

        assert(num_structs == 15);

        //VERIFY(DSP_ReadPipeIfPossible(2, 0, dsp_addrs.data(), 30, &len_read));
        //if (len_read != 30) {
        //    printf("Reading struct addrs body: Could only read %i bytes!\n", len_read);
        //    return nullopt;
        //}

        std::array<u16, 15> dsp_addrs;
        const auto dsp_addrs_buf = lle.PipeRead(DspPipe::Audio, 30);
        assert(dsp_addrs_buf.size() == 30);
        std::memcpy(dsp_addrs.data(), dsp_addrs_buf.data(), sizeof(dsp_addrs));

        for (int i = 0; i < 15; i++) {
            const u32 addr0 = static_cast<u32>(dsp_addrs[i]);
            const u32 addr1 = static_cast<u32>(dsp_addrs[i]) | 0x10000;
            dsp_structs[i][0] = reinterpret_cast<u16*>(lle.GetDspDataPointer(addr0 * 2));
            dsp_structs[i][1] = reinterpret_cast<u16*>(lle.GetDspDataPointer(addr1 * 2));
        }

        for (int i = 0; i < 2; i++) {
            shared_mem[i].frame_counter = reinterpret_cast<u16*>(dsp_structs[0][i]);

            shared_mem[i].source_configurations = reinterpret_cast<DSP::HLE::SourceConfiguration*>(dsp_structs[1][i]);
            shared_mem[i].source_statuses = reinterpret_cast<DSP::HLE::SourceStatus*>(dsp_structs[2][i]);
            shared_mem[i].adpcm_coefficients = reinterpret_cast<DSP::HLE::AdpcmCoefficients*>(dsp_structs[3][i]);

            shared_mem[i].dsp_configuration = reinterpret_cast<DSP::HLE::DspConfiguration*>(dsp_structs[4][i]);
            shared_mem[i].dsp_status = reinterpret_cast<DSP::HLE::DspStatus*>(dsp_structs[5][i]);

            shared_mem[i].final_samples = reinterpret_cast<DSP::HLE::FinalMixSamples*>(dsp_structs[6][i]);
            shared_mem[i].intermediate_mix_samples = reinterpret_cast<DSP::HLE::IntermediateMixSamples*>(dsp_structs[7][i]);

            shared_mem[i].compressor = reinterpret_cast<DSP::HLE::Compressor*>(dsp_structs[8][i]);

            shared_mem[i].dsp_debug = reinterpret_cast<DSP::HLE::DspDebug*>(dsp_structs[9][i]);
        }
    }

    // Poke the DSP again.
    // VERIFY(DSP_SetSemaphore(0x4000));
    lle.SetSemaphore(0x4000);

    dsp_structs[0][0][0] = frame_id;
    frame_id++;
    // VERIFY(svcSignalEvent(dsp_semaphore));
}

AudioState::~AudioState() {
    // dsp_mode == 1 (request shutdown of DSP)
    constexpr u32 dsp_mode = 1;
    std::vector<u8> buffer(4, 0);
    buffer[0] = dsp_mode;
    lle.PipeWrite(DspPipe::Audio, buffer);

    // DSP_RegisterInterruptEvents(0, 2, 2);
    // svcCloseHandle(state.dsp_semaphore);
    // DSP_UnloadComponent();
    lle.UnloadComponent();

    // dspExit();
}

void AudioState::initSharedMem() {
    for (auto& config : write().source_configurations->config) {
        {
            config.enable = 0;
            config.enable_dirty.Assign(1);
        }

        {
            config.interpolation_mode = DSP::HLE::SourceConfiguration::Configuration::InterpolationMode::None;
            config.interpolation_related = 0;
            config.interpolation_dirty.Assign(1);
        }

        {
            config.rate_multiplier = 1.0;
            config.rate_multiplier_dirty.Assign(1);
        }

        {
            config.simple_filter_enabled.Assign(0);
            config.biquad_filter_enabled.Assign(0);
            config.filters_enabled_dirty.Assign(1);
        }

        {
            for (auto& gain : config.gain) {
                for (auto& g : gain) {
                    g = 0.0;
                }
            }
            config.gain[0][0] = 1.0;
            config.gain[0][1] = 1.0;
            config.gain_0_dirty.Assign(1);
            config.gain_1_dirty.Assign(1);
            config.gain_2_dirty.Assign(1);
        }

        {
            config.sync = 1;
            config.sync_dirty.Assign(1);
        }

        {
            config.reset_flag.Assign(1);
        }
    }

    {
        write().dsp_configuration->volume[0] = 1.0;
        write().dsp_configuration->volume[1] = 0.0;
        write().dsp_configuration->volume[2] = 0.0;
        write().dsp_configuration->volume_0_dirty.Assign(1);
        write().dsp_configuration->volume_1_dirty.Assign(1);
        write().dsp_configuration->volume_2_dirty.Assign(1);
    }

    {
        write().dsp_configuration->output_format = DSP::HLE::DspConfiguration::OutputFormat::Stereo;
        write().dsp_configuration->output_format_dirty.Assign(1);
    }

    {
        write().dsp_configuration->limiter_enabled = 0;
        write().dsp_configuration->limiter_enabled_dirty.Assign(1);
    }

    {
        // https://www.3dbrew.org/wiki/Configuration_Memory
        write().dsp_configuration->headphones_connected = false;
        write().dsp_configuration->headphones_connected_dirty.Assign(1);
    }
}

const SharedMem& AudioState::write() const {
    return shared_mem[frame_id % 2 == 1 ? 1 : 0];
}
const SharedMem& AudioState::read() const {
    return shared_mem[frame_id % 2 == 1 ? 1 : 0];
}

void AudioState::waitForSync() {
    while (!irq2) {
        lle.RunTeakraSlice();
    }
    irq2 = false;
}

void AudioState::notifyDsp() {
    write().frame_counter[0] = frame_id;
    frame_id++;
    lle.SetSemaphore(0x2000);
    //svcSignalEvent(dsp_semaphore);
}
