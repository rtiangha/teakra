#include <memory>
#include <catch.hpp>
#include <filesystem>
#include <fstream>
#include <iterator>
#include "audio.h"
#include "../src/matcher.h"
#include "../src/interpreter.h"
#include "../src/processor.h"
#include "lle.h"

// World's worst triangle wave generator.
// Generates PCM16.
void fillBuffer(u32 *audio_buffer, size_t size, unsigned freq) {
    for (size_t i = 0; i < size; i++) {
        u32 data = (i % freq) * 256;
        audio_buffer[i] = (data<<16) | (data&0xFFFF);
    }
}

void SourceStatus() {
    auto firmware = loadDspFirmFromFile();
    AudioState state(std::move(firmware));

    u32 current_offset = 0;

    // Allocate audio buffer
    auto& fcram = state.lle.fcram;
    const auto linearAlloc = [&](u32 bytes) {
        u8* ptr = fcram.get() + current_offset;
        current_offset += bytes;
        return ptr;
    };

    const auto fillBuffer = [&](u32 *audio_buffer, size_t size, unsigned freq) {
        for (size_t i = 0; i < size; i++) {
            u32 data = (i % freq) * 256;
            audio_buffer[i] = (data<<16) | (data&0xFFFF);
        }
    };

    constexpr size_t NUM_SAMPLES = 160*200;
    u32 *audio_buffer = (u32*)linearAlloc(NUM_SAMPLES * sizeof(u32));
    fillBuffer(audio_buffer, NUM_SAMPLES, 160);
    u32 *audio_buffer2 = (u32*)linearAlloc(NUM_SAMPLES * sizeof(u32));
    fillBuffer(audio_buffer2, NUM_SAMPLES, 80);
    u32 *audio_buffer3 = (u32*)linearAlloc(NUM_SAMPLES * sizeof(u32));
    fillBuffer(audio_buffer3, NUM_SAMPLES, 40);

    state.waitForSync();
    state.initSharedMem();
    state.notifyDsp();
    state.waitForSync();
    state.notifyDsp();
    state.waitForSync();
    state.notifyDsp();
    state.waitForSync();
    state.notifyDsp();
    state.waitForSync();
    state.notifyDsp();

    const auto osConvertVirtToPhys = [&](u32* addr) {
        return ((u8*)addr - fcram.get()) + FCRAM_PADDR;
    };

    {
        while (true) {
            state.waitForSync();
            printf("sync = %i, play = %i\n", state.read().source_statuses->status[0].sync, state.read().source_statuses->status[0].is_enabled);
            if (state.read().source_statuses->status[0].sync == 1) break;
            state.notifyDsp();
        }
        printf("fi: %i\n", state.frame_id);

        u16 buffer_id = 0;
        size_t next_queue_position = 0;

        state.write().source_configurations->config[0].play_position = 0;
        state.write().source_configurations->config[0].physical_address = osConvertVirtToPhys(audio_buffer3);
        state.write().source_configurations->config[0].length = NUM_SAMPLES;
        state.write().source_configurations->config[0].mono_or_stereo.Assign(DSP::HLE::SourceConfiguration::Configuration::MonoOrStereo::Stereo);
        state.write().source_configurations->config[0].format.Assign(DSP::HLE::SourceConfiguration::Configuration::Format::PCM16);
        state.write().source_configurations->config[0].fade_in.Assign(0);
        state.write().source_configurations->config[0].adpcm_dirty.Assign(0);
        state.write().source_configurations->config[0].is_looping.Assign(0);
        state.write().source_configurations->config[0].buffer_id = ++buffer_id;
        state.write().source_configurations->config[0].partial_reset_flag.Assign(1);
        state.write().source_configurations->config[0].play_position_dirty.Assign(1);
        state.write().source_configurations->config[0].embedded_buffer_dirty.Assign(1);

        state.write().source_configurations->config[0].buffers[next_queue_position].physical_address = osConvertVirtToPhys(buffer_id % 2 ? audio_buffer2 : audio_buffer);
        state.write().source_configurations->config[0].buffers[next_queue_position].length = NUM_SAMPLES;
        state.write().source_configurations->config[0].buffers[next_queue_position].adpcm_dirty = false;
        state.write().source_configurations->config[0].buffers[next_queue_position].is_looping = false;
        state.write().source_configurations->config[0].buffers[next_queue_position].buffer_id = ++buffer_id;
        state.write().source_configurations->config[0].buffers_dirty |= 1 << next_queue_position;
        next_queue_position = (next_queue_position + 1) % 4;
        state.write().source_configurations->config[0].buffer_queue_dirty.Assign(1);
        state.write().source_configurations->config[0].enable = true;
        state.write().source_configurations->config[0].enable_dirty.Assign(1);

        state.notifyDsp();

        for (size_t frame_count = 0; frame_count < 1950; frame_count++) {
            state.waitForSync();

            if (!state.read().source_statuses->status[0].is_enabled) {
                printf("%zu !\n", frame_count);
                state.write().source_configurations->config[0].enable = true;
                state.write().source_configurations->config[0].enable_dirty.Assign(1);
            }

            if (state.read().source_statuses->status[0].current_buffer_id_dirty) {
                printf("%zu %i (curr:%i)\n", frame_count, state.read().source_statuses->status[0].current_buffer_id, buffer_id+1);
                if (state.read().source_statuses->status[0].current_buffer_id == buffer_id || state.read().source_statuses->status[0].current_buffer_id == 0) {
                    state.write().source_configurations->config[0].buffers[next_queue_position].physical_address = osConvertVirtToPhys(buffer_id % 2 ? audio_buffer2 : audio_buffer);
                    state.write().source_configurations->config[0].buffers[next_queue_position].length = NUM_SAMPLES;
                    state.write().source_configurations->config[0].buffers[next_queue_position].adpcm_dirty = false;
                    state.write().source_configurations->config[0].buffers[next_queue_position].is_looping = false;
                    state.write().source_configurations->config[0].buffers[next_queue_position].buffer_id = ++buffer_id;
                    state.write().source_configurations->config[0].buffers_dirty |= 1 << next_queue_position;
                    next_queue_position = (next_queue_position + 1) % 4;
                    state.write().source_configurations->config[0].buffer_queue_dirty.Assign(1);
                }
            }

            state.notifyDsp();
        }

        u16 prev_read_bid = state.read().source_statuses->status[0].current_buffer_id;
        for (size_t frame_count = 1950; frame_count < 2208; frame_count++) {
            state.waitForSync();

            if (!state.read().source_statuses->status[0].is_enabled) {
                printf("%zu !\n", frame_count);
            }

            if (state.read().source_statuses->status[0].current_buffer_id_dirty) {
                printf("%zu d\n", frame_count);
            }

            if (prev_read_bid != state.read().source_statuses->status[0].current_buffer_id) {
                printf("%zu %i\n", frame_count, state.read().source_statuses->status[0].current_buffer_id);
                prev_read_bid = state.read().source_statuses->status[0].current_buffer_id;
            }

            state.notifyDsp();
        }

        printf("last buf id %i\n", buffer_id);

        state.waitForSync();
        state.write().source_configurations->config[0].sync = 2;
        state.write().source_configurations->config[0].sync_dirty.Assign(1);
        state.notifyDsp();

        while (true) {
            state.waitForSync();
            printf("sync = %i, play = %i\n", state.read().source_statuses->status[0].sync, state.read().source_statuses->status[0].is_enabled);
            if (state.read().source_statuses->status[0].sync == 2) break;
            state.notifyDsp();
        }
        state.notifyDsp();

        printf("Done!\n");
        std::vector<std::pair<std::string, u32>> names;
        auto& read_count = state.lle.teakra.GetProcessor().Interp().read_count;
        for (const auto& [name, count] : read_count) {
            names.emplace_back(name, count);
        }
        std::ranges::sort(names, [](auto& a, auto& b) { return a.second > b.second; });
        for (auto& [name, count] : names) {
            printf("%s %d\n", name.c_str(), count);
        }
    }
}

void InterpLinear() {
    auto firmware = loadDspFirmFromFile();
    AudioState state(std::move(firmware));

    u32 current_offset = 0;

    // Allocate audio buffer
    auto& fcram = state.lle.fcram;
    const auto linearAlloc = [&](u32 bytes) {
        u8* ptr = fcram.get() + current_offset;
        current_offset += bytes;
        return ptr;
    };

    // Very slow triangle wave, mono PCM16
    const auto fillBuffer = [&](s16 *audio_buffer, size_t size) {
        for (size_t i = 0; i < size; i++) {
            audio_buffer[i] = rand();
        }
    };

    constexpr size_t NUM_SAMPLES = 160*200;
    s16 *audio_buffer = (s16*)linearAlloc(NUM_SAMPLES * sizeof(s16));
    fillBuffer(audio_buffer, NUM_SAMPLES);

    const auto osConvertVirtToPhys = [&](s16* addr) {
        return ((u8*)addr - fcram.get()) + FCRAM_PADDR;
    };

    {
        float rate_multiplier = rand() % 512 / 128.f;
        printf("rate_multiplier = %f\n", rate_multiplier);

        std::array<s32, 160> expected_output;
        std::array<bool, 160> bad;
        {
            constexpr s32 scale = 1 << 16;
            u32 scaled_rate = rate_multiplier * scale;
            int fposition = -2 * scale;
            for (int i=0; i<160; i++) {
                int position = fposition >> 16;
                const s32 x0 = position+0 >= 0 ? audio_buffer[position+0] : 0;
                const s32 x1 = position+1 >= 0 ? audio_buffer[position+1] : 0;

                s32 delta = x1 - x0;
                if (delta > 0x7FFF) delta = 0x7FFF;
                if (delta < -0x8000) delta = -0x8000;

                u16 f0 = fposition & 0xFFFF;

                if (f0) {
                    expected_output[i] = x0 + ((f0 * delta) >> 16);
                    bad[i] = abs(x0 - x1) > 0x8000;
                } else {
                    expected_output[i] = x0;
                }

                fposition += scaled_rate;
            }
        }

        state.waitForSync();
        state.initSharedMem();
        state.write().dsp_configuration->mixer1_enabled_dirty.Assign(1);
        state.write().dsp_configuration->mixer1_enabled = true;
        state.write().source_configurations->config[0].gain[1][0] = 1.0;
        state.write().source_configurations->config[0].gain_1_dirty.Assign(1);
        state.notifyDsp();
        state.waitForSync();
        printf("init\n");

        bool entered = false;
        bool passed = true;
        {
            u16 buffer_id = 0;

            state.write().source_configurations->config[0].play_position = 0;
            state.write().source_configurations->config[0].physical_address = osConvertVirtToPhys(audio_buffer);
            state.write().source_configurations->config[0].length = NUM_SAMPLES;
            state.write().source_configurations->config[0].mono_or_stereo.Assign(DSP::HLE::SourceConfiguration::Configuration::MonoOrStereo::Mono);
            state.write().source_configurations->config[0].format.Assign(DSP::HLE::SourceConfiguration::Configuration::Format::PCM16);
            state.write().source_configurations->config[0].fade_in.Assign(0);
            state.write().source_configurations->config[0].adpcm_dirty.Assign(0);
            state.write().source_configurations->config[0].is_looping.Assign(0);
            state.write().source_configurations->config[0].buffer_id = ++buffer_id;
            state.write().source_configurations->config[0].partial_reset_flag.Assign(1);
            state.write().source_configurations->config[0].play_position_dirty.Assign(1);
            state.write().source_configurations->config[0].embedded_buffer_dirty.Assign(1);

            state.write().source_configurations->config[0].enable = true;
            state.write().source_configurations->config[0].enable_dirty.Assign(1);

            state.write().source_configurations->config[0].rate_multiplier = rate_multiplier;
            state.write().source_configurations->config[0].rate_multiplier_dirty.Assign(1);
            state.write().source_configurations->config[0].interpolation_mode = DSP::HLE::SourceConfiguration::Configuration::InterpolationMode::Linear;
            state.write().source_configurations->config[0].interpolation_related = 0;
            state.write().source_configurations->config[0].interpolation_dirty.Assign(1);

            state.notifyDsp();

            bool continue_reading = true;
            for (size_t frame_count = 0; continue_reading && frame_count < 10; frame_count++) {
                state.waitForSync();

                for (size_t i = 0; i < 160; i++) {
                    if (state.write().intermediate_mix_samples->mix1.pcm32[0][i]) {
                        entered = true;
                        printf("[intermediate] frame=%zu, sample=%zu\n", frame_count, i);
                        for (size_t j = 0; j < 160; j++) {
                            s32 real = (s32)state.write().intermediate_mix_samples->mix1.pcm32[0][j];
                            s32 expect = (s32)expected_output[j];
                            if (real != expect) {
                                printf("[%zu] real=%08x expect=%08x %s\n", j, real, expect, bad[j] ? "bad" : "");
                                passed = false;
                            }
                        }
                        continue_reading = false;
                        printf("\n");
                        break;
                    }
                }

                state.notifyDsp();
            }

            printf("Done!\n");
            if (entered && passed) {
                printf("Test passed!\n");
            } else {
                printf("FAIL\n");
            }
        }
    }
}

TEST_CASE("Audio firmware", "[interpreter]") {
    SECTION("Source Status") {
        SourceStatus();
    }
    SECTION("Linear Interp") {
        InterpLinear();
    }
}
