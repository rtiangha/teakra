#pragma once

#include <atomic>
#include <cassert>
#include <vector>
#include <span>
#include <functional>
#include "audio_types.h"
#include "dsp1.h"
#include "teakra/teakra.h"

class DspLle final {
public:
    explicit DspLle() {
        fcram = std::make_unique<u8[]>(FCRAM_N3DS_SIZE);
        Teakra::AHBMCallback ahbm;
        ahbm.read8 = [this](u32 address) -> u8 {
            return fcram.get()[address - FCRAM_PADDR];
        };
        ahbm.write8 = [this](u32 address, u8 value) {
            fcram.get()[address - FCRAM_PADDR] = value;
        };
        ahbm.read16 = [this](u32 address) -> u16 {
            u16 value;
            std::memcpy(&value, fcram.get() + address - FCRAM_PADDR, sizeof(u16));
            return value;
        };
        ahbm.write16 = [this](u32 address, u16 value) {
            std::memcpy(fcram.get() + address - FCRAM_PADDR, &value, sizeof(u16));
        };
        ahbm.read32 = [this](u32 address) -> u32 {
            u32 value;
            std::memcpy(&value, fcram.get() + address - FCRAM_PADDR, sizeof(u32));
            return value;
        };
        ahbm.write32 = [this](u32 address, u32 value) {
            std::memcpy(fcram.get() + address - FCRAM_PADDR, &value, sizeof(u32));
        };
        teakra.SetAHBMCallback(ahbm);
        teakra.SetAudioCallback(
            [](std::array<s16, 2> sample) {
                //std::printf("Pushing sample 0x%x\n", std::bit_cast<u32>(sample));
            });
    }
    ~DspLle() {

    }

    u16 RecvData(u32 register_number) {
        while (!teakra.RecvDataIsReady(register_number)) {
            teakra.Run(TeakraSlice);
        }
        return teakra.RecvData(static_cast<u8>(register_number));
    }

    bool RecvDataIsReady(u32 register_number) const {
        return teakra.RecvDataIsReady(register_number);
    }

    void SetSemaphore(u16 semaphore_value) {
        teakra.SetSemaphore(semaphore_value);
    }

    std::vector<u8> PipeRead(DspPipe pipe_number, std::size_t length) {
        return ReadPipe(static_cast<u8>(pipe_number), static_cast<u16>(length));
    }

    std::size_t GetPipeReadableSize(DspPipe pipe_number) const {
        return GetPipeReadableSize(static_cast<u8>(pipe_number));
    }

    void PipeWrite(DspPipe pipe_number, std::span<const u8> buffer) {
        WritePipe(static_cast<u8>(pipe_number), buffer);
    }

    std::array<u8, DSP_RAM_SIZE>& GetDspMemory() {
        return teakra.GetDspMemory();
    }

    void RunTeakraSlice() {
        teakra.Run(TeakraSlice);
    }

    void SetInterruptHandler(
        std::function<void(InterruptType type, DspPipe pipe)> handler) {
        teakra.SetRecvDataHandler(0, [this, handler]() {
            if (!loaded) {
                return;
            }
            handler(InterruptType::Zero, static_cast<DspPipe>(0));
        });
        teakra.SetRecvDataHandler(1, [this, handler]() {
            if (!loaded) {
                return;
            }
            handler(InterruptType::One, static_cast<DspPipe>(0));
        });

        auto ProcessPipeEvent = [this, handler](bool event_from_data) {
            if (!loaded)
                return;

            if (event_from_data) {
                data_signaled = true;
            } else {
                if ((teakra.GetSemaphore() & 0x8000) == 0)
                    return;
                semaphore_signaled = true;
            }
            if (semaphore_signaled && data_signaled) {
                semaphore_signaled = data_signaled = false;
                u16 slot = teakra.RecvData(2);
                u16 side = slot % 2;
                u16 pipe = slot / 2;
                assert(pipe < 16);
                if (side != static_cast<u16>(PipeDirection::DSPtoCPU))
                    return;
                if (pipe == 0) {
                    // pipe 0 is for debug. 3DS automatically drains this pipe and discards the data
                    ReadPipe(static_cast<u8>(pipe), GetPipeReadableSize(static_cast<u8>(pipe)));
                } else {
                    handler(InterruptType::Pipe, static_cast<DspPipe>(pipe));
                }
            }
        };

        teakra.SetRecvDataHandler(2, [ProcessPipeEvent]() { ProcessPipeEvent(true); });
        teakra.SetSemaphoreHandler([ProcessPipeEvent]() { ProcessPipeEvent(false); });
    }

    void LoadComponent(std::span<const u8> buffer) {
        if (loaded) {
            std::printf("Component already loaded!\n");
            return;
        }

        teakra.Reset();

        Dsp1 dsp(buffer);

        const auto load_firmware = [&dsp](auto& dsp_memory) {
            u8* program = dsp_memory.data();
            u8* data = dsp_memory.data() + DspDataOffset;
            for (const auto& segment : dsp.segments) {
                if (segment.memory_type == SegmentType::ProgramA ||
                    segment.memory_type == SegmentType::ProgramB) {
                    std::memcpy(program + segment.target * 2, segment.data.data(), segment.data.size());
                } else if (segment.memory_type == SegmentType::Data) {
                    std::memcpy(data + segment.target * 2, segment.data.data(), segment.data.size());
                }
            }
        };

        load_firmware(teakra.GetDspMemory());
        load_firmware(teakra.GetInterpDspMemory());

        // TODO: load special segment
        //core_timing.ScheduleEvent(TeakraSlice, teakra_slice_event, 0);

        // Wait for initialization
        if (dsp.recv_data_on_start) {
            for (u8 i = 0; i < 3; ++i) {
                do {
                    WaitPipe(i);
                } while (teakra.RecvData(i) != 1);
            }
        }

        // Get pipe base address
        WaitPipe(2);
        pipe_base_waddr = teakra.RecvData(2);

        loaded = true;
    }

    void WaitPipe(u8 index) {
        while (!teakra.RecvDataIsReady(index)) {
            teakra.Run(TeakraSlice);
        }
    }

    void UnloadComponent() {
        if (!loaded) {
            std::printf("Component not loaded!\n");
            return;
        }

        loaded = false;

        // Send finalization signal via command/reply register 2
        constexpr u16 FinalizeSignal = 0x8000;
        while (!teakra.SendDataIsEmpty(2))
            teakra.Run(TeakraSlice);;

        teakra.SendData(2, FinalizeSignal);

        // Wait for completion
        while (!teakra.RecvDataIsReady(2))
            teakra.Run(TeakraSlice);

        teakra.RecvData(2); // discard the value

        //core_timing.UnscheduleEvent(teakra_slice_event, 0);
    }

    u8* GetDspDataPointer(u32 baddr) {
        auto& memory = teakra.GetDspMemory();
        return &memory[DspDataOffset + baddr];
    }

    u8* GetInterpDspDataPointer(u32 baddr) {
        auto& memory = teakra.GetInterpDspMemory();
        return &memory[DspDataOffset + baddr];
    }

    const u8* GetDspDataPointer(u32 baddr) const {
        auto& memory = teakra.GetDspMemory();
        return &memory[DspDataOffset + baddr];
    }

    PipeStatus GetPipeStatus(u8 pipe_index, PipeDirection direction) const {
        u8 slot_index = PipeIndexToSlotIndex(pipe_index, direction);
        PipeStatus pipe_status;
        std::memcpy(&pipe_status,
                    GetDspDataPointer(pipe_base_waddr * 2 + slot_index * sizeof(PipeStatus)),
                    sizeof(PipeStatus));
        assert(pipe_status.slot_index == slot_index);
        return pipe_status;
    }

    void UpdatePipeStatus(const PipeStatus& pipe_status) {
        u8 slot_index = pipe_status.slot_index;
        u8* status_address =
            GetDspDataPointer(pipe_base_waddr * 2 + slot_index * sizeof(PipeStatus));
        if (slot_index % 2 == 0) {
            std::memcpy(status_address + 4, &pipe_status.read_bptr, sizeof(u16));
        } else {
            std::memcpy(status_address + 6, &pipe_status.write_bptr, sizeof(u16));
        }

        status_address =
            GetInterpDspDataPointer(pipe_base_waddr * 2 + slot_index * sizeof(PipeStatus));
        if (slot_index % 2 == 0) {
            std::memcpy(status_address + 4, &pipe_status.read_bptr, sizeof(u16));
        } else {
            std::memcpy(status_address + 6, &pipe_status.write_bptr, sizeof(u16));
        }
    }

    void WritePipe(u8 pipe_index, std::span<const u8> data) {
        PipeStatus pipe_status = GetPipeStatus(pipe_index, PipeDirection::CPUtoDSP);
        bool need_update = false;
        const u8* buffer_ptr = data.data();
        u16 bsize = static_cast<u16>(data.size());
        while (bsize != 0) {
            assert(!pipe_status.IsFull());
            u16 write_bend;
            if (pipe_status.IsWrapped())
                write_bend = pipe_status.read_bptr & PipeStatus::PtrMask;
            else
                write_bend = pipe_status.bsize;
            u16 write_bbegin = pipe_status.write_bptr & PipeStatus::PtrMask;
            assert(write_bend > write_bbegin);
            u16 write_bsize = std::min<u16>(bsize, write_bend - write_bbegin);
            std::memcpy(GetInterpDspDataPointer(pipe_status.waddress * 2 + write_bbegin), buffer_ptr,
                        write_bsize);
            std::memcpy(GetDspDataPointer(pipe_status.waddress * 2 + write_bbegin), buffer_ptr,
                        write_bsize);
            buffer_ptr += write_bsize;
            pipe_status.write_bptr += write_bsize;
            bsize -= write_bsize;
            assert((pipe_status.write_bptr & PipeStatus::PtrMask) <= pipe_status.bsize);
            if ((pipe_status.write_bptr & PipeStatus::PtrMask) == pipe_status.bsize) {
                pipe_status.write_bptr &= PipeStatus::WrapBit;
                pipe_status.write_bptr ^= PipeStatus::WrapBit;
            }
            need_update = true;
        }
        if (need_update) {
            UpdatePipeStatus(pipe_status);
            while (!teakra.SendDataIsEmpty(2))
                teakra.Run(TeakraSlice);
            teakra.SendData(2, pipe_status.slot_index);
        }
    }

    std::vector<u8> ReadPipe(u8 pipe_index, u16 bsize) {
        PipeStatus pipe_status = GetPipeStatus(pipe_index, PipeDirection::DSPtoCPU);
        bool need_update = false;
        std::vector<u8> data(bsize);
        u8* buffer_ptr = data.data();
        while (bsize != 0) {
            assert(!pipe_status.IsEmpty());
            u16 read_bend;
            if (pipe_status.IsWrapped()) {
                read_bend = pipe_status.bsize;
            } else {
                read_bend = pipe_status.write_bptr & PipeStatus::PtrMask;
            }
            u16 read_bbegin = pipe_status.read_bptr & PipeStatus::PtrMask;
            assert(read_bend > read_bbegin);
            u16 read_bsize = std::min<u16>(bsize, read_bend - read_bbegin);
            std::vector<u8> tmp(bsize);
            std::memcpy(tmp.data(), GetInterpDspDataPointer(pipe_status.waddress * 2 + read_bbegin),
                        read_bsize);
            std::memcpy(buffer_ptr, GetDspDataPointer(pipe_status.waddress * 2 + read_bbegin),
                        read_bsize);
            assert(std::memcmp(buffer_ptr, tmp.data(), read_bsize) == 0);
            buffer_ptr += read_bsize;
            pipe_status.read_bptr += read_bsize;
            bsize -= read_bsize;
            assert((pipe_status.read_bptr & PipeStatus::PtrMask) <= pipe_status.bsize);
            if ((pipe_status.read_bptr & PipeStatus::PtrMask) == pipe_status.bsize) {
                pipe_status.read_bptr &= PipeStatus::WrapBit;
                pipe_status.read_bptr ^= PipeStatus::WrapBit;
            }
            need_update = true;
        }
        if (need_update) {
            UpdatePipeStatus(pipe_status);
            while (!teakra.SendDataIsEmpty(2))
                teakra.Run(TeakraSlice);
            teakra.SendData(2, pipe_status.slot_index);
        }
        return data;
    }
    u16 GetPipeReadableSize(u8 pipe_index) const {
        PipeStatus pipe_status = GetPipeStatus(pipe_index, PipeDirection::DSPtoCPU);
        u16 size = pipe_status.write_bptr - pipe_status.read_bptr;
        if (pipe_status.IsWrapped()) {
            size += pipe_status.bsize;
        }
        return size & PipeStatus::PtrMask;
    }

public:
    Teakra::Teakra teakra;
    u16 pipe_base_waddr = 0;

    bool semaphore_signaled = false;
    bool data_signaled = false;
    std::atomic_bool loaded = false;
    std::atomic_bool stop_signal = false;
    std::size_t stop_generation;
    std::unique_ptr<u8[]> fcram;

    static constexpr u32 DspDataOffset = 0x40000;
    static constexpr u32 TeakraSlice = 16384 * 2;
};
