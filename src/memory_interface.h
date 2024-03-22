#pragma once

#include <array>
#include <bit>
#include "common_types.h"
#include "crash.h"

namespace Teakra {

class MemoryInterfaceUnit {
public:
    u16 x_page = 0, y_page = 0, z_page = 0;
    u32 x_off_storage = 0, y_off_storage = 0, z_off_storage = 0;
    u32 *x_offset{&x_off_storage}, *y_offset{&y_off_storage}, *z_offset{&z_off_storage};
    std::array<u16, 2> x_size{{0x20, 0x20}};
    std::array<u16, 2> y_size{{0x1E, 0x1E}};
    u16 page_mode_storage = 0;
    u16* page_mode{&page_mode_storage};
    u16 mmio_base_storage = 0x8000;
    u16* mmio_base{&mmio_base_storage};

    static constexpr u16 MMIOSize = 0x0800;
    static constexpr u32 DataMemoryOffset = 0x20000;
    static constexpr u32 DataMemoryBankSize = 0x10000;
    static constexpr u32 DataMemoryBankShift = std::bit_width(DataMemoryBankSize);
    static constexpr u16 XYSizeResolution = 0x400;

    void SetPageMode(u16* page_mode_) {
        page_mode = page_mode_;
    }

    void SetMmioBase(u16* mmio_base_) {
        mmio_base = mmio_base_;
    }

    void SetOffsets(u32* x_off, u32* y_off, u32* z_off) {
        x_offset = x_off;
        y_offset = y_off;
        z_offset = z_off;
    }

    void Reset() {
        x_page = 0;
        y_page = 0;
        z_page = 0;
        x_size[0] = x_size[1] = 0x20;
        y_size[0] = y_size[1] = 0x1E;
        page_mode_storage = 0;
        mmio_base_storage = 0x8000;
        x_off_storage = DataMemoryOffset;
        y_off_storage = DataMemoryOffset;
        z_off_storage = DataMemoryOffset;
    }

    bool InMMIO(u16 addr) const {
        return addr >= *mmio_base && addr < *mmio_base + MMIOSize;
    }
    u16 ToMMIO(u16 addr) const {
        ASSERT(z_page == 0);
        // according to GBATek ("DSi Teak I/O Ports (on ARM9 Side)"), these are mirrored
        return (addr - *mmio_base) & (MMIOSize - 1);
    }

    u32 ConvertDataAddress(u16 addr) const {
        if (*page_mode == 0) {
            ASSERT(z_page < 2);
            return DataMemoryOffset + addr + z_page * DataMemoryBankSize;
        } else {
            if (addr <= 0x1E * XYSizeResolution) {
                ASSERT(x_page < 2);
                return DataMemoryOffset + addr + x_page * DataMemoryBankSize;
            } else {
                ASSERT(y_page < 2);
                return DataMemoryOffset + addr + y_page * DataMemoryBankSize;
            }
        }
    }
};

struct SharedMemory;
class MMIORegion;

class MemoryInterface {
public:
    MemoryInterface(SharedMemory& shared_memory, MemoryInterfaceUnit& memory_interface_unit,
                    MMIORegion& mmio);
    u16 ProgramRead(u32 address) const;
    void ProgramWrite(u32 address, u16 value);
    u16 DataRead(u16 address,
                 bool bypass_mmio = false); // not const because it can be a FIFO register
    void DataWrite(u16 address, u16 value, bool bypass_mmio = false);
    u16 DataReadA32(u32 address) const;
    void DataWriteA32(u32 address, u16 value);
    u16 MMIORead(u16 address);
    void MMIOWrite(u16 address, u16 value);
    SharedMemory& GetMemory() {
        return shared_memory;
    }

public:
    SharedMemory& shared_memory;
    MemoryInterfaceUnit& memory_interface_unit;
    MMIORegion& mmio;
};

} // namespace Teakra
