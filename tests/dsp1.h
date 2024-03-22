#pragma once

#include <cstring>
#include <span>
#include <vector>
#include "../src/common_types.h"

enum class SegmentType : u8 {
    ProgramA = 0,
    ProgramB = 1,
    Data = 2,
};

class Dsp1 {
public:
    explicit Dsp1(std::span<const u8> raw) {
        Header header;
        std::memcpy(&header, raw.data(), sizeof(header));
        recv_data_on_start = header.recv_data_on_start != 0;
        segments.reserve(header.num_segments);
        for (u32 i = 0; i < header.num_segments; ++i) {
            Segment segment;
            segment.data =
                std::vector<u8>(raw.begin() + header.segments[i].offset,
                                raw.begin() + header.segments[i].offset + header.segments[i].size);
            segment.memory_type = header.segments[i].memory_type;
            segment.target = header.segments[i].address;
            segments.push_back(std::move(segment));
        }
    }

    struct Header {
        std::array<u8, 0x100> signature;
        std::array<u8, 0x4> magic;
        u32 binary_size;
        u16 memory_layout;
        u8 pad[3];
        SegmentType special_segment_type;
        u8 num_segments;
        union {
            u8 recv_data_on_start : 1;
            u8 load_special_segment : 1;
        };
        u32 special_segment_address;
        u32 special_segment_size;
        u64 zero;
        struct Segment {
            u32 offset;
            u32 address;
            u32 size;
            u8 pad[3];
            SegmentType memory_type;
            std::array<u8, 0x20> sha256;
        };
        std::array<Segment, 10> segments;
    };
    static_assert(sizeof(Header) == 0x300);

    struct Segment {
        std::vector<u8> data;
        SegmentType memory_type;
        u32 target;
    };

    std::vector<Segment> segments;
    bool recv_data_on_start;
};

struct PipeStatus {
    u16 waddress;
    u16 bsize;
    u16 read_bptr;
    u16 write_bptr;
    u8 slot_index;
    u8 flags;

    static constexpr u16 WrapBit = 0x8000;
    static constexpr u16 PtrMask = 0x7FFF;

    bool IsFull() const {
        return (read_bptr ^ write_bptr) == WrapBit;
    }

    bool IsEmpty() const {
        return (read_bptr ^ write_bptr) == 0;
    }

    /*
     * IsWrapped: Are read and write pointers not in the same pass.
     * false:  ----[xxxx]----
     * true:   xxxx]----[xxxx (data is wrapping around the end)
     */
    bool IsWrapped() const {
        return (read_bptr ^ write_bptr) >= WrapBit;
    }
};

static_assert(sizeof(PipeStatus) == 10);

enum class PipeDirection : u8 {
    DSPtoCPU = 0,
    CPUtoDSP = 1,
};

static u8 PipeIndexToSlotIndex(u8 pipe_index, PipeDirection direction) {
    return (pipe_index << 1) + static_cast<u8>(direction);
}
