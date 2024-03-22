#pragma once

#include "ir/basic_block.h"
#include "ir/opcode.h"

namespace Teakra::IR {

class IREmitter {
public:
    explicit IREmitter(Block& block_, LocationDescriptor descriptor)
        : block(block_), current_location(descriptor) {}

    u32 PC() const;
    U8 Imm8(u8 value) const;
    U16 Imm16(u16 value) const;
    U32 Imm32(u32 value) const;

    U16 GetRegister16(RegName reg);
    U32 GetXFactor(u32 unit);
    U32 GetYFactor(u32 unit);
    U64 GetAcc(RegName reg);
    void SetRegister16(RegName reg, const U16& value);
    void SetAcc(RegName reg, const U64& value);
    void SetProduct(u32 unit, const U32& value);

    void SetAccFlag(const U64& value);
    U64 SaturateAcc(const U64& value, bool flag);

    U32 LogicalShiftLeft32(const U32& value, const U8& shift);
    U32 LogicalShiftRight32(const U32& value, const U8& shift);
    U32 And32(const U32& a, const U32& b);
    U32 SignExtend32(const U32& value, const U8& bits);

    U16 Add16(const U16& a, const U16& b);
    U16 Sub16(const U16& a, const U16& b);
    U32 Mul32(const U32& a, const U32& b);

    U1 TestBit(const U32& value, const U8& bit);
    U16 ExtractHalf32(const U32& value, u8 half);
    U16 ExtractHalf64(const U64& value, u8 half);
    U32 Pack2x16To1x32(const U16& l, const U16& h);

    U16 ReadMemory16(const U16& address);
    void WriteMemory16(const U16& address, const U16& data);

private:
    template <typename T = Value, typename... Args>
    T Inst(Opcode op, Args... args) {
        auto iter = block.PrependNewInst(insertion_point, op, {Value(args)...});
        return T(Value(&*iter));
    }

private:
    Block& block;
    LocationDescriptor current_location;
    Block::iterator insertion_point;
};

} // namespace Teakra::IR
