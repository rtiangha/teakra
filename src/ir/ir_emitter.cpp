#include "ir/ir_emitter.h"

namespace Teakra::IR {

u32 IREmitter::PC() const {
    return 0;
}

U8 IREmitter::Imm8(u8 imm) const {
    return U8(Value(imm));
}

U16 IREmitter::Imm16(u16 imm16) const {
    return U16(Value(imm16));
}

U32 IREmitter::Imm32(u32 imm32) const {
    return U32(Value(imm32));
}

U16 IREmitter::GetRegister16(RegName source_reg) {
    return Inst<U16>(Opcode::GetRegister16, Value(source_reg));
}

U32 IREmitter::GetXFactor(u32 unit) {
    return Inst<U32>(Opcode::GetXFactor, IR::Value(unit));
}

U32 IREmitter::GetYFactor(u32 unit) {
    return Inst<U32>(Opcode::GetYFactor, IR::Value(unit));
}

U64 IREmitter::GetAcc(RegName source_reg) {
    return Inst<U64>(Opcode::GetAcc, Value(source_reg));
}

void IREmitter::SetRegister16(RegName reg, const U16& value) {
    Inst(Opcode::SetRegister16, IR::Value(reg), value);
}

void IREmitter::SetAcc(RegName reg, const U64& value) {
    Inst(Opcode::SetAcc, Value(reg), Value(value));
}

void IREmitter::SetProduct(u32 unit, const U32& value) {
    Inst(Opcode::SetProduct, IR::Value(unit), value);
}

void IREmitter::SetAccFlag(const U64& value) {
    Inst(Opcode::SetAccFlag, Value(value));
}

U64 IREmitter::SaturateAcc(const U64& value, bool flag) {
    return Inst<U64>(Opcode::SaturateAcc, value, Value(flag));
}

U32 IREmitter::LogicalShiftLeft32(const U32& value, const U8& shift) {
    return Inst<U32>(Opcode::LogicalShiftLeft32, value, shift);
}

U32 IREmitter::LogicalShiftRight32(const U32& value, const U8& shift) {
    return Inst<U32>(Opcode::LogicalShiftRight32, value, shift);
}

U32 IREmitter::And32(const U32& a, const U32& b) {
    return Inst<U32>(Opcode::And32, a, b);
}

U32 IREmitter::SignExtend32(const U32& value, const U8& bits) {
    return Inst<U32>(Opcode::SignExtend32, value, bits);
}

U16 IREmitter::Add16(const U16& a, const U16& b) {
    return Inst<U16>(Opcode::Add16, a, b);
}

U16 IREmitter::Sub16(const U16& a, const U16& b) {
    return Inst<U16>(Opcode::Sub16, a, b);
}

U32 IREmitter::Mul32(const U32& a, const U32& b) {
    return Inst<U32>(Opcode::Mul32, a, b);
}

U1 IREmitter::TestBit(const U32& value, const U8& bit) {
    return Inst<U1>(Opcode::TestBit, value, bit);
}

U16 IREmitter::ExtractHalf32(const U32& value, u8 half) {
    return Inst<U16>(Opcode::ExtractHalf32, value, IR::Value(half));
}

U16 IREmitter::ExtractHalf64(const U64& value, u8 half) {
    return Inst<U16>(Opcode::ExtractHalf64, value, IR::Value(half));
}

U32 IREmitter::Pack2x16To1x32(const U16& l, const U16& h) {
    return Inst<U32>(Opcode::Pack2x16To1x32, l, h);
}

U16 IREmitter::ReadMemory16(const U16& address) {
    return Inst<U16>(Opcode::ReadMemory16, address);
}

void IREmitter::WriteMemory16(const U16& address, const U16& data) {
    Inst(Opcode::WriteMemory16, address, data);
}

} // namespace Teakra::IR
