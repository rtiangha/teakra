#pragma once

#include "operand.h"

class TranslatorVisitor final {
    void nop() {
        // literally nothing
    }

    void norm(Ax a, Rn b, StepZIDS bs) {}
    void swap(SwapType swap) {}
    void trap() {
        // throw UnimplementedException();
    }

    void DoMultiplication(u32 unit, bool x_sign, bool y_sign) {
        u32 x = regs.x[unit];
        u32 y = regs.y[unit];
        if (regs.hwm == 1 || (regs.hwm == 3 && unit == 0)) {
            y >>= 8;
        } else if (regs.hwm == 2 || (regs.hwm == 3 && unit == 1)) {
            y &= 0xFF;
        }
        if (x_sign)
            x = SignExtend<16>(x);
        if (y_sign)
            y = SignExtend<16>(y);
        regs.p[unit] = x * y;
        if (x_sign || y_sign)
            regs.pe[unit] = regs.p[unit] >> 31;
        else
            regs.pe[unit] = 0;
    }

    u64 AddSub(u64 a, u64 b, bool sub) {
        a &= 0xFF'FFFF'FFFF;
        b &= 0xFF'FFFF'FFFF;
        u64 result = sub ? a - b : a + b;
        regs.fc0 = (result >> 40) & 1;
        if (sub)
            b = ~b;
        regs.fv = ((~(a ^ b) & (a ^ result)) >> 39) & 1;
        if (regs.fv) {
            regs.fvl = 1;
        }
        return SignExtend<40>(result);
    }

    void ProductSum(SumBase base, RegName acc, bool sub_p0, bool p0_align, bool sub_p1,
                    bool p1_align) {
        u64 value_a = ProductToBus40(Px{0});
        u64 value_b = ProductToBus40(Px{1});
        if (p0_align) {
            value_a = SignExtend<24>(value_a >> 16);
        }
        if (p1_align) {
            value_b = SignExtend<24>(value_b >> 16);
        }
        u64 value_c;
        switch (base) {
        case SumBase::Zero:
            value_c = 0;
            break;
        case SumBase::Acc:
            value_c = GetAcc(acc);
            break;
        case SumBase::Sv:
            value_c = SignExtend<32, u64>((u64)regs.sv << 16);
            break;
        case SumBase::SvRnd:
            value_c = SignExtend<32, u64>((u64)regs.sv << 16) | 0x8000;
            break;
        default:
            UNREACHABLE();
        }
        u64 result = AddSub(value_c, value_a, sub_p0);
        u16 temp_c = regs.fc0;
        u16 temp_v = regs.fv;
        result = AddSub(result, value_b, sub_p1);
        // Is this correct?
        if (sub_p0 == sub_p1) {
            regs.fc0 |= temp_c;
            regs.fv |= temp_v;
        } else {
            regs.fc0 ^= temp_c;
            regs.fv ^= temp_v;
        }
        SatAndSetAccAndFlag(acc, result);
    }

    void AlmGeneric(AlmOp op, u64 a, Ax b) {
        switch (op) {
        case AlmOp::Or: {
            u64 value = GetAcc(b.GetName());
            value |= a;
            value = SignExtend<40>(value);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::And: {
            u64 value = GetAcc(b.GetName());
            value &= a;
            value = SignExtend<40>(value);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::Xor: {
            u64 value = GetAcc(b.GetName());
            value ^= a;
            value = SignExtend<40>(value);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::Tst0: {
            u64 value = GetAcc(b.GetName()) & 0xFFFF;
            regs.fz = (value & a) == 0;
            break;
        }
        case AlmOp::Tst1: {
            u64 value = GetAcc(b.GetName()) & 0xFFFF;
            regs.fz = (value & ~a) == 0;
            break;
        }
        case AlmOp::Cmp:
        case AlmOp::Cmpu:
        case AlmOp::Sub:
        case AlmOp::Subl:
        case AlmOp::Subh:
        case AlmOp::Add:
        case AlmOp::Addl:
        case AlmOp::Addh: {
            u64 value = GetAcc(b.GetName());
            bool sub = !(op == AlmOp::Add || op == AlmOp::Addl || op == AlmOp::Addh);
            u64 result = AddSub(value, a, sub);
            if (op == AlmOp::Cmp || op == AlmOp::Cmpu) {
                SetAccFlag(result);
            } else {
                SatAndSetAccAndFlag(b.GetName(), result);
            }
            break;
        }
        case AlmOp::Msu: {
            u64 value = GetAcc(b.GetName());
            u64 product = ProductToBus40(Px{0});
            u64 result = AddSub(value, product, true);
            SatAndSetAccAndFlag(b.GetName(), result);

            regs.x[0] = a & 0xFFFF;
            DoMultiplication(0, true, true);
            break;
        }
        case AlmOp::Sqra: {
            u64 value = GetAcc(b.GetName());
            u64 product = ProductToBus40(Px{0});
            u64 result = AddSub(value, product, false);
            SatAndSetAccAndFlag(b.GetName(), result);
        }
            [[fallthrough]];
        case AlmOp::Sqr: {
            regs.y[0] = regs.x[0] = a & 0xFFFF;
            DoMultiplication(0, true, true);
            break;
        }

        default:
            UNREACHABLE();
        }
    }

    u64 ExtendOperandForAlm(AlmOp op, u16 a) {
        switch (op) {
        case AlmOp::Cmp:
        case AlmOp::Sub:
        case AlmOp::Add:
            return SignExtend<16, u64>(a);
        case AlmOp::Addh:
        case AlmOp::Subh:
            return SignExtend<32, u64>((u64)a << 16);
        default:
            return a;
        }
    }

    void alm(Alm op, MemImm8 a, Ax b) {
        u16 value = LoadFromMemory(a);
        AlmGeneric(op.GetName(), ExtendOperandForAlm(op.GetName(), value), b);
    }
    void alm(Alm op, Rn a, StepZIDS as, Ax b) {
        u16 address = RnAddressAndModify(a.Index(), as.GetName());
        u16 value = mem.DataRead(address);
        AlmGeneric(op.GetName(), ExtendOperandForAlm(op.GetName(), value), b);
    }
    void alm(Alm op, Register a, Ax b) {
        u64 value;
        auto CheckBus40OperandAllowed = [op] {
            static const std::unordered_set<AlmOp> allowed_instruction{
                AlmOp::Or, AlmOp::And, AlmOp::Xor, AlmOp::Add, AlmOp::Cmp, AlmOp::Sub,
            };
            if (allowed_instruction.count(op.GetName()) == 0)
                throw UnimplementedException(); // weird effect. probably undefined
        };
        switch (a.GetName()) {
        // need more test
        case RegName::p:
            CheckBus40OperandAllowed();
            value = ProductToBus40(Px{0});
            break;
        case RegName::a0:
        case RegName::a1:
            CheckBus40OperandAllowed();
            value = GetAcc(a.GetName());
            break;
        default:
            value = ExtendOperandForAlm(op.GetName(), RegToBus16(a.GetName()));
            break;
        }
        AlmGeneric(op.GetName(), value, b);
    }
    void alm_r6(Alm op, Ax b) {
        u16 value = regs.r[6];
        AlmGeneric(op.GetName(), ExtendOperandForAlm(op.GetName(), value), b);
    }

    void alu(Alu op, MemImm16 a, Ax b) {
        u16 value = LoadFromMemory(a);
        AlmGeneric(op.GetName(), ExtendOperandForAlm(op.GetName(), value), b);
    }
    void alu(Alu op, MemR7Imm16 a, Ax b) {
        u16 value = LoadFromMemory(a);
        AlmGeneric(op.GetName(), ExtendOperandForAlm(op.GetName(), value), b);
    }
    void alu(Alu op, Imm16 a, Ax b) {
        u16 value = a.Unsigned16();
        AlmGeneric(op.GetName(), ExtendOperandForAlm(op.GetName(), value), b);
    }
    void alu(Alu op, Imm8 a, Ax b) {
        u16 value = a.Unsigned16();
        u64 and_backup = 0;
        if (op.GetName() == AlmOp::And) {
            // AND instruction has a special treatment:
            // bit 8~15 are unaffected in the accumulator, but the flags are set as if they are
            // affected
            and_backup = GetAcc(b.GetName()) & 0xFF00;
        }
        AlmGeneric(op.GetName(), ExtendOperandForAlm(op.GetName(), value), b);
        if (op.GetName() == AlmOp::And) {
            u64 and_new = GetAcc(b.GetName()) & 0xFFFF'FFFF'FFFF'00FF;
            SetAcc(b.GetName(), and_backup | and_new);
        }
    }
    void alu(Alu op, MemR7Imm7s a, Ax b) {
        u16 value = LoadFromMemory(a);
        AlmGeneric(op.GetName(), ExtendOperandForAlm(op.GetName(), value), b);
    }

    void or_(Ab a, Ax b, Ax c) {
        u64 value = GetAcc(a.GetName()) | GetAcc(b.GetName());
        SetAccAndFlag(c.GetName(), value);
    }
    void or_(Ax a, Bx b, Ax c) {
        u64 value = GetAcc(a.GetName()) | GetAcc(b.GetName());
        SetAccAndFlag(c.GetName(), value);
    }
    void or_(Bx a, Bx b, Ax c) {
        u64 value = GetAcc(a.GetName()) | GetAcc(b.GetName());
        SetAccAndFlag(c.GetName(), value);
    }

    u16 GenericAlb(Alb op, u16 a, u16 b) {
        u16 result;
        switch (op.GetName()) {
        case AlbOp::Set: {
            result = a | b;
            regs.fm = result >> 15;
            break;
        }
        case AlbOp::Rst: {
            result = ~a & b;
            regs.fm = result >> 15;
            break;
        }
        case AlbOp::Chng: {
            result = a ^ b;
            regs.fm = result >> 15;
            break;
        }
        case AlbOp::Addv: {
            u32 r = a + b;
            regs.fc0 = (r >> 16) != 0;
            regs.fm = (SignExtend<16, u32>(b) + SignExtend<16, u32>(a)) >> 31; // !
            result = r & 0xFFFF;
            break;
        }
        case AlbOp::Tst0: {
            result = (a & b) != 0;
            break;
        }
        case AlbOp::Tst1: {
            result = (a & ~b) != 0;
            break;
        }
        case AlbOp::Cmpv:
        case AlbOp::Subv: {
            u32 r = b - a;
            regs.fc0 = (r >> 16) != 0;
            regs.fm = (SignExtend<16, u32>(b) - SignExtend<16, u32>(a)) >> 31; // !
            result = r & 0xFFFF;
            break;
        }
        default:
            UNREACHABLE();
        }
        regs.fz = result == 0;
        return result;
    }

    static bool IsAlbModifying(Alb op) {
        switch (op.GetName()) {
        case AlbOp::Set:
        case AlbOp::Rst:
        case AlbOp::Chng:
        case AlbOp::Addv:
        case AlbOp::Subv:
            return true;
        case AlbOp::Tst0:
        case AlbOp::Tst1:
        case AlbOp::Cmpv:
            return false;
        default:
            UNREACHABLE();
        }
    }

    void alb(Alb op, Imm16 a, MemImm8 b) {
        u16 bv = LoadFromMemory(b);
        u16 result = GenericAlb(op, a.Unsigned16(), bv);
        if (IsAlbModifying(op))
            StoreToMemory(b, result);
    }
    void alb(Alb op, Imm16 a, Rn b, StepZIDS bs) {
        u16 address = RnAddressAndModify(b.Index(), bs.GetName());
        u16 bv = mem.DataRead(address);
        u16 result = GenericAlb(op, a.Unsigned16(), bv);
        if (IsAlbModifying(op))
            mem.DataWrite(address, result);
    }
    void alb(Alb op, Imm16 a, Register b) {
        u16 bv;
        if (b.GetName() == RegName::p) {
            bv = (u16)(ProductToBus40(Px{0}) >> 16);
        } else if (b.GetName() == RegName::a0 || b.GetName() == RegName::a1) {
            throw UnimplementedException(); // weird effect;
        } else {
            bv = RegToBus16(b.GetName());
        }
        u16 result = GenericAlb(op, a.Unsigned16(), bv);
        if (IsAlbModifying(op)) {
            switch (b.GetName()) {
            case RegName::a0:
            case RegName::a1:
                UNREACHABLE();
            // operation on accumulators doesn't go through regular bus with flag and saturation
            case RegName::a0l:
                regs.a[0] = (regs.a[0] & 0xFFFF'FFFF'FFFF'0000) | result;
                break;
            case RegName::a1l:
                regs.a[1] = (regs.a[1] & 0xFFFF'FFFF'FFFF'0000) | result;
                break;
            case RegName::b0l:
                regs.b[0] = (regs.b[0] & 0xFFFF'FFFF'FFFF'0000) | result;
                break;
            case RegName::b1l:
                regs.b[1] = (regs.b[1] & 0xFFFF'FFFF'FFFF'0000) | result;
                break;
            case RegName::a0h:
                regs.a[0] = (regs.a[0] & 0xFFFF'FFFF'0000'FFFF) | ((u64)result << 16);
                break;
            case RegName::a1h:
                regs.a[1] = (regs.a[1] & 0xFFFF'FFFF'0000'FFFF) | ((u64)result << 16);
                break;
            case RegName::b0h:
                regs.b[0] = (regs.b[0] & 0xFFFF'FFFF'0000'FFFF) | ((u64)result << 16);
                break;
            case RegName::b1h:
                regs.b[1] = (regs.b[1] & 0xFFFF'FFFF'0000'FFFF) | ((u64)result << 16);
                break;
            default:
                RegFromBus16(b.GetName(), result); // including RegName:p (p0h)
            }
        }
    }
    void alb_r6(Alb op, Imm16 a) {
        u16 bv = regs.r[6];
        u16 result = GenericAlb(op, a.Unsigned16(), bv);
        if (IsAlbModifying(op))
            regs.r[6] = result;
    }
    void alb(Alb op, Imm16 a, SttMod b) {
        u16 bv = RegToBus16(b.GetName());
        u16 result = GenericAlb(op, a.Unsigned16(), bv);
        if (IsAlbModifying(op))
            RegFromBus16(b.GetName(), result);
    }

    void add(Ab a, Bx b) {
        u64 value_a = GetAcc(a.GetName());
        u64 value_b = GetAcc(b.GetName());
        u64 result = AddSub(value_b, value_a, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void add(Bx a, Ax b) {
        u64 value_a = GetAcc(a.GetName());
        u64 value_b = GetAcc(b.GetName());
        u64 result = AddSub(value_b, value_a, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void add_p1(Ax b) {
        u64 value_a = ProductToBus40(Px{1});
        u64 value_b = GetAcc(b.GetName());
        u64 result = AddSub(value_b, value_a, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void add(Px a, Bx b) {
        u64 value_a = ProductToBus40(a);
        u64 value_b = GetAcc(b.GetName());
        u64 result = AddSub(value_b, value_a, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }

    void sub(Ab a, Bx b) {
        u64 value_a = GetAcc(a.GetName());
        u64 value_b = GetAcc(b.GetName());
        u64 result = AddSub(value_b, value_a, true);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void sub(Bx a, Ax b) {
        u64 value_a = GetAcc(a.GetName());
        u64 value_b = GetAcc(b.GetName());
        u64 result = AddSub(value_b, value_a, true);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void sub_p1(Ax b) {
        u64 value_a = ProductToBus40(Px{1});
        u64 value_b = GetAcc(b.GetName());
        u64 result = AddSub(value_b, value_a, true);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void sub(Px a, Bx b) {
        u64 value_a = ProductToBus40(a);
        u64 value_b = GetAcc(b.GetName());
        u64 result = AddSub(value_b, value_a, true);
        SatAndSetAccAndFlag(b.GetName(), result);
    }

    void app(Ab c, SumBase base, bool sub_p0, bool p0_align, bool sub_p1, bool p1_align) {
        ProductSum(base, c.GetName(), sub_p0, p0_align, sub_p1, p1_align);
    }

    void add_add(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        auto [ui, uj] = GetArpRnUnit(a);
        auto [si, sj] = GetArpStep(asi, asj);
        auto [oi, oj] = GetArpOffset(asi, asj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 high = SignExtend<16, u64>(mem.DataRead(j)) + SignExtend<16, u64>(mem.DataRead(i));
        u16 low = mem.DataRead(OffsetAddress(uj, j, oj)) + mem.DataRead(OffsetAddress(ui, i, oi));
        u64 result = (high << 16) | low;
        SetAcc(b.GetName(), result);
    }
    void add_sub(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        auto [ui, uj] = GetArpRnUnit(a);
        auto [si, sj] = GetArpStep(asi, asj);
        auto [oi, oj] = GetArpOffset(asi, asj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 high = SignExtend<16, u64>(mem.DataRead(j)) + SignExtend<16, u64>(mem.DataRead(i));
        u16 low = mem.DataRead(OffsetAddress(uj, j, oj)) - mem.DataRead(OffsetAddress(ui, i, oi));
        u64 result = (high << 16) | low;
        SetAcc(b.GetName(), result);
    }
    void sub_add(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        auto [ui, uj] = GetArpRnUnit(a);
        auto [si, sj] = GetArpStep(asi, asj);
        auto [oi, oj] = GetArpOffset(asi, asj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 high = SignExtend<16, u64>(mem.DataRead(j)) - SignExtend<16, u64>(mem.DataRead(i));
        u16 low = mem.DataRead(OffsetAddress(uj, j, oj)) + mem.DataRead(OffsetAddress(ui, i, oi));
        u64 result = (high << 16) | low;
        SetAcc(b.GetName(), result);
    }
    void sub_sub(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        auto [ui, uj] = GetArpRnUnit(a);
        auto [si, sj] = GetArpStep(asi, asj);
        auto [oi, oj] = GetArpOffset(asi, asj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 high = SignExtend<16, u64>(mem.DataRead(j)) - SignExtend<16, u64>(mem.DataRead(i));
        u16 low = mem.DataRead(OffsetAddress(uj, j, oj)) - mem.DataRead(OffsetAddress(ui, i, oi));
        u64 result = (high << 16) | low;
        SetAcc(b.GetName(), result);
    }
    void add_sub_sv(ArRn1 a, ArStep1 as, Ab b) {
        u16 u = GetArRnUnit(a);
        auto s = GetArStep(as);
        auto o = GetArOffset(as);
        u16 address = RnAddressAndModify(u, s);
        u64 high = SignExtend<16, u64>(mem.DataRead(address)) + SignExtend<16, u64>(regs.sv);
        u16 low = mem.DataRead(OffsetAddress(u, address, o)) - regs.sv;
        u64 result = (high << 16) | low;
        SetAcc(b.GetName(), result);
    }
    void sub_add_sv(ArRn1 a, ArStep1 as, Ab b) {
        u16 u = GetArRnUnit(a);
        auto s = GetArStep(as);
        auto o = GetArOffset(as);
        u16 address = RnAddressAndModify(u, s);
        u64 high = SignExtend<16, u64>(mem.DataRead(address)) - SignExtend<16, u64>(regs.sv);
        u16 low = mem.DataRead(OffsetAddress(u, address, o)) + regs.sv;
        u64 result = (high << 16) | low;
        SetAcc(b.GetName(), result);
    }
    void sub_add_i_mov_j_sv(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        auto [ui, uj] = GetArpRnUnit(a);
        auto [si, sj] = GetArpStep(asi, asj);
        OffsetValue oi;
        std::tie(oi, std::ignore) = GetArpOffset(asi, asj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 high = SignExtend<16, u64>(mem.DataRead(i)) - SignExtend<16, u64>(regs.sv);
        u16 low = mem.DataRead(OffsetAddress(ui, i, oi)) + regs.sv;
        u64 result = (high << 16) | low;
        SetAcc(b.GetName(), result);
        regs.sv = mem.DataRead(j);
    }
    void sub_add_j_mov_i_sv(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        auto [ui, uj] = GetArpRnUnit(a);
        auto [si, sj] = GetArpStep(asi, asj);
        OffsetValue oj;
        std::tie(std::ignore, oj) = GetArpOffset(asi, asj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 high = SignExtend<16, u64>(mem.DataRead(j)) - SignExtend<16, u64>(regs.sv);
        u16 low = mem.DataRead(OffsetAddress(uj, j, oj)) + regs.sv;
        u64 result = (high << 16) | low;
        SetAcc(b.GetName(), result);
        regs.sv = mem.DataRead(i);
    }
    void add_sub_i_mov_j(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        auto [ui, uj] = GetArpRnUnit(a);
        auto [si, sj] = GetArpStep(asi, asj);
        OffsetValue oi;
        std::tie(oi, std::ignore) = GetArpOffset(asi, asj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 high = SignExtend<16, u64>(mem.DataRead(i)) + SignExtend<16, u64>(regs.sv);
        u16 low = mem.DataRead(OffsetAddress(ui, i, oi)) - regs.sv;
        u64 result = (high << 16) | low;
        u16 exchange = (u16)(GetAndSatAccNoFlag(b.GetName()) & 0xFFFF);
        SetAcc(b.GetName(), result);
        mem.DataWrite(j, exchange);
    }
    void add_sub_j_mov_i(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        auto [ui, uj] = GetArpRnUnit(a);
        auto [si, sj] = GetArpStep(asi, asj);
        OffsetValue oj;
        std::tie(std::ignore, oj) = GetArpOffset(asi, asj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 high = SignExtend<16, u64>(mem.DataRead(j)) + SignExtend<16, u64>(regs.sv);
        u16 low = mem.DataRead(OffsetAddress(uj, j, oj)) - regs.sv;
        u64 result = (high << 16) | low;
        u16 exchange = (u16)(GetAndSatAccNoFlag(b.GetName()) & 0xFFFF);
        SetAcc(b.GetName(), result);
        mem.DataWrite(i, exchange);
    }

    void Moda(ModaOp op, RegName a, Cond cond) {
        if (regs.ConditionPass(cond)) {
            switch (op) {
            case ModaOp::Shr: {
                ShiftBus40(GetAcc(a), 0xFFFF, a);
                break;
            }
            case ModaOp::Shr4: {
                ShiftBus40(GetAcc(a), 0xFFFC, a);
                break;
            }
            case ModaOp::Shl: {
                ShiftBus40(GetAcc(a), 1, a);
                break;
            }
            case ModaOp::Shl4: {
                ShiftBus40(GetAcc(a), 4, a);
                break;
            }
            case ModaOp::Ror: {
                u64 value = GetAcc(a) & 0xFF'FFFF'FFFF;
                u16 old_fc = regs.fc0;
                regs.fc0 = value & 1;
                value >>= 1;
                value |= (u64)old_fc << 39;
                value = SignExtend<40>(value);
                SetAccAndFlag(a, value);
                break;
            }
            case ModaOp::Rol: {
                u64 value = GetAcc(a);
                u16 old_fc = regs.fc0;
                regs.fc0 = (value >> 39) & 1;
                value <<= 1;
                value |= old_fc;
                value = SignExtend<40>(value);
                SetAccAndFlag(a, value);
                break;
            }
            case ModaOp::Clr: {
                SatAndSetAccAndFlag(a, 0);
                break;
            }
            case ModaOp::Not: {
                u64 result = ~GetAcc(a);
                SetAccAndFlag(a, result);
                break;
            }
            case ModaOp::Neg: {
                u64 value = GetAcc(a);
                regs.fc0 = value != 0;                    // ?
                regs.fv = value == 0xFFFF'FF80'0000'0000; // ?
                if (regs.fv)
                    regs.fvl = 1;
                u64 result = SignExtend<40, u64>(~GetAcc(a) + 1);
                SatAndSetAccAndFlag(a, result);
                break;
            }
            case ModaOp::Rnd: {
                u64 value = GetAcc(a);
                u64 result = AddSub(value, 0x8000, false);
                SatAndSetAccAndFlag(a, result);
                break;
            }
            case ModaOp::Pacr: {
                u64 value = ProductToBus40(Px{0});
                u64 result = AddSub(value, 0x8000, false);
                SatAndSetAccAndFlag(a, result);
                break;
            }
            case ModaOp::Clrr: {
                SatAndSetAccAndFlag(a, 0x8000);
                break;
            }
            case ModaOp::Inc: {
                u64 value = GetAcc(a);
                u64 result = AddSub(value, 1, false);
                SatAndSetAccAndFlag(a, result);
                break;
            }
            case ModaOp::Dec: {
                u64 value = GetAcc(a);
                u64 result = AddSub(value, 1, true);
                SatAndSetAccAndFlag(a, result);
                break;
            }
            case ModaOp::Copy: {
                // note: bX doesn't support
                u64 value = GetAcc(a == RegName::a0 ? RegName::a1 : RegName::a0);
                SatAndSetAccAndFlag(a, value);
                break;
            }
            default:
                UNREACHABLE();
            }
        }
    }

    void moda4(Moda4 op, Ax a, Cond cond) {
        Moda(op.GetName(), a.GetName(), cond);
    }

    void moda3(Moda3 op, Bx a, Cond cond) {
        Moda(op.GetName(), a.GetName(), cond);
    }

    void pacr1(Ax a) {
        u64 value = ProductToBus40(Px{1});
        u64 result = AddSub(value, 0x8000, false);
        SatAndSetAccAndFlag(a.GetName(), result);
    }

    void FilterDoubleClr(RegName& a, RegName& b) {
        if (a == RegName::b0) {
            b = RegName::b1;
        } else if (a == RegName::b1) {
            b = RegName::b0;
        } else if (a == RegName::a0) {
            if (b == RegName::a0)
                b = RegName::a1;
        } else
            b = b == RegName::b1 ? RegName::b1 : RegName::b0;
    }

    void clr(Ab a, Ab b) {
        RegName a_name = a.GetName();
        RegName b_name = b.GetName();
        FilterDoubleClr(a_name, b_name);
        SatAndSetAccAndFlag(a_name, 0);
        SatAndSetAccAndFlag(b_name, 0);
    }
    void clrr(Ab a, Ab b) {
        RegName a_name = a.GetName();
        RegName b_name = b.GetName();
        FilterDoubleClr(a_name, b_name);
        SatAndSetAccAndFlag(a_name, 0x8000);
        SatAndSetAccAndFlag(b_name, 0x8000);
    }

    void BlockRepeat(u16 lc, u32 address) {
        ASSERT(regs.bcn <= 3);
        regs.bkrep_stack[regs.bcn].start = regs.pc;
        regs.bkrep_stack[regs.bcn].end = address;
        regs.bkrep_stack[regs.bcn].lc = lc;
        regs.lp = 1;
        ++regs.bcn;
    }

    void bkrep(Imm8 a, Address16 addr) {
        u16 lc = a.Unsigned16();
        u32 address = addr.Address32() | (regs.pc & 0x30000);
        BlockRepeat(lc, address);
    }
    void bkrep(Register a, Address18_16 addr_low, Address18_2 addr_high) {
        u16 lc = RegToBus16(a.GetName());
        u32 address = Address32(addr_low, addr_high);
        BlockRepeat(lc, address);
    }
    void bkrep_r6(Address18_16 addr_low, Address18_2 addr_high) {
        u16 lc = regs.r[6];
        u32 address = Address32(addr_low, addr_high);
        BlockRepeat(lc, address);
    }

    void RestoreBlockRepeat(u16& address_reg) {
        if (regs.lp) {
            ASSERT(regs.bcn <= 3);
            std::copy_backward(regs.bkrep_stack.begin(), regs.bkrep_stack.begin() + regs.bcn,
                               regs.bkrep_stack.begin() + regs.bcn + 1);
            ++regs.bcn;
        }
        u32 flag = mem.DataRead(address_reg++);
        u16 valid = flag >> 15;
        if (regs.lp) {
            ASSERT(valid);
        } else {
            if (valid)
                regs.lp = regs.bcn = 1;
        }
        regs.bkrep_stack[0].end = mem.DataRead(address_reg++) | (((flag >> 8) & 3) << 16);
        regs.bkrep_stack[0].start = mem.DataRead(address_reg++) | ((flag & 3) << 16);
        regs.bkrep_stack[0].lc = mem.DataRead(address_reg++);
    }
    void StoreBlockRepeat(u16& address_reg) {
        mem.DataWrite(--address_reg, regs.bkrep_stack[0].lc);
        mem.DataWrite(--address_reg, regs.bkrep_stack[0].start & 0xFFFF);
        mem.DataWrite(--address_reg, regs.bkrep_stack[0].end & 0xFFFF);
        u16 flag = regs.lp << 15;
        flag |= regs.bkrep_stack[0].start >> 16;
        flag |= (regs.bkrep_stack[0].end >> 16) << 8;
        mem.DataWrite(--address_reg, flag);
        if (regs.lp) {
            std::copy(regs.bkrep_stack.begin() + 1, regs.bkrep_stack.begin() + regs.bcn,
                      regs.bkrep_stack.begin());
            --regs.bcn;
            if (regs.bcn == 0)
                regs.lp = 0;
        }
    }
    void bkreprst(ArRn2 a) {
        RestoreBlockRepeat(regs.r[GetArRnUnit(a)]);
    }
    void bkreprst_memsp() {
        RestoreBlockRepeat(regs.sp);
    }
    void bkrepsto(ArRn2 a) {
        StoreBlockRepeat(regs.r[GetArRnUnit(a)]);
    }
    void bkrepsto_memsp() {
        StoreBlockRepeat(regs.sp);
    }

    void banke(BankFlags flags) {
        if (flags.Cfgi()) {
            std::swap(regs.stepi, regs.stepib);
            std::swap(regs.modi, regs.modib);
            if (regs.stp16)
                std::swap(regs.stepi0, regs.stepi0b);
        }
        if (flags.R4()) {
            std::swap(regs.r[4], regs.r4b);
        }
        if (flags.R1()) {
            std::swap(regs.r[1], regs.r1b);
        }
        if (flags.R0()) {
            std::swap(regs.r[0], regs.r0b);
        }
        if (flags.R7()) {
            std::swap(regs.r[7], regs.r7b);
        }
        if (flags.Cfgj()) {
            std::swap(regs.stepj, regs.stepjb);
            std::swap(regs.modj, regs.modjb);
            if (regs.stp16)
                std::swap(regs.stepj0, regs.stepj0b);
        }
    }
    void bankr() {
        regs.SwapAllArArp();
    }
    void bankr(Ar a) {
        regs.SwapAr(a.Index());
    }
    void bankr(Ar a, Arp b) {
        regs.SwapAr(a.Index());
        regs.SwapArp(b.Index());
    }
    void bankr(Arp a) {
        regs.SwapArp(a.Index());
    }

    void bitrev(Rn a) {
        u32 unit = a.Index();
        regs.r[unit] = BitReverse(regs.r[unit]);
    }
    void bitrev_dbrv(Rn a) {
        u32 unit = a.Index();
        regs.r[unit] = BitReverse(regs.r[unit]);
        regs.br[unit] = 0;
    }
    void bitrev_ebrv(Rn a) {
        u32 unit = a.Index();
        regs.r[unit] = BitReverse(regs.r[unit]);
        regs.br[unit] = 1;
    }

    void br(Address18_16 addr_low, Address18_2 addr_high, Cond cond) {
        if (regs.ConditionPass(cond)) {
            SetPC(Address32(addr_low, addr_high));
        }
    }

    void brr(RelAddr7 addr, Cond cond) {
        if (regs.ConditionPass(cond)) {
            regs.pc += addr.Relative32(); // note: pc is the address of the NEXT instruction
            if (addr.Relative32() == 0xFFFFFFFF) {
                idle = true;
            }
        }
    }

    void break_() {
        ASSERT(regs.lp);
        --regs.bcn;
        regs.lp = regs.bcn != 0;
        // Note: unlike one would expect, the "break" instruction doesn't jump out of the block
    }

    void call(Address18_16 addr_low, Address18_2 addr_high, Cond cond) {
        if (regs.ConditionPass(cond)) {
            PushPC();
            SetPC(Address32(addr_low, addr_high));
        }
    }
    void calla(Axl a) {
        PushPC();
        SetPC(RegToBus16(a.GetName())); // use pcmhi?
    }
    void calla(Ax a) {
        PushPC();
        SetPC(GetAcc(a.GetName()) & 0x3FFFF); // no saturation ?
    }
    void callr(RelAddr7 addr, Cond cond) {
        if (regs.ConditionPass(cond)) {
            PushPC();
            regs.pc += addr.Relative32();
        }
    }

    void ContextStore() {
        regs.ShadowStore();
        regs.ShadowSwap();
        if (!regs.crep) {
            regs.repcs = regs.repc;
        }
        if (!regs.ccnta) {
            regs.a1s = regs.a[1];
            regs.b1s = regs.b[1];
        } else {
            u64 a = regs.a[1];
            u64 b = regs.b[1];
            regs.b[1] = a;
            SetAccAndFlag(RegName::a1, b); // Flag set on b1->a1
        }
    }

    void ContextRestore() {
        regs.ShadowRestore();
        regs.ShadowSwap();
        if (!regs.crep) {
            regs.repc = regs.repcs;
        }
        if (!regs.ccnta) {
            regs.a[1] = regs.a1s;
            regs.b[1] = regs.b1s;
        } else {
            std::swap(regs.a[1], regs.b[1]);
        }
    }

    void cntx_s() {
        ContextStore();
    }
    void cntx_r() {
        ContextRestore();
    }

    void ret(Cond c) {
        if (regs.ConditionPass(c)) {
            PopPC();
        }
    }
    void retd() {
        throw UnimplementedException();
    }
    void reti(Cond c) {
        if (regs.ConditionPass(c)) {
            PopPC();
            regs.ie = 1;
        }
    }
    void retic(Cond c) {
        if (regs.ConditionPass(c)) {
            PopPC();
            regs.ie = 1;
            ContextRestore();
        }
    }
    void retid() {
        UNREACHABLE();
    }
    void retidc() {
        UNREACHABLE();
    }
    void rets(Imm8 a) {
        PopPC();
        regs.sp += a.Unsigned16();
    }

    void load_ps(Imm2 a) {
        regs.ps[0] = a.Unsigned16();
    }
    void load_stepi(Imm7s a) {
        // Although this is signed, we still only store the lower 7 bits
        regs.stepi = a.Signed16() & 0x7F;
    }
    void load_stepj(Imm7s a) {
        regs.stepj = a.Signed16() & 0x7F;
    }
    void load_page(Imm8 a) {
        regs.page = a.Unsigned16();
    }
    void load_modi(Imm9 a) {
        regs.modi = a.Unsigned16();
    }
    void load_modj(Imm9 a) {
        regs.modj = a.Unsigned16();
    }
    void load_movpd(Imm2 a) {
        regs.pcmhi = a.Unsigned16();
    }
    void load_ps01(Imm4 a) {
        regs.ps[0] = a.Unsigned16() & 3;
        regs.ps[1] = a.Unsigned16() >> 2;
    }

    void push(Imm16 a) {
        mem.DataWrite(--regs.sp, a.Unsigned16());
    }
    void push(Register a) {
        u16 value = RegToBus16(a.GetName(), true);
        mem.DataWrite(--regs.sp, value);
    }
    void push(Abe a) {
        u16 value = (GetAndSatAcc(a.GetName()) >> 32) & 0xFFFF;
        mem.DataWrite(--regs.sp, value);
    }
    void push(ArArpSttMod a) {
        u16 value = RegToBus16(a.GetName());
        mem.DataWrite(--regs.sp, value);
    }
    void push_prpage() {
        mem.DataWrite(--regs.sp, regs.prpage);
    }
    void push(Px a) {
        u32 value = (u32)ProductToBus40(a);
        u16 h = value >> 16;
        u16 l = value & 0xFFFF;
        mem.DataWrite(--regs.sp, l);
        mem.DataWrite(--regs.sp, h);
    }
    void push_r6() {
        u16 value = regs.r[6];
        mem.DataWrite(--regs.sp, value);
    }
    void push_repc() {
        u16 value = regs.repc;
        mem.DataWrite(--regs.sp, value);
    }
    void push_x0() {
        u16 value = regs.x[0];
        mem.DataWrite(--regs.sp, value);
    }
    void push_x1() {
        u16 value = regs.x[1];
        mem.DataWrite(--regs.sp, value);
    }
    void push_y1() {
        u16 value = regs.y[1];
        mem.DataWrite(--regs.sp, value);
    }
    void pusha(Ax a) {
        u32 value = GetAndSatAcc(a.GetName()) & 0xFFFF'FFFF;
        u16 h = value >> 16;
        u16 l = value & 0xFFFF;
        mem.DataWrite(--regs.sp, l);
        mem.DataWrite(--regs.sp, h);
    }
    void pusha(Bx a) {
        u32 value = GetAndSatAcc(a.GetName()) & 0xFFFF'FFFF;
        u16 h = value >> 16;
        u16 l = value & 0xFFFF;
        mem.DataWrite(--regs.sp, l);
        mem.DataWrite(--regs.sp, h);
    }

    void pop(Register a) {
        u16 value = mem.DataRead(regs.sp++);
        RegFromBus16(a.GetName(), value);
    }
    void pop(Abe a) {
        u32 value32 = SignExtend<8, u32>(mem.DataRead(regs.sp++) & 0xFF);
        u64 acc = GetAcc(a.GetName());
        SetAccAndFlag(a.GetName(), (acc & 0xFFFFFFFF) | (u64)value32 << 32);
    }
    void pop(ArArpSttMod a) {
        u16 value = mem.DataRead(regs.sp++);
        RegFromBus16(a.GetName(), value);
    }
    void pop(Bx a) {
        u16 value = mem.DataRead(regs.sp++);
        RegFromBus16(a.GetName(), value);
    }
    void pop_prpage() {
        regs.prpage = mem.DataRead(regs.sp++);
    }
    void pop(Px a) {
        u16 h = mem.DataRead(regs.sp++);
        u16 l = mem.DataRead(regs.sp++);
        u32 value = ((u32)h << 16) | l;
        ProductFromBus32(a, value);
    }
    void pop_r6() {
        u16 value = mem.DataRead(regs.sp++);
        regs.r[6] = value;
    }
    void pop_repc() {
        u16 value = mem.DataRead(regs.sp++);
        regs.repc = value;
    }
    void pop_x0() {
        u16 value = mem.DataRead(regs.sp++);
        regs.x[0] = value;
    }
    void pop_x1() {
        u16 value = mem.DataRead(regs.sp++);
        regs.x[1] = value;
    }
    void pop_y1() {
        u16 value = mem.DataRead(regs.sp++);
        regs.y[1] = value;
    }
    void popa(Ab a) {
        u16 h = mem.DataRead(regs.sp++);
        u16 l = mem.DataRead(regs.sp++);
        u64 value = SignExtend<32, u64>(((u64)h << 16) | l);
        SetAccAndFlag(a.GetName(), value);
    }

    void Repeat(u16 repc) {
        regs.repc = repc;
        regs.rep = true;
    }

    void rep(Imm8 a) {
        Repeat(a.Unsigned16());
    }
    void rep(Register a) {
        Repeat(RegToBus16(a.GetName()));
    }
    void rep_r6() {
        Repeat(regs.r[6]);
    }

    void shfc(Ab a, Ab b, Cond cond) {
        if (regs.ConditionPass(cond)) {
            u64 value = GetAcc(a.GetName());
            u16 sv = regs.sv;
            ShiftBus40(value, sv, b.GetName());
        }
    }
    void shfi(Ab a, Ab b, Imm6s s) {
        u64 value = GetAcc(a.GetName());
        u16 sv = s.Signed16();
        ShiftBus40(value, sv, b.GetName());
    }

    void tst4b(ArRn2 b, ArStep2 bs) {
        u16 address = RnAddressAndModify(GetArRnUnit(b), GetArStep(bs));
        u16 value = mem.DataRead(address);
        u64 bit = GetAcc(RegName::a0) & 0xF;
        // Is this correct? an why?
        regs.fz = regs.fc0 = (value >> bit) & 1;
    }
    void tst4b(ArRn2 b, ArStep2 bs, Ax c) {
        u64 a = GetAcc(RegName::a0);
        u64 bit = a & 0xF;
        u16 fv = regs.fv;
        u16 fvl = regs.fvl;
        u16 fm = regs.fm;
        u16 fn = regs.fn;
        u16 fe = regs.fe;
        u16 sv = regs.sv;
        ShiftBus40(a, sv, c.GetName());
        regs.fc1 = regs.fc0;
        regs.fv = fv;
        regs.fvl = fvl;
        regs.fm = fm;
        regs.fn = fn;
        regs.fe = fe;
        u16 address = RnAddressAndModify(GetArRnUnit(b), GetArStep(bs));
        u16 value = mem.DataRead(address);
        regs.fz = regs.fc0 = (value >> bit) & 1;
    }
    void tstb(MemImm8 a, Imm4 b) {
        u16 value = LoadFromMemory(a);
        regs.fz = (value >> b.Unsigned16()) & 1;
    }
    void tstb(Rn a, StepZIDS as, Imm4 b) {
        u16 address = RnAddressAndModify(a.Index(), as.GetName());
        u16 value = mem.DataRead(address);
        regs.fz = (value >> b.Unsigned16()) & 1;
    }
    void tstb(Register a, Imm4 b) {
        u16 value = RegToBus16(a.GetName());
        regs.fz = (value >> b.Unsigned16()) & 1;
    }
    void tstb_r6(Imm4 b) {
        u16 value = regs.r[6];
        regs.fz = (value >> b.Unsigned16()) & 1;
    }
    void tstb(SttMod a, Imm16 b) {
        u16 value = RegToBus16(a.GetName());
        regs.fz = (value >> b.Unsigned16()) & 1;
    }

    void and_(Ab a, Ab b, Ax c) {
        u64 value = GetAcc(a.GetName()) & GetAcc(b.GetName());
        SetAccAndFlag(c.GetName(), value);
    }

    void dint() {
        regs.ie = 0;
    }
    void eint() {
        regs.ie = 1;
    }

    void MulGeneric(MulOp op, Ax a) {
        if (op != MulOp::Mpy && op != MulOp::Mpysu) {
            u64 value = GetAcc(a.GetName());
            u64 product = ProductToBus40(Px{0});
            if (op == MulOp::Maa || op == MulOp::Maasu) {
                product >>= 16;
                product = SignExtend<24>(product);
            }
            u64 result = AddSub(value, product, false);
            SatAndSetAccAndFlag(a.GetName(), result);
        }

        switch (op) {
        case MulOp::Mpy:
        case MulOp::Mac:
        case MulOp::Maa:
            DoMultiplication(0, true, true);
            break;
        case MulOp::Mpysu:
        case MulOp::Macsu:
        case MulOp::Maasu:
            // Note: the naming conventin of "mpysu" is "multiply signed *y* by unsigned *x*"
            DoMultiplication(0, false, true);
            break;
        case MulOp::Macus:
            DoMultiplication(0, true, false);
            break;
        case MulOp::Macuu:
            DoMultiplication(0, false, false);
            break;
        }
    }

    void mul(Mul3 op, Rn y, StepZIDS ys, Imm16 x, Ax a) {
        u16 address = RnAddressAndModify(y.Index(), ys.GetName());
        regs.y[0] = mem.DataRead(address);
        regs.x[0] = x.Unsigned16();
        MulGeneric(op.GetName(), a);
    }
    void mul_y0(Mul3 op, Rn x, StepZIDS xs, Ax a) {
        u16 address = RnAddressAndModify(x.Index(), xs.GetName());
        regs.x[0] = mem.DataRead(address);
        MulGeneric(op.GetName(), a);
    }
    void mul_y0(Mul3 op, Register x, Ax a) {
        regs.x[0] = RegToBus16(x.GetName());
        MulGeneric(op.GetName(), a);
    }
    void mul(Mul3 op, R45 y, StepZIDS ys, R0123 x, StepZIDS xs, Ax a) {
        u16 address_y = RnAddressAndModify(y.Index(), ys.GetName());
        u16 address_x = RnAddressAndModify(x.Index(), xs.GetName());
        regs.y[0] = mem.DataRead(address_y);
        regs.x[0] = mem.DataRead(address_x);
        MulGeneric(op.GetName(), a);
    }
    void mul_y0_r6(Mul3 op, Ax a) {
        regs.x[0] = regs.r[6];
        MulGeneric(op.GetName(), a);
    }
    void mul_y0(Mul2 op, MemImm8 x, Ax a) {
        regs.x[0] = LoadFromMemory(x);
        MulGeneric(op.GetName(), a);
    }

    void mpyi(Imm8s x) {
        regs.x[0] = x.Signed16();
        DoMultiplication(0, true, true);
    }

    void msu(R45 y, StepZIDS ys, R0123 x, StepZIDS xs, Ax a) {
        u16 yi = RnAddressAndModify(y.Index(), ys.GetName());
        u16 xi = RnAddressAndModify(x.Index(), xs.GetName());
        u64 value = GetAcc(a.GetName());
        u64 product = ProductToBus40(Px{0});
        u64 result = AddSub(value, product, true);
        SatAndSetAccAndFlag(a.GetName(), result);
        regs.y[0] = mem.DataRead(yi);
        regs.x[0] = mem.DataRead(xi);
        DoMultiplication(0, true, true);
    }
    void msu(Rn y, StepZIDS ys, Imm16 x, Ax a) {
        u16 yi = RnAddressAndModify(y.Index(), ys.GetName());
        u64 value = GetAcc(a.GetName());
        u64 product = ProductToBus40(Px{0});
        u64 result = AddSub(value, product, true);
        SatAndSetAccAndFlag(a.GetName(), result);
        regs.y[0] = mem.DataRead(yi);
        regs.x[0] = x.Unsigned16();
        DoMultiplication(0, true, true);
    }
    void msusu(ArRn2 x, ArStep2 xs, Ax a) {
        u16 xi = RnAddressAndModify(GetArRnUnit(x), GetArStep(xs));
        u64 value = GetAcc(a.GetName());
        u64 product = ProductToBus40(Px{0});
        u64 result = AddSub(value, product, true);
        SatAndSetAccAndFlag(a.GetName(), result);
        regs.x[0] = mem.DataRead(xi);
        DoMultiplication(0, false, true);
    }
    void mac_x1to0(Ax a) {
        u64 value = GetAcc(a.GetName());
        u64 product = ProductToBus40(Px{0});
        u64 result = AddSub(value, product, false);
        SatAndSetAccAndFlag(a.GetName(), result);
        regs.x[0] = regs.x[1];
        DoMultiplication(0, true, true);
    }
    void mac1(ArpRn1 xy, ArpStep1 xis, ArpStep1 yjs, Ax a) {
        auto [ui, uj] = GetArpRnUnit(xy);
        auto [si, sj] = GetArpStep(xis, yjs);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 value = GetAcc(a.GetName());
        u64 product = ProductToBus40(Px{1});
        u64 result = AddSub(value, product, false);
        SatAndSetAccAndFlag(a.GetName(), result);
        regs.x[1] = mem.DataRead(i);
        regs.y[1] = mem.DataRead(j);
        DoMultiplication(1, true, true);
    }

    void modr(Rn a, StepZIDS as) {
        u32 unit = a.Index();
        RnAndModify(unit, as.GetName());
        regs.fr = regs.r[unit] == 0;
    }
    void modr_dmod(Rn a, StepZIDS as) {
        u32 unit = a.Index();
        RnAndModify(unit, as.GetName(), true);
        regs.fr = regs.r[unit] == 0;
    }
    void modr_i2(Rn a) {
        u32 unit = a.Index();
        RnAndModify(unit, StepValue::Increase2Mode1);
        regs.fr = regs.r[unit] == 0;
    }
    void modr_i2_dmod(Rn a) {
        u32 unit = a.Index();
        RnAndModify(unit, StepValue::Increase2Mode1, true);
        regs.fr = regs.r[unit] == 0;
    }
    void modr_d2(Rn a) {
        u32 unit = a.Index();
        RnAndModify(unit, StepValue::Decrease2Mode1);
        regs.fr = regs.r[unit] == 0;
    }
    void modr_d2_dmod(Rn a) {
        u32 unit = a.Index();
        RnAndModify(unit, StepValue::Decrease2Mode1, true);
        regs.fr = regs.r[unit] == 0;
    }
    void modr_eemod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        u32 uniti, unitj;
        StepValue stepi, stepj;
        std::tie(uniti, unitj) = GetArpRnUnit(a);
        std::tie(stepi, stepj) = GetArpStep(asi, asj);
        RnAndModify(uniti, stepi);
        RnAndModify(unitj, stepj);
    }
    void modr_edmod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        u32 uniti, unitj;
        StepValue stepi, stepj;
        std::tie(uniti, unitj) = GetArpRnUnit(a);
        std::tie(stepi, stepj) = GetArpStep(asi, asj);
        RnAndModify(uniti, stepi);
        RnAndModify(unitj, stepj, true);
    }
    void modr_demod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        u32 uniti, unitj;
        StepValue stepi, stepj;
        std::tie(uniti, unitj) = GetArpRnUnit(a);
        std::tie(stepi, stepj) = GetArpStep(asi, asj);
        RnAndModify(uniti, stepi, true);
        RnAndModify(unitj, stepj);
    }
    void modr_ddmod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        u32 uniti, unitj;
        StepValue stepi, stepj;
        std::tie(uniti, unitj) = GetArpRnUnit(a);
        std::tie(stepi, stepj) = GetArpStep(asi, asj);
        RnAndModify(uniti, stepi, true);
        RnAndModify(unitj, stepj, true);
    }

    void movd(R0123 a, StepZIDS as, R45 b, StepZIDS bs) {
        u16 address_s = RnAddressAndModify(a.Index(), as.GetName());
        u32 address_d = RnAddressAndModify(b.Index(), bs.GetName());
        address_d |= (u32)regs.pcmhi << 16;
        mem.ProgramWrite(address_d, mem.DataRead(address_s));
    }
    void movp(Axl a, Register b) {
        u32 address = RegToBus16(a.GetName());
        address |= (u32)regs.pcmhi << 16;
        u16 value = mem.ProgramRead(address);
        RegFromBus16(b.GetName(), value);
    }
    void movp(Ax a, Register b) {
        u32 address = GetAcc(a.GetName()) & 0x3FFFF; // no saturation
        u16 value = mem.ProgramRead(address);
        RegFromBus16(b.GetName(), value);
    }
    void movp(Rn a, StepZIDS as, R0123 b, StepZIDS bs) {
        u32 address_s = RnAddressAndModify(a.Index(), as.GetName());
        u16 address_d = RnAddressAndModify(b.Index(), bs.GetName());
        address_s |= (u32)regs.pcmhi << 16;
        mem.DataWrite(address_d, mem.ProgramRead(address_s));
    }
    void movpdw(Ax a) {
        u32 address = GetAcc(a.GetName()) & 0x3FFFF; // no saturation
        // the endianess doesn't seem to be affected by regs.cpc
        u16 h = mem.ProgramRead(address);
        u16 l = mem.ProgramRead(address + 1);
        SetPC(l | ((u32)h << 16));
    }

    void mov(Ab a, Ab b) {
        u64 value = GetAcc(a.GetName());
        SatAndSetAccAndFlag(b.GetName(), value);
    }
    void mov_dvm(Abl a) {
        UNREACHABLE();
    }
    void mov_x0(Abl a) {
        u16 value16 = RegToBus16(a.GetName(), true);
        regs.x[0] = value16;
    }
    void mov_x1(Abl a) {
        u16 value16 = RegToBus16(a.GetName(), true);
        regs.x[1] = value16;
    }
    void mov_y1(Abl a) {
        u16 value16 = RegToBus16(a.GetName(), true);
        regs.y[1] = value16;
    }

    void StoreToMemory(MemImm8 addr, u16 value) {
        mem.DataWrite(addr.Unsigned16() + (regs.page << 8), value);
    }
    void StoreToMemory(MemImm16 addr, u16 value) {
        mem.DataWrite(addr.Unsigned16(), value);
    }
    void StoreToMemory(MemR7Imm16 addr, u16 value) {
        mem.DataWrite(addr.Unsigned16() + regs.r[7], value);
    }
    void StoreToMemory(MemR7Imm7s addr, u16 value) {
        mem.DataWrite(addr.Signed16() + regs.r[7], value);
    }

    void mov(Ablh a, MemImm8 b) {
        u16 value16 = RegToBus16(a.GetName(), true);
        StoreToMemory(b, value16);
    }
    void mov(Axl a, MemImm16 b) {
        u16 value16 = RegToBus16(a.GetName(), true);
        StoreToMemory(b, value16);
    }
    void mov(Axl a, MemR7Imm16 b) {
        u16 value16 = RegToBus16(a.GetName(), true);
        StoreToMemory(b, value16);
    }
    void mov(Axl a, MemR7Imm7s b) {
        u16 value16 = RegToBus16(a.GetName(), true);
        StoreToMemory(b, value16);
    }

    u16 LoadFromMemory(MemImm8 addr) {
        return mem.DataRead(addr.Unsigned16() + (regs.page << 8));
    }
    u16 LoadFromMemory(MemImm16 addr) {
        return mem.DataRead(addr.Unsigned16());
    }
    u16 LoadFromMemory(MemR7Imm16 addr) {
        return mem.DataRead(addr.Unsigned16() + regs.r[7]);
    }
    u16 LoadFromMemory(MemR7Imm7s addr) {
        return mem.DataRead(addr.Signed16() + regs.r[7]);
    }

    void mov(MemImm16 a, Ax b) {
        u16 value = LoadFromMemory(a);
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemImm8 a, Ab b) {
        u16 value = LoadFromMemory(a);
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemImm8 a, Ablh b) {
        u16 value = LoadFromMemory(a);
        RegFromBus16(b.GetName(), value);
    }
    void mov_eu(MemImm8 a, Axh b) {
        u16 value = LoadFromMemory(a);
        u64 acc = GetAcc(b.GetName());
        acc &= 0xFFFF'FFFF'0000'0000;
        acc |= (u64)value << 16;
        SetAccAndFlag(b.GetName(), acc); // ?
    }
    void mov(MemImm8 a, RnOld b) {
        u16 value = LoadFromMemory(a);
        RegFromBus16(b.GetName(), value);
    }
    void mov_sv(MemImm8 a) {
        u16 value = LoadFromMemory(a);
        regs.sv = value;
    }
    void mov_dvm_to(Ab b) {
        UNREACHABLE();
    }
    void mov_icr_to(Ab b) {
        u16 value = regs.Get<icr>();
        RegFromBus16(b.GetName(), value);
    }
    void mov(Imm16 a, Bx b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov(Imm16 a, Register b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_icr(Imm5 a) {
        u16 value = regs.Get<icr>();
        value &= ~0x1F;
        value |= a.Unsigned16();
        regs.Set<icr>(value);
    }
    void mov(Imm8s a, Axh b) {
        u16 value = a.Signed16();
        RegFromBus16(b.GetName(), value);
    }
    void mov(Imm8s a, RnOld b) {
        u16 value = a.Signed16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_sv(Imm8s a) {
        u16 value = a.Signed16();
        regs.sv = value;
    }
    void mov(Imm8 a, Axl b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemR7Imm16 a, Ax b) {
        u16 value = LoadFromMemory(a);
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemR7Imm7s a, Ax b) {
        u16 value = LoadFromMemory(a);
        RegFromBus16(b.GetName(), value);
    }
    void mov(Rn a, StepZIDS as, Bx b) {
        u16 address = RnAddressAndModify(a.Index(), as.GetName());
        u16 value = mem.DataRead(address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(Rn a, StepZIDS as, Register b) {
        u16 address = RnAddressAndModify(a.Index(), as.GetName());
        u16 value = mem.DataRead(address);
        RegFromBus16(b.GetName(), value);
    }
    void mov_memsp_to(Register b) {
        u16 value = mem.DataRead(regs.sp);
        RegFromBus16(b.GetName(), value);
    }
    void mov_mixp_to(Register b) {
        u16 value = regs.mixp;
        RegFromBus16(b.GetName(), value);
    }
    void mov(RnOld a, MemImm8 b) {
        u16 value = RegToBus16(a.GetName());
        StoreToMemory(b, value);
    }
    void mov_icr(Register a) {
        u16 value = RegToBus16(a.GetName(), true);
        regs.Set<icr>(value);
    }
    void mov_mixp(Register a) {
        u16 value = RegToBus16(a.GetName(), true);
        regs.mixp = value;
    }
    void mov(Register a, Rn b, StepZIDS bs) {
        // a = a0 or a1 is overrided
        u16 value = RegToBus16(a.GetName(), true);
        u16 address = RnAddressAndModify(b.Index(), bs.GetName());
        mem.DataWrite(address, value);
    }
    void mov(Register a, Bx b) {
        if (a.GetName() == RegName::p) {
            u64 value = ProductToBus40(Px{0});
            SatAndSetAccAndFlag(b.GetName(), value);
        } else if (a.GetName() == RegName::a0 || a.GetName() == RegName::a1) {
            // Is there any difference from the mov(Ab, Ab) instruction?
            u64 value = GetAcc(a.GetName());
            SatAndSetAccAndFlag(b.GetName(), value);
        } else {
            u16 value = RegToBus16(a.GetName(), true);
            RegFromBus16(b.GetName(), value);
        }
    }
    void mov(Register a, Register b) {
        // a = a0 or a1 is overrided
        if (a.GetName() == RegName::p) {
            // b loses its typical meaning in this case
            RegName b_name = b.GetNameForMovFromP();
            u64 value = ProductToBus40(Px{0});
            SatAndSetAccAndFlag(b_name, value);
        } else if (a.GetName() == RegName::pc) {
            if (b.GetName() == RegName::a0 || b.GetName() == RegName::a1) {
                SatAndSetAccAndFlag(b.GetName(), regs.pc);
            } else {
                RegFromBus16(b.GetName(), regs.pc & 0xFFFF);
            }
        } else {
            u16 value = RegToBus16(a.GetName(), true);
            RegFromBus16(b.GetName(), value);
        }
    }
    void mov_repc_to(Ab b) {
        u16 value = regs.repc;
        RegFromBus16(b.GetName(), value);
    }
    void mov_sv_to(MemImm8 b) {
        u16 value = regs.sv;
        StoreToMemory(b, value);
    }
    void mov_x0_to(Ab b) {
        u16 value = regs.x[0];
        RegFromBus16(b.GetName(), value);
    }
    void mov_x1_to(Ab b) {
        u16 value = regs.x[1];
        RegFromBus16(b.GetName(), value);
    }
    void mov_y1_to(Ab b) {
        u16 value = regs.y[1];
        RegFromBus16(b.GetName(), value);
    }
    void mov(Imm16 a, ArArp b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_r6(Imm16 a) {
        u16 value = a.Unsigned16();
        regs.r[6] = value;
    }
    void mov_repc(Imm16 a) {
        u16 value = a.Unsigned16();
        regs.repc = value;
    }
    void mov_stepi0(Imm16 a) {
        u16 value = a.Unsigned16();
        regs.stepi0 = value;
    }
    void mov_stepj0(Imm16 a) {
        u16 value = a.Unsigned16();
        regs.stepj0 = value;
    }
    void mov(Imm16 a, SttMod b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_prpage(Imm4 a) {
        regs.prpage = a.Unsigned16();
    }

    void mov_a0h_stepi0() {
        u16 value = RegToBus16(RegName::a0h, true);
        regs.stepi0 = value;
    }
    void mov_a0h_stepj0() {
        u16 value = RegToBus16(RegName::a0h, true);
        regs.stepj0 = value;
    }
    void mov_stepi0_a0h() {
        u16 value = regs.stepi0;
        RegFromBus16(RegName::a0h, value);
    }
    void mov_stepj0_a0h() {
        u16 value = regs.stepj0;
        RegFromBus16(RegName::a0h, value);
    }

    void mov_prpage(Abl a) {
        regs.prpage = (u16)(GetAcc(a.GetName()) & 0xF); // ?
    }
    void mov_repc(Abl a) {
        u16 value = RegToBus16(a.GetName(), true);
        regs.repc = value;
    }
    void mov(Abl a, ArArp b) {
        u16 value = RegToBus16(a.GetName(), true);
        RegFromBus16(b.GetName(), value);
    }
    void mov(Abl a, SttMod b) {
        u16 value = RegToBus16(a.GetName(), true);
        RegFromBus16(b.GetName(), value);
    }

    void mov_prpage_to(Abl b) {
        RegFromBus16(b.GetName(), regs.prpage); // ?
    }
    void mov_repc_to(Abl b) {
        u16 value = regs.repc;
        RegFromBus16(b.GetName(), value);
    }
    void mov(ArArp a, Abl b) {
        u16 value = RegToBus16(a.GetName());
        RegFromBus16(b.GetName(), value);
    }
    void mov(SttMod a, Abl b) {
        u16 value = RegToBus16(a.GetName());
        RegFromBus16(b.GetName(), value);
    }

    void mov_repc_to(ArRn1 b, ArStep1 bs) {
        u16 address = RnAddressAndModify(GetArRnUnit(b), GetArStep(bs));
        u16 value = regs.repc;
        mem.DataWrite(address, value);
    }
    void mov(ArArp a, ArRn1 b, ArStep1 bs) {
        u16 address = RnAddressAndModify(GetArRnUnit(b), GetArStep(bs));
        u16 value = RegToBus16(a.GetName());
        mem.DataWrite(address, value);
    }
    void mov(SttMod a, ArRn1 b, ArStep1 bs) {
        u16 address = RnAddressAndModify(GetArRnUnit(b), GetArStep(bs));
        u16 value = RegToBus16(a.GetName());
        mem.DataWrite(address, value);
    }

    void mov_repc(ArRn1 a, ArStep1 as) {
        u16 address = RnAddressAndModify(GetArRnUnit(a), GetArStep(as));
        u16 value = mem.DataRead(address);
        regs.repc = value;
    }
    void mov(ArRn1 a, ArStep1 as, ArArp b) {
        u16 address = RnAddressAndModify(GetArRnUnit(a), GetArStep(as));
        u16 value = mem.DataRead(address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(ArRn1 a, ArStep1 as, SttMod b) {
        u16 address = RnAddressAndModify(GetArRnUnit(a), GetArStep(as));
        u16 value = mem.DataRead(address);
        RegFromBus16(b.GetName(), value);
    }

    void mov_repc_to(MemR7Imm16 b) {
        u16 value = regs.repc;
        StoreToMemory(b, value);
    }
    void mov(ArArpSttMod a, MemR7Imm16 b) {
        u16 value = RegToBus16(a.GetName());
        StoreToMemory(b, value);
    }

    void mov_repc(MemR7Imm16 a) {
        u16 value = LoadFromMemory(a);
        regs.repc = value;
    }
    void mov(MemR7Imm16 a, ArArpSttMod b) {
        u16 value = LoadFromMemory(a);
        RegFromBus16(b.GetName(), value);
    }

    void mov_pc(Ax a) {
        u64 value = GetAcc(a.GetName());
        SetPC(value & 0xFFFFFFFF);
    }
    void mov_pc(Bx a) {
        u64 value = GetAcc(a.GetName());
        SetPC(value & 0xFFFFFFFF);
    }

    void mov_mixp_to(Bx b) {
        u16 value = regs.mixp;
        RegFromBus16(b.GetName(), value);
    }
    void mov_mixp_r6() {
        u16 value = regs.mixp;
        regs.r[6] = value;
    }
    void mov_p0h_to(Bx b) {
        u16 value = (ProductToBus40(Px{0}) >> 16) & 0xFFFF;
        RegFromBus16(b.GetName(), value);
    }
    void mov_p0h_r6() {
        u16 value = (ProductToBus40(Px{0}) >> 16) & 0xFFFF;
        regs.r[6] = value;
    }
    void mov_p0h_to(Register b) {
        u16 value = (ProductToBus40(Px{0}) >> 16) & 0xFFFF;
        RegFromBus16(b.GetName(), value);
    }
    void mov_p0(Ab a) {
        u32 value = GetAndSatAcc(a.GetName()) & 0xFFFFFFFF;
        ProductFromBus32(Px{0}, value);
    }
    void mov_p1_to(Ab b) {
        u64 value = ProductToBus40(Px{1});
        SatAndSetAccAndFlag(b.GetName(), value);
    }

    void mov2(Px a, ArRn2 b, ArStep2 bs) {
        u32 value = ProductToBus32_NoShift(a);
        u16 l = value & 0xFFFF;
        u16 h = (value >> 16) & 0xFFFF;
        u16 unit = GetArRnUnit(b);
        u16 address = RnAddressAndModify(unit, GetArStep(bs));
        u16 address2 = OffsetAddress(unit, address, GetArOffset(bs));
        // NOTE: keep the write order exactly like this.
        mem.DataWrite(address2, l);
        mem.DataWrite(address, h);
    }
    void mov2s(Px a, ArRn2 b, ArStep2 bs) {
        u64 value = ProductToBus40(a);
        u16 l = value & 0xFFFF;
        u16 h = (value >> 16) & 0xFFFF;
        u16 unit = GetArRnUnit(b);
        u16 address = RnAddressAndModify(unit, GetArStep(bs));
        u16 address2 = OffsetAddress(unit, address, GetArOffset(bs));
        // NOTE: keep the write order exactly like this.
        mem.DataWrite(address2, l);
        mem.DataWrite(address, h);
    }
    void mov2(ArRn2 a, ArStep2 as, Px b) {
        u16 unit = GetArRnUnit(a);
        u16 address = RnAddressAndModify(unit, GetArStep(as));
        u16 address2 = OffsetAddress(unit, address, GetArOffset(as));
        u16 l = mem.DataRead(address2);
        u16 h = mem.DataRead(address);
        u32 value = ((u32)h << 16) | l;
        ProductFromBus32(b, value);
    }
    void mova(Ab a, ArRn2 b, ArStep2 bs) {
        u64 value = GetAndSatAcc(a.GetName());
        u16 l = value & 0xFFFF;
        u16 h = (value >> 16) & 0xFFFF;
        u16 unit = GetArRnUnit(b);
        u16 address = RnAddressAndModify(unit, GetArStep(bs));
        u16 address2 = OffsetAddress(unit, address, GetArOffset(bs));
        // NOTE: keep the write order exactly like this. The second one overrides the first one if
        // the offset is zero.
        mem.DataWrite(address2, l);
        mem.DataWrite(address, h);
    }
    void mova(ArRn2 a, ArStep2 as, Ab b) {
        u16 unit = GetArRnUnit(a);
        u16 address = RnAddressAndModify(unit, GetArStep(as));
        u16 address2 = OffsetAddress(unit, address, GetArOffset(as));
        u16 l = mem.DataRead(address2);
        u16 h = mem.DataRead(address);
        u64 value = SignExtend<32, u64>(((u64)h << 16) | l);
        SatAndSetAccAndFlag(b.GetName(), value);
    }

    void mov_r6_to(Bx b) {
        u16 value = regs.r[6];
        RegFromBus16(b.GetName(), value);
    }
    void mov_r6_mixp() {
        u16 value = regs.r[6];
        regs.mixp = value;
    }
    void mov_r6_to(Register b) {
        u16 value = regs.r[6];
        RegFromBus16(b.GetName(), value);
    }
    void mov_r6(Register a) {
        u16 value = RegToBus16(a.GetName(), true);
        regs.r[6] = value;
    }
    void mov_memsp_r6() {
        u16 value = mem.DataRead(regs.sp);
        regs.r[6] = value;
    }
    void mov_r6_to(Rn b, StepZIDS bs) {
        u16 value = regs.r[6];
        u16 address = RnAddressAndModify(b.Index(), bs.GetName());
        mem.DataWrite(address, value);
    }
    void mov_r6(Rn a, StepZIDS as) {
        u16 address = RnAddressAndModify(a.Index(), as.GetName());
        u16 value = mem.DataRead(address);
        regs.r[6] = value;
    }

    void mov2_axh_m_y0_m(Axh a, ArRn2 b, ArStep2 bs) {
        u16 u = (u16)((GetAndSatAccNoFlag(a.GetName()) >> 16) & 0xFFFF);
        u16 v = regs.y[0];
        u16 unit = GetArRnUnit(b);
        u16 ua = RnAddressAndModify(unit, GetArStep(bs));
        u16 va = OffsetAddress(unit, ua, GetArOffset(bs));
        // keep the order
        mem.DataWrite(va, v);
        mem.DataWrite(ua, u);
    }

    void mov2_ax_mij(Ab a, ArpRn1 b, ArpStep1 bsi, ArpStep1 bsj) {
        auto [ui, uj] = GetArpRnUnit(b);
        auto [si, sj] = GetArpStep(bsi, bsj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 value = GetAndSatAccNoFlag(a.GetName());
        mem.DataWrite(i, (u16)((value >> 16) & 0xFFFF));
        mem.DataWrite(j, (u16)(value & 0xFFFF));
    }
    void mov2_ax_mji(Ab a, ArpRn1 b, ArpStep1 bsi, ArpStep1 bsj) {
        auto [ui, uj] = GetArpRnUnit(b);
        auto [si, sj] = GetArpStep(bsi, bsj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 value = GetAndSatAccNoFlag(a.GetName());
        mem.DataWrite(j, (u16)((value >> 16) & 0xFFFF));
        mem.DataWrite(i, (u16)(value & 0xFFFF));
    }
    void mov2_mij_ax(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        auto [ui, uj] = GetArpRnUnit(a);
        auto [si, sj] = GetArpStep(asi, asj);
        u16 h = mem.DataRead(RnAddressAndModify(ui, si));
        u16 l = mem.DataRead(RnAddressAndModify(uj, sj));
        u64 value = SignExtend<32, u64>(((u64)h << 16) | l);
        SetAcc(b.GetName(), value);
    }
    void mov2_mji_ax(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        auto [ui, uj] = GetArpRnUnit(a);
        auto [si, sj] = GetArpStep(asi, asj);
        u16 l = mem.DataRead(RnAddressAndModify(ui, si));
        u16 h = mem.DataRead(RnAddressAndModify(uj, sj));
        u64 value = SignExtend<32, u64>(((u64)h << 16) | l);
        SetAcc(b.GetName(), value);
    }
    void mov2_abh_m(Abh ax, Abh ay, ArRn1 b, ArStep1 bs) {
        u16 u = (u16)((GetAndSatAccNoFlag(ax.GetName()) >> 16) & 0xFFFF);
        u16 v = (u16)((GetAndSatAccNoFlag(ay.GetName()) >> 16) & 0xFFFF);
        u16 unit = GetArRnUnit(b);
        u16 ua = RnAddressAndModify(unit, GetArStep(bs));
        u16 va = OffsetAddress(unit, ua, GetArOffset(bs));
        // keep the order
        mem.DataWrite(va, v);
        mem.DataWrite(ua, u);
    }
    void exchange_iaj(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        auto [ui, uj] = GetArpRnUnit(b);
        auto [si, sj] = GetArpStep(bsi, bsj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 value = GetAndSatAccNoFlag(a.GetName());
        mem.DataWrite(j, (u16)((value >> 16) & 0xFFFF));
        value = SignExtend<32, u64>((u64)mem.DataRead(i) << 16);
        SetAcc(a.GetName(), value);
    }
    void exchange_riaj(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        auto [ui, uj] = GetArpRnUnit(b);
        auto [si, sj] = GetArpStep(bsi, bsj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 value = GetAndSatAccNoFlag(a.GetName());
        mem.DataWrite(j, (u16)((value >> 16) & 0xFFFF));
        value = SignExtend<32, u64>(((u64)mem.DataRead(i) << 16) | 0x8000);
        SetAcc(a.GetName(), value);
    }
    void exchange_jai(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        auto [ui, uj] = GetArpRnUnit(b);
        auto [si, sj] = GetArpStep(bsi, bsj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 value = GetAndSatAccNoFlag(a.GetName());
        mem.DataWrite(i, (u16)((value >> 16) & 0xFFFF));
        value = SignExtend<32, u64>((u64)mem.DataRead(j) << 16);
        SetAcc(a.GetName(), value);
    }
    void exchange_rjai(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        auto [ui, uj] = GetArpRnUnit(b);
        auto [si, sj] = GetArpStep(bsi, bsj);
        u16 i = RnAddressAndModify(ui, si);
        u16 j = RnAddressAndModify(uj, sj);
        u64 value = GetAndSatAccNoFlag(a.GetName());
        mem.DataWrite(i, (u16)((value >> 16) & 0xFFFF));
        value = SignExtend<32, u64>(((u64)mem.DataRead(j) << 16) | 0x8000);
        SetAcc(a.GetName(), value);
    }

    void ShiftBus40(u64 value, u16 sv, RegName dest) {
        value &= 0xFF'FFFF'FFFF;
        u64 original_sign = value >> 39;
        if ((sv >> 15) == 0) {
            // left shift
            if (sv >= 40) {
                if (regs.s == 0) {
                    regs.fv = value != 0;
                    if (regs.fv) {
                        regs.fvl = 1;
                    }
                }
                value = 0;
                regs.fc0 = 0;
            } else {
                if (regs.s == 0) {
                    regs.fv = SignExtend<40>(value) != SignExtend(value, 40 - sv);
                    if (regs.fv) {
                        regs.fvl = 1;
                    }
                }
                value <<= sv;
                regs.fc0 = (value & ((u64)1 << 40)) != 0;
            }
        } else {
            // right shift
            u16 nsv = ~sv + 1;
            if (nsv >= 40) {
                if (regs.s == 0) {
                    regs.fc0 = (value >> 39) & 1;
                    value = regs.fc0 ? 0xFF'FFFF'FFFF : 0;
                } else {
                    value = 0;
                    regs.fc0 = 0;
                }
            } else {
                regs.fc0 = (value & ((u64)1 << (nsv - 1))) != 0;
                value >>= nsv;
                if (regs.s == 0) {
                    value = SignExtend(value, 40 - nsv);
                }
            }

            if (regs.s == 0) {
                regs.fv = 0;
            }
        }

        value = SignExtend<40>(value);
        SetAccFlag(value);
        if (regs.s == 0 && regs.sata == 0) {
            if (regs.fv || SignExtend<32>(value) != value) {
                regs.flm = 1;
                value = original_sign == 1 ? 0xFFFF'FFFF'8000'0000 : 0x7FFF'FFFF;
            }
        }
        SetAcc(dest, value);
    }

    void movs(MemImm8 a, Ab b) {
        u64 value = SignExtend<16, u64>(LoadFromMemory(a));
        u16 sv = regs.sv;
        ShiftBus40(value, sv, b.GetName());
    }
    void movs(Rn a, StepZIDS as, Ab b) {
        u16 address = RnAddressAndModify(a.Index(), as.GetName());
        u64 value = SignExtend<16, u64>(mem.DataRead(address));
        u16 sv = regs.sv;
        ShiftBus40(value, sv, b.GetName());
    }
    void movs(Register a, Ab b) {
        u64 value = SignExtend<16, u64>(RegToBus16(a.GetName()));
        u16 sv = regs.sv;
        ShiftBus40(value, sv, b.GetName());
    }
    void movs_r6_to(Ax b) {
        u64 value = SignExtend<16, u64>(regs.r[6]);
        u16 sv = regs.sv;
        ShiftBus40(value, sv, b.GetName());
    }
    void movsi(RnOld a, Ab b, Imm5s s) {
        u64 value = SignExtend<16, u64>(RegToBus16(a.GetName()));
        u16 sv = s.Signed16();
        ShiftBus40(value, sv, b.GetName());
    }

    void movr(ArRn2 a, ArStep2 as, Abh b) {
        u16 value16 = mem.DataRead(RnAddressAndModify(GetArRnUnit(a), GetArStep(as)));
        u64 value = SignExtend<32, u64>((u64)value16 << 16);
        u64 result = AddSub(value, 0x8000, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void movr(Rn a, StepZIDS as, Ax b) {
        u16 value16 = mem.DataRead(RnAddressAndModify(a.Index(), as.GetName()));
        // Do 16-bit arithmetic. Flag C is set according to bit 16 but Flag V is always cleared
        // Looks like a hardware bug to me
        u64 result = (u64)value16 + 0x8000;
        regs.fc0 = (u16)(result >> 16);
        regs.fv = 0;
        result &= 0xFFFF;
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void movr(Register a, Ax b) {
        u64 result;
        if (a.GetName() == RegName::a0 || a.GetName() == RegName::a1) {
            u64 value = GetAcc(a.GetName());
            result = AddSub(value, 0x8000, false);
        } else if (a.GetName() == RegName::p) {
            u64 value = ProductToBus40(Px{0});
            result = AddSub(value, 0x8000, false);
        } else {
            u16 value16 = RegToBus16(a.GetName());
            result = (u64)value16 + 0x8000;
            regs.fc0 = (u16)(result >> 16);
            regs.fv = 0;
            result &= 0xFFFF;
        }
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void movr(Bx a, Ax b) {
        u64 value = GetAcc(a.GetName());
        u64 result = AddSub(value, 0x8000, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void movr_r6_to(Ax b) {
        u16 value16 = regs.r[6];
        u64 result = (u64)value16 + 0x8000;
        regs.fc0 = (u16)(result >> 16);
        regs.fv = 0;
        result &= 0xFFFF;
        SatAndSetAccAndFlag(b.GetName(), result);
    }

    u16 Exp(u64 value) {
        u64 sign = (value >> 39) & 1;
        u16 bit = 38, count = 0;
        while (true) {
            if (((value >> bit) & 1) != sign)
                break;
            ++count;
            if (bit == 0)
                break;
            --bit;
        }
        return count - 8;
    }

    void ExpStore(Ax b) {
        SetAcc(b.GetName(), SignExtend<16, u64>(regs.sv));
    }

    void exp(Bx a) {
        u64 value = GetAcc(a.GetName());
        regs.sv = Exp(value);
    }
    void exp(Bx a, Ax b) {
        exp(a);
        ExpStore(b);
    }
    void exp(Rn a, StepZIDS as) {
        u16 address = RnAddressAndModify(a.Index(), as.GetName());
        u64 value = SignExtend<32>((u64)mem.DataRead(address) << 16);
        regs.sv = Exp(value);
    }
    void exp(Rn a, StepZIDS as, Ax b) {
        exp(a, as);
        ExpStore(b);
    }
    void exp(Register a) {
        u64 value;
        if (a.GetName() == RegName::a0 || a.GetName() == RegName::a1) {
            value = GetAcc(a.GetName());
        } else {
            // RegName::p follows the usual rule
            value = SignExtend<32>((u64)RegToBus16(a.GetName()) << 16);
        }
        regs.sv = Exp(value);
    }
    void exp(Register a, Ax b) {
        exp(a);
        ExpStore(b);
    }
    void exp_r6() {
        u64 value = SignExtend<32>((u64)RegToBus16(RegName::r6) << 16);
        regs.sv = Exp(value);
    }
    void exp_r6(Ax b) {
        exp_r6();
        ExpStore(b);
    }

    void lim(Ax a, Ax b) {
        u64 value = GetAcc(a.GetName());
        value = SaturateAcc(value);
        SetAccAndFlag(b.GetName(), value);
    }

    void vtrclr0() {
        regs.vtr0 = 0;
    }
    void vtrclr1() {
        regs.vtr1 = 0;
    }
    void vtrclr() {
        regs.vtr0 = 0;
        regs.vtr1 = 0;
    }
    void vtrmov0(Axl a) {
        SatAndSetAccAndFlag(a.GetName(), regs.vtr0);
    }
    void vtrmov1(Axl a) {
        SatAndSetAccAndFlag(a.GetName(), regs.vtr1);
    }
    void vtrmov(Axl a) {
        SatAndSetAccAndFlag(a.GetName(), (regs.vtr1 & 0xFF00) | (regs.vtr0 >> 8));
    }
    void vtrshr() {
        // TODO: This instruction has one cycle delay on vtr0, but not on vtr1
        regs.vtr0 = (regs.vtr0 >> 1) | (regs.fc0 << 15);
        regs.vtr1 = (regs.vtr1 >> 1) | (regs.fc1 << 15);
    }

    void clrp0() {
        ProductFromBus32(Px{0}, 0);
    }
    void clrp1() {
        ProductFromBus32(Px{1}, 0);
    }
    void clrp() {
        ProductFromBus32(Px{0}, 0);
        ProductFromBus32(Px{1}, 0);
    }

    void max_ge(Ax a, StepZIDS bs) {
        u64 u = GetAcc(a.GetName());
        u64 v = GetAcc(CounterAcc(a.GetName()));
        u64 d = v - u;
        u16 r0 = RnAndModify(0, bs.GetName());
        if (((d >> 63) & 1) == 0) {
            regs.fm = 1;
            regs.mixp = r0;
            SetAcc(a.GetName(), v);
        } else {
            regs.fm = 0;
        }
    }
    void max_gt(Ax a, StepZIDS bs) {
        u64 u = GetAcc(a.GetName());
        u64 v = GetAcc(CounterAcc(a.GetName()));
        u64 d = v - u;
        u16 r0 = RnAndModify(0, bs.GetName());
        if (((d >> 63) & 1) == 0 && d != 0) {
            regs.fm = 1;
            regs.mixp = r0;
            SetAcc(a.GetName(), v);
        } else {
            regs.fm = 0;
        }
    }
    void min_le(Ax a, StepZIDS bs) {
        u64 u = GetAcc(a.GetName());
        u64 v = GetAcc(CounterAcc(a.GetName()));
        u64 d = v - u;
        u16 r0 = RnAndModify(0, bs.GetName());
        if (((d >> 63) & 1) == 1 || d == 0) {
            regs.fm = 1;
            regs.mixp = r0;
            SetAcc(a.GetName(), v);
        } else {
            regs.fm = 0;
        }
    }
    void min_lt(Ax a, StepZIDS bs) {
        u64 u = GetAcc(a.GetName());
        u64 v = GetAcc(CounterAcc(a.GetName()));
        u64 d = v - u;
        u16 r0 = RnAndModify(0, bs.GetName());
        if (((d >> 63) & 1) == 1) {
            regs.fm = 1;
            regs.mixp = r0;
            SetAcc(a.GetName(), v);
        } else {
            regs.fm = 0;
        }
    }

    void max_ge_r0(Ax a, StepZIDS bs) {
        u64 u = GetAcc(a.GetName());
        u16 r0 = RnAndModify(0, bs.GetName());
        u64 v = SignExtend<16, u64>(mem.DataRead(RnAddress(0, r0)));
        u64 d = v - u;
        if (((d >> 63) & 1) == 0) {
            regs.fm = 1;
            regs.mixp = r0;
            SetAcc(a.GetName(), v);
        } else {
            regs.fm = 0;
        }
    }
    void max_gt_r0(Ax a, StepZIDS bs) {
        u64 u = GetAcc(a.GetName());
        u16 r0 = RnAndModify(0, bs.GetName());
        u64 v = SignExtend<16, u64>(mem.DataRead(RnAddress(0, r0)));
        u64 d = v - u;
        if (((d >> 63) & 1) == 0 && d != 0) {
            regs.fm = 1;
            regs.mixp = r0;
            SetAcc(a.GetName(), v);
        } else {
            regs.fm = 0;
        }
    }
    void min_le_r0(Ax a, StepZIDS bs) {
        u64 u = GetAcc(a.GetName());
        u16 r0 = RnAndModify(0, bs.GetName());
        u64 v = SignExtend<16, u64>(mem.DataRead(RnAddress(0, r0)));
        u64 d = v - u;
        if (((d >> 63) & 1) == 1 || d == 0) {
            regs.fm = 1;
            regs.mixp = r0;
            SetAcc(a.GetName(), v);
        } else {
            regs.fm = 0;
        }
    }
    void min_lt_r0(Ax a, StepZIDS bs) {
        u64 u = GetAcc(a.GetName());
        u16 r0 = RnAndModify(0, bs.GetName());
        u64 v = SignExtend<16, u64>(mem.DataRead(RnAddress(0, r0)));
        u64 d = v - u;
        if (((d >> 63) & 1) == 1) {
            regs.fm = 1;
            regs.mixp = r0;
            SetAcc(a.GetName(), v);
        } else {
            regs.fm = 0;
        }
    }

    void divs(MemImm8 a, Ax b) {
        u16 da = LoadFromMemory(a);
        u64 db = GetAcc(b.GetName());
        u64 value = db - ((u64)da << 15);
        if (value >> 63) {
            SetAccAndFlag(b.GetName(), SignExtend<40>(db << 1));
        } else {
            SetAccAndFlag(b.GetName(), SignExtend<40>((value << 1) + 1));
        }
    }

    void sqr_sqr_add3(Ab a, Ab b) {
        u64 value = GetAcc(a.GetName());
        app(b, SumBase::Acc, false, false, false, false);
        regs.x[0] = regs.y[0] = (u16)((value >> 16) & 0xFFFF);
        regs.x[1] = regs.y[1] = (u16)(value & 0xFFFF);
        DoMultiplication(0, true, true);
        DoMultiplication(1, true, true);
    }

    void sqr_sqr_add3(ArRn2 a, ArStep2 as, Ab b) {
        app(b, SumBase::Acc, false, false, false, false);
        u16 unit = GetArRnUnit(a);
        u16 address0 = RnAddressAndModify(unit, GetArStep(as));
        u16 address1 = OffsetAddress(unit, address0, GetArOffset(as));
        regs.x[0] = regs.y[0] = mem.DataRead(address0);
        regs.x[1] = regs.y[1] = mem.DataRead(address1);
        DoMultiplication(0, true, true);
        DoMultiplication(1, true, true);
    }

    void sqr_mpysu_add3a(Ab a, Ab b) {
        u64 value = GetAcc(a.GetName());
        app(b, SumBase::Acc, false, false, false, true);
        regs.x[0] = regs.y[0] = regs.y[1] = (u16)((value >> 16) & 0xFFFF);
        regs.x[1] = (u16)(value & 0xFFFF);
        DoMultiplication(0, true, true);
        DoMultiplication(1, false, true);
    }

    void cmp(Ax a, Bx b) {}
    void cmp_b0_b1() {}
    void cmp_b1_b0() {}
    void cmp(Bx a, Ax b) {}
    void cmp_p1_to(Ax b) {}

    void MinMaxVtr(RegName a, RegName b, bool min) {}

    void max2_vtr(Ax a) {}
    void min2_vtr(Ax a) {}
    void max2_vtr(Ax a, Bx b) {}
    void min2_vtr(Ax a, Bx b) {}
    void max2_vtr_movl(Ax a, Bx b, ArRn1 c, ArStep1 cs) {}
    void max2_vtr_movh(Ax a, Bx b, ArRn1 c, ArStep1 cs) {}
    void max2_vtr_movl(Bx a, Ax b, ArRn1 c, ArStep1 cs) {}
    void max2_vtr_movh(Bx a, Ax b, ArRn1 c, ArStep1 cs) {}
    void min2_vtr_movl(Ax a, Bx b, ArRn1 c, ArStep1 cs) {}
    void min2_vtr_movh(Ax a, Bx b, ArRn1 c, ArStep1 cs) {}
    void min2_vtr_movl(Bx a, Ax b, ArRn1 c, ArStep1 cs) {}
    void min2_vtr_movh(Bx a, Ax b, ArRn1 c, ArStep1 cs) {}
    void max2_vtr_movij(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {}
    void max2_vtr_movji(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {}
    void min2_vtr_movij(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {}
    void min2_vtr_movji(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {}

    template <typename ArpStepX>
    void mov_sv_app(ArRn1 a, ArpStepX as, Bx b, SumBase base, bool sub_p0, bool p0_align,
                    bool sub_p1, bool p1_align) {}

    void CodebookSearch(u16 u, u16 v, u16 r, CbsCond c) {}

    void cbs(Axh a, CbsCond c) {}
    void cbs(Axh a, Bxh b, CbsCond c) {}
    void cbs(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, CbsCond c) {}

    void mma(RegName a, bool x0_sign, bool y0_sign, bool x1_sign, bool y1_sign, SumBase base,
             bool sub_p0, bool p0_align, bool sub_p1, bool p1_align) {}

    template <typename ArpRnX, typename ArpStepX>
    void mma(ArpRnX xy, ArpStepX i, ArpStepX j, bool dmodi, bool dmodj, RegName a, bool x0_sign,
             bool y0_sign, bool x1_sign, bool y1_sign, SumBase base, bool sub_p0, bool p0_align,
             bool sub_p1, bool p1_align) {}

    void mma_mx_xy(ArRn1 y, ArStep1 ys, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {}

    void mma_xy_mx(ArRn1 y, ArStep1 ys, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {}

    void mma_my_my(ArRn1 x, ArStep1 xs, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {}

    void mma_mov(Axh u, Bxh v, ArRn1 w, ArStep1 ws, RegName a, bool x0_sign, bool y0_sign,
                 bool x1_sign, bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                 bool p1_align) {}

    void mma_mov(ArRn2 w, ArStep1 ws, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                 bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                 bool p1_align) {}

    void addhp(ArRn2 a, ArStep2 as, Px b, Ax c) {}

    void mov_ext0(Imm8s a) {}
    void mov_ext1(Imm8s a) {}
    void mov_ext2(Imm8s a) {}
    void mov_ext3(Imm8s a) {}
};
