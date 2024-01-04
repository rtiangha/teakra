#pragma once
#include <utility>
#include <atomic>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include "bit.h"
#include "decoder.h"
#include "ir/ir_emitter.h"
#include "matcher.h"
#include "operand.h"
#include "register.h"

namespace Teakra {

class UnimplementedException : public std::runtime_error {
public:
    UnimplementedException() : std::runtime_error("unimplemented") {}
};

class TranslatorVisitor {
public:
    TranslatorVisitor(RegisterState& regs, IR::Block& block_, IR::LocationDescriptor descriptor) : ir(block_, descriptor), regs(regs) {}

    IR::IREmitter ir;

    void PushPC() {
        const IR::U16 sp = ir.GetRegister16(RegName::sp);
        const IR::U16 addr0 = ir.Sub16(sp, ir.Imm16(1));
        const IR::U16 addr1 = ir.Sub16(addr0, ir.Imm16(1));

        const u32 pc = ir.PC();
        const u16 l = pc & 0xFFFF;
        const u16 h = pc >> 16;

        if (regs.cpc == 1) {
            ir.WriteMemory16(addr0, ir.Imm16(h));
            ir.WriteMemory16(addr1, ir.Imm16(l));
        } else {
            ir.WriteMemory16(addr0, ir.Imm16(l));
            ir.WriteMemory16(addr1, ir.Imm16(h));
        }

        ir.SetRegister16(RegName::sp, addr1);
    }

    void PopPC() {
        const IR::U16 sp = ir.GetRegister16(RegName::sp);
        const IR::U16 addr0 = ir.Add16(sp, ir.Imm16(1));
        const IR::U16 addr1 = ir.Add16(addr0, ir.Imm16(1));

        IR::U16 l, h;
        if (regs.cpc == 1) {
            l = ir.ReadMemory16(addr0);
            h = ir.ReadMemory16(addr1);
        } else {
            h = ir.ReadMemory16(addr0);
            l = ir.ReadMemory16(addr1);
        }

        ir.SetRegister16(RegName::pc, ir.Pack2x16To1x32(l, h));
        // ir.SetTerm(IR::Term::ReturnToDispatch{});
    }

    void SetPC(u32 new_pc) {
        ASSERT(new_pc < 0x40000);
        regs.pc = new_pc;
    }

    using instruction_return_type = void;

    void nop() {
        // literally nothing
    }

    void norm(Ax a, Rn b, StepZIDS bs) {

    }
    void swap(SwapType swap) {
        RegName s0, d0, s1, d1;
        IR::U64 u, v;
        switch (swap.GetName()) {
        case SwapTypeValue::a0b0:
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b0;
            break;
        case SwapTypeValue::a0b1:
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b1;
            break;
        case SwapTypeValue::a1b0:
            s0 = d1 = RegName::a1;
            s1 = d0 = RegName::b0;
            break;
        case SwapTypeValue::a1b1:
            s0 = d1 = RegName::a1;
            s1 = d0 = RegName::b1;
            break;
        case SwapTypeValue::a0b0a1b1:
            u = ir.GetAcc(RegName::a1);
            v = ir.GetAcc(RegName::b1);
            SatAndSetAccAndFlag(RegName::a1, v);
            SatAndSetAccAndFlag(RegName::b1, u);
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b0;
            break;
        case SwapTypeValue::a0b1a1b0:
            u = ir.GetAcc(RegName::a1);
            v = ir.GetAcc(RegName::b0);
            SatAndSetAccAndFlag(RegName::a1, v);
            SatAndSetAccAndFlag(RegName::b0, u);
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b1;
            break;
        case SwapTypeValue::a0b0a1:
            s0 = RegName::a0;
            d0 = s1 = RegName::b0;
            d1 = RegName::a1;
            break;
        case SwapTypeValue::a0b1a1:
            s0 = RegName::a0;
            d0 = s1 = RegName::b1;
            d1 = RegName::a1;
            break;
        case SwapTypeValue::a1b0a0:
            s0 = RegName::a1;
            d0 = s1 = RegName::b0;
            d1 = RegName::a0;
            break;
        case SwapTypeValue::a1b1a0:
            s0 = RegName::a1;
            d0 = s1 = RegName::b1;
            d1 = RegName::a0;
            break;
        case SwapTypeValue::b0a0b1:
            s0 = d1 = RegName::a0;
            d0 = RegName::b1;
            s1 = RegName::b0;
            break;
        case SwapTypeValue::b0a1b1:
            s0 = d1 = RegName::a1;
            d0 = RegName::b1;
            s1 = RegName::b0;
            break;
        case SwapTypeValue::b1a0b0:
            s0 = d1 = RegName::a0;
            d0 = RegName::b0;
            s1 = RegName::b1;
            break;
        case SwapTypeValue::b1a1b0:
            s0 = d1 = RegName::a1;
            d0 = RegName::b0;
            s1 = RegName::b1;
            break;
        default:
            UNREACHABLE();
        }
        u = ir.GetAcc(s0);
        v = ir.GetAcc(s1);
        SatAndSetAccAndFlag(d0, u);
        SatAndSetAccAndFlag(d1, v); // only this one affects flags (except for fl)
    }
    void trap() {
        throw UnimplementedException();
    }

    void DoMultiplication(u32 unit, bool x_sign, bool y_sign) {
        IR::U32 x = ir.GetXFactor(unit);
        IR::U32 y = ir.GetYFactor(unit);
        if (regs.hwm == 1 || (regs.hwm == 3 && unit == 0)) {
            y = ir.LogicalShiftRight32(y, ir.Imm8(8));
        } else if (regs.hwm == 2 || (regs.hwm == 3 && unit == 1)) {
            y = ir.And32(y, ir.Imm32(0xFF));
        }
        if (x_sign) {
            x = ir.SignExtend32(x, ir.Imm8(16));
        }
        if (y_sign) {
            y = ir.SignExtend32(y, ir.Imm8(16));
        }
        const IR::U32 product = ir.Mul32(x, y);
        ir.SetProduct(unit, product);
        if (x_sign || y_sign)
            regs.pe[unit] = regs.p[unit] >> 31;
        else
            regs.pe[unit] = 0;
    }

private:
    RegisterState& regs;

    IR::U64 GetAndSatAcc(RegName name) {
        IR::U64 value = ir.GetAcc(name);
        if (!regs.sat) {
            return ir.SaturateAcc(value, true);
        }
        return value;
    }

    IR::U64 GetAndSatAccNoFlag(RegName name) {
        IR::U64 value = ir.GetAcc(name);
        if (!regs.sat) {
            return ir.SaturateAcc(value, false);
        }
        return value;
    }

    IR::U16 RegToBus16(RegName reg, bool enable_sat_for_mov = false) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            // get aXl, but unlike using RegName::aXl, this does never saturate.
            // This only happen to insturctions using "Register" operand,
            // and doesn't apply to all instructions. Need test and special check.
            return ir.ExtractHalf64(ir.GetAcc(reg), 0);
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            if (enable_sat_for_mov) {
                return ir.ExtractHalf64(GetAndSatAcc(reg), 0);
            }
            return ir.ExtractHalf64(ir.GetAcc(reg), 0);
        case RegName::a0h:
        case RegName::a1h:
        case RegName::b0h:
        case RegName::b1h:
            if (enable_sat_for_mov) {
                return ir.ExtractHalf64(GetAndSatAcc(reg), 1);
            }
            return ir.ExtractHalf64(ir.GetAcc(reg), 1);
        case RegName::a0e:
        case RegName::a1e:
        case RegName::b0e:
        case RegName::b1e:
            UNREACHABLE();

        case RegName::r0:
        case RegName::r1:
        case RegName::r2:
        case RegName::r3:
        case RegName::r4:
        case RegName::r5:
        case RegName::r6:
        case RegName::r7:
            return ir.GetRegister16(reg);

        case RegName::y0:
            return ir.GetYFactor(0);
        case RegName::p:
            // This only happen to insturctions using "Register" operand,
            // and doesn't apply to all instructions. Need test and special check.
            return (ProductToBus40(Px{0}) >> 16) & 0xFFFF;

        case RegName::pc:
            UNREACHABLE();
        case RegName::sp:
            return regs.sp;
        case RegName::sv:
            return regs.sv;
        case RegName::lc:
            return regs.Lc();

        case RegName::ar0:
            return regs.Get<ar0>();
        case RegName::ar1:
            return regs.Get<ar1>();

        case RegName::arp0:
            return regs.Get<arp0>();
        case RegName::arp1:
            return regs.Get<arp1>();
        case RegName::arp2:
            return regs.Get<arp2>();
        case RegName::arp3:
            return regs.Get<arp3>();

        case RegName::ext0:
            return regs.ext[0];
        case RegName::ext1:
            return regs.ext[1];
        case RegName::ext2:
            return regs.ext[2];
        case RegName::ext3:
            return regs.ext[3];

        case RegName::stt0:
            return regs.Get<stt0>();
        case RegName::stt1:
            return regs.Get<stt1>();
        case RegName::stt2:
            return regs.Get<stt2>();

        case RegName::st0:
            return regs.Get<st0>();
        case RegName::st1:
            return regs.Get<st1>();
        case RegName::st2:
            return regs.Get<st2>();

        case RegName::cfgi:
            return regs.Get<cfgi>();
        case RegName::cfgj:
            return regs.Get<cfgj>();

        case RegName::mod0:
            return regs.Get<mod0>();
        case RegName::mod1:
            return regs.Get<mod1>();
        case RegName::mod2:
            return regs.Get<mod2>();
        case RegName::mod3:
            return regs.Get<mod3>();
        default:
            UNREACHABLE();
        }
    }

    void SetAcc(RegName name, u64 value) {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            regs.a[0] = value;
            break;
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            regs.a[1] = value;
            break;
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            regs.b[0] = value;
            break;
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            regs.b[1] = value;
            break;
        default:
            UNREACHABLE();
        }
    }

    void SatAndSetAccAndFlag(RegName name, const IR::U64& value) {
        ir.SetAccFlag(value);
        if (!regs.sata) {
            const IR::U64 sat = ir.SaturateAcc(value);
            ir.SetAcc(name, sat);
            return;
        }
        ir.SetAcc(name, value);
    }

    void SetAccAndFlag(RegName name, const IR::U64& value) {
        ir.SetAccFlag(value);
        ir.SetAcc(name, value);
    }

    void RegFromBus16(RegName reg, u16 value) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            SatAndSetAccAndFlag(reg, SignExtend<16, u64>(value));
            break;
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            SatAndSetAccAndFlag(reg, (u64)value);
            break;
        case RegName::a0h:
        case RegName::a1h:
        case RegName::b0h:
        case RegName::b1h:
            SatAndSetAccAndFlag(reg, SignExtend<32, u64>(value << 16));
            break;
        case RegName::a0e:
        case RegName::a1e:
        case RegName::b0e:
        case RegName::b1e:
            UNREACHABLE();

        case RegName::r0:
            regs.r[0] = value;
            break;
        case RegName::r1:
            regs.r[1] = value;
            break;
        case RegName::r2:
            regs.r[2] = value;
            break;
        case RegName::r3:
            regs.r[3] = value;
            break;
        case RegName::r4:
            regs.r[4] = value;
            break;
        case RegName::r5:
            regs.r[5] = value;
            break;
        case RegName::r6:
            regs.r[6] = value;
            break;
        case RegName::r7:
            regs.r[7] = value;
            break;

        case RegName::y0:
            regs.y[0] = value;
            break;
        case RegName::p: // p0h
            regs.pe[0] = value > 0x7FFF;
            regs.p[0] = (regs.p[0] & 0xFFFF) | (value << 16);
            break;

        case RegName::pc:
            UNREACHABLE();
        case RegName::sp:
            regs.sp = value;
            break;
        case RegName::sv:
            regs.sv = value;
            break;
        case RegName::lc:
            regs.Lc() = value;
            break;

        case RegName::ar0:
            regs.Set<ar0>(value);
            break;
        case RegName::ar1:
            regs.Set<ar1>(value);
            break;

        case RegName::arp0:
            regs.Set<arp0>(value);
            break;
        case RegName::arp1:
            regs.Set<arp1>(value);
            break;
        case RegName::arp2:
            regs.Set<arp2>(value);
            break;
        case RegName::arp3:
            regs.Set<arp3>(value);
            break;

        case RegName::ext0:
            regs.ext[0] = value;
            break;
        case RegName::ext1:
            regs.ext[1] = value;
            break;
        case RegName::ext2:
            regs.ext[2] = value;
            break;
        case RegName::ext3:
            regs.ext[3] = value;
            break;

        case RegName::stt0:
            regs.Set<stt0>(value);
            break;
        case RegName::stt1:
            regs.Set<stt1>(value);
            break;
        case RegName::stt2:
            regs.Set<stt2>(value);
            break;

        case RegName::st0:
            regs.Set<st0>(value);
            break;
        case RegName::st1:
            regs.Set<st1>(value);
            break;
        case RegName::st2:
            regs.Set<st2>(value);
            break;

        case RegName::cfgi:
            regs.Set<cfgi>(value);
            break;
        case RegName::cfgj:
            regs.Set<cfgj>(value);
            break;

        case RegName::mod0:
            regs.Set<mod0>(value);
            break;
        case RegName::mod1:
            regs.Set<mod1>(value);
            break;
        case RegName::mod2:
            regs.Set<mod2>(value);
            break;
        case RegName::mod3:
            regs.Set<mod3>(value);
            break;
        default:
            UNREACHABLE();
        }
    }

    template <typename ArRnX>
    u16 GetArRnUnit(ArRnX arrn) const {
        static_assert(std::is_same_v<ArRnX, ArRn1> || std::is_same_v<ArRnX, ArRn2>);
        return regs.arrn[arrn.Index()];
    }

    template <typename ArpRnX>
    std::tuple<u16, u16> GetArpRnUnit(ArpRnX arprn) const {
        static_assert(std::is_same_v<ArpRnX, ArpRn1> || std::is_same_v<ArpRnX, ArpRn2>);
        return std::make_tuple(regs.arprni[arprn.Index()], regs.arprnj[arprn.Index()] + 4);
    }

    static StepValue ConvertArStep(u16 arvalue) {
        switch (arvalue) {
        case 0:
            return StepValue::Zero;
        case 1:
            return StepValue::Increase;
        case 2:
            return StepValue::Decrease;
        case 3:
            return StepValue::PlusStep;
        case 4:
            return StepValue::Increase2Mode1;
        case 5:
            return StepValue::Decrease2Mode1;
        case 6:
            return StepValue::Increase2Mode2;
        case 7:
            return StepValue::Decrease2Mode2;
        default:
            UNREACHABLE();
        }
    }

    template <typename ArStepX>
    StepValue GetArStep(ArStepX arstep) const {
        static_assert(std::is_same_v<ArStepX, ArStep1> || std::is_same_v<ArStepX, ArStep1Alt> ||
                      std::is_same_v<ArStepX, ArStep2>);
        return ConvertArStep(regs.arstep[arstep.Index()]);
    }

    template <typename ArpStepX>
    std::tuple<StepValue, StepValue> GetArpStep(ArpStepX arpstepi, ArpStepX arpstepj) const {
        static_assert(std::is_same_v<ArpStepX, ArpStep1> || std::is_same_v<ArpStepX, ArpStep2>);
        return std::make_tuple(ConvertArStep(regs.arpstepi[arpstepi.Index()]),
                               ConvertArStep(regs.arpstepj[arpstepj.Index()]));
    }

    enum class OffsetValue : u16 {
        Zero = 0,
        PlusOne = 1,
        MinusOne = 2,
        MinusOneDmod = 3,
    };

    template <typename ArStepX>
    OffsetValue GetArOffset(ArStepX arstep) const {
        static_assert(std::is_same_v<ArStepX, ArStep1> || std::is_same_v<ArStepX, ArStep2>);
        return (OffsetValue)regs.aroffset[arstep.Index()];
    }

    template <typename ArpStepX>
    std::tuple<OffsetValue, OffsetValue> GetArpOffset(ArpStepX arpstepi, ArpStepX arpstepj) const {
        static_assert(std::is_same_v<ArpStepX, ArpStep1> || std::is_same_v<ArpStepX, ArpStep2>);
        return std::make_tuple((OffsetValue)regs.arpoffseti[arpstepi.Index()],
                               (OffsetValue)regs.arpoffsetj[arpstepj.Index()]);
    }

    u16 RnAddress(unsigned unit, unsigned value) {
        u16 ret = value;
        if (regs.br[unit] && !regs.m[unit]) {
            ret = BitReverse(ret);
        }
        return ret;
    }

    u16 RnAddressAndModify(unsigned unit, StepValue step, bool dmod = false) {
        return RnAddress(unit, RnAndModify(unit, step, dmod));
    }

    u16 OffsetAddress(unsigned unit, u16 address, OffsetValue offset, bool dmod = false) {
        if (offset == OffsetValue::Zero)
            return address;
        if (offset == OffsetValue::MinusOneDmod) {
            return address - 1;
        }
        bool emod = regs.m[unit] & !regs.br[unit] & !dmod;
        u16 mod = unit < 4 ? regs.modi : regs.modj;
        u16 mask = 1; // mod = 0 still have one bit mask
        for (unsigned i = 0; i < 9; ++i) {
            mask |= mod >> i;
        }
        if (offset == OffsetValue::PlusOne) {
            if (!emod)
                return address + 1;
            if ((address & mask) == mod)
                return address & ~mask;
            return address + 1;
        } else { // OffsetValue::MinusOne
            if (!emod)
                return address - 1;
            throw UnimplementedException();
            // TODO: sometimes this would return two addresses,
            // neither of which is the original Rn value.
            // This only happens for memory writing, but not for memory reading.
            // Might be some undefined behaviour.
            if ((address & mask) == 0)
                return address | mod;
            return address - 1;
        }
    }

    u16 StepAddress(unsigned unit, u16 address, StepValue step, bool dmod = false) {
        u16 s;
        bool legacy = regs.cmd;
        bool step2_mode1 = false;
        bool step2_mode2 = false;
        switch (step) {
        case StepValue::Zero:
            s = 0;
            break;
        case StepValue::Increase:
            s = 1;
            break;
        case StepValue::Decrease:
            s = 0xFFFF;
            break;
        // TODO: Increase/Decrease2Mode1/2 sometimes have wrong result if Offset=+/-1.
        // This however never happens with modr instruction.
        case StepValue::Increase2Mode1:
            s = 2;
            step2_mode1 = !legacy;
            break;
        case StepValue::Decrease2Mode1:
            s = 0xFFFE;
            step2_mode1 = !legacy;
            break;
        case StepValue::Increase2Mode2:
            s = 2;
            step2_mode2 = !legacy;
            break;
        case StepValue::Decrease2Mode2:
            s = 0xFFFE;
            step2_mode2 = !legacy;
            break;
        case StepValue::PlusStep: {
            if (regs.br[unit] && !regs.m[unit]) {
                s = unit < 4 ? regs.stepi0 : regs.stepj0;
            } else {
                s = unit < 4 ? regs.stepi : regs.stepj;
                s = SignExtend<7>(s);
            }
            if (regs.stp16 == 1 && !legacy) {
                s = unit < 4 ? regs.stepi0 : regs.stepj0;
                if (regs.m[unit]) {
                    s = SignExtend<9>(s);
                }
            }
            break;
        }
        default:
            UNREACHABLE();
        }

        if (s == 0)
            return address;

        if (!dmod && !regs.br[unit] && regs.m[unit]) {
            u16 mod = unit < 4 ? regs.modi : regs.modj;

            if (mod == 0) {
                return address;
            }

            if (mod == 1 && step2_mode2) {
                return address;
            }

            unsigned iteration = 1;
            if (step2_mode1) {
                iteration = 2;
                s = SignExtend<15, u16>(s >> 1);
            }

            for (unsigned i = 0; i < iteration; ++i) {
                if (legacy || step2_mode2) {
                    bool negative = false;
                    u16 m = mod;
                    if (s >> 15) {
                        negative = true;
                        m |= ~s;
                    } else {
                        m |= s;
                    }

                    u16 mask = (1 << std20::log2p1(m)) - 1;
                    u16 next;
                    if (!negative) {
                        if ((address & mask) == mod && (!step2_mode2 || mod != mask)) {
                            next = 0;
                        } else {
                            next = (address + s) & mask;
                        }
                    } else {
                        if ((address & mask) == 0 && (!step2_mode2 || mod != mask)) {
                            next = mod;
                        } else {
                            next = (address + s) & mask;
                        }
                    }
                    address &= ~mask;
                    address |= next;
                } else {
                    u16 mask = (1 << std20::log2p1(mod)) - 1;
                    u16 next;
                    if (s < 0x8000) {
                        next = (address + s) & mask;
                        if (next == ((mod + 1) & mask)) {
                            next = 0;
                        }
                    } else {
                        next = address & mask;
                        if (next == 0) {
                            next = mod + 1;
                        }
                        next += s;
                        next &= mask;
                    }
                    address &= ~mask;
                    address |= next;
                }
            }
        } else {
            address += s;
        }
        return address;
    }

    u16 RnAndModify(unsigned unit, StepValue step, bool dmod = false) {
        u16 ret = regs.r[unit];
        if ((unit == 3 && regs.epi) || (unit == 7 && regs.epj)) {
            if (step != StepValue::Increase2Mode1 && step != StepValue::Decrease2Mode1 &&
                step != StepValue::Increase2Mode2 && step != StepValue::Decrease2Mode2) {
                regs.r[unit] = 0;
                return ret;
            }
        }
        regs.r[unit] = StepAddress(unit, regs.r[unit], step, dmod);
        return ret;
    }

    u32 ProductToBus32_NoShift(Px reg) const {
        return regs.p[reg.Index()];
    }

    u64 ProductToBus40(Px reg) const {
        u16 unit = reg.Index();
        u64 value = regs.p[unit] | ((u64)regs.pe[unit] << 32);
        switch (regs.ps[unit]) {
        case 0:
            value = SignExtend<33>(value);
            break;
        case 1:
            value >>= 1;
            value = SignExtend<32>(value);
            break;
        case 2:
            value <<= 1;
            value = SignExtend<34>(value);
            break;
        case 3:
            value <<= 2;
            value = SignExtend<35>(value);
            break;
        }
        return value;
    }

    void ProductFromBus32(Px reg, u32 value) {
        u16 unit = reg.Index();
        regs.p[unit] = value;
        regs.pe[unit] = value >> 31;
    }

    static RegName CounterAcc(RegName in) {
        static std::unordered_map<RegName, RegName> map{
                                                        {RegName::a0, RegName::a1},   {RegName::a1, RegName::a0},
                                                        {RegName::b0, RegName::b1},   {RegName::b1, RegName::b0},
                                                        {RegName::a0l, RegName::a1l}, {RegName::a1l, RegName::a0l},
                                                        {RegName::b0l, RegName::b1l}, {RegName::b1l, RegName::b0l},
                                                        {RegName::a0h, RegName::a1h}, {RegName::a1h, RegName::a0h},
                                                        {RegName::b0h, RegName::b1h}, {RegName::b1h, RegName::b0h},
                                                        {RegName::a0e, RegName::a1e}, {RegName::a1e, RegName::a0e},
                                                        {RegName::b0e, RegName::b1e}, {RegName::b1e, RegName::b0e},
                                                        };
        return map.at(in);
    }

    const std::vector<Matcher<TranslatorVisitor>> decoders = GetDecoderTable<TranslatorVisitor>();
};

} // namespace Teakra
