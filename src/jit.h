#pragma once
#include <utility>
#include <atomic>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include "decoder.h"
#include "ir/ir_emitter.h"
#include "matcher.h"
#include "operand.h"
#include "jit_regs.h"

namespace Teakra {

class UnimplementedException : public std::runtime_error {
public:
    UnimplementedException() : std::runtime_error("unimplemented") {}
};

class TranslatorVisitor {
public:
    TranslatorVisitor(JitRegisters& regs, IR::Block& block_,
                      IR::LocationDescriptor descriptor) : ir(block_, descriptor), regs(regs) {}

    IR::IREmitter ir;

    using instruction_return_type = void;

    void trap() {
        throw UnimplementedException();
    }

private:
    JitRegisters& regs;

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
            UNREACHABLE();
        case RegName::pc:
            UNREACHABLE();
        case RegName::sp:
            return ir.GetRegister16(RegName::sp);
        case RegName::sv:
            return ir.GetRegister16(RegName::sv);
        case RegName::lc:
            UNREACHABLE();
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

    const std::vector<Matcher<TranslatorVisitor>> decoders = GetDecoderTable<TranslatorVisitor>();
};

} // namespace Teakra
