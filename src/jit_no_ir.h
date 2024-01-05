#pragma once
#include <utility>
#include <atomic>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <xbyak/xbyak.h>
#include "bit.h"
#include <bit>
#include "core_timing.h"
#include "decoder.h"
#include "memory_interface.h"
#include "operand.h"
#include "jit_regs.h"
#include "xbyak_abi.h"

using namespace Xbyak::util;
using Xbyak::Label;
using Xbyak::Reg16;
using Xbyak::Reg32;
using Xbyak::Reg64;

namespace Teakra {

class UnimplementedException : public std::runtime_error {
public:
    UnimplementedException() : std::runtime_error("unimplemented") {}
};

// The following is used to alias some commonly used registers.

/// XpertTeak GPRs
/// r6 is seldom used, so we allocate its host reg to y0
constexpr Reg64 R[] = {r8, r9, r10, r11, r12, r13, Reg64{}, r15};
/// XpertTeak accumulators
constexpr Reg64 A[] = {rsi, rdi};
constexpr Reg64 B[] = {rbp, rsp};
/// Y0 is used by firmware way more than any other factor registers
constexpr Reg64 Y0 = r14;
/// Holds the register structure pointer
constexpr Reg64 REGS = rdx;

struct alignas(16) StackLayout {
    s64 cycles_remaining;
    s64 cycles_to_run;
};

class EmitX64 : public Xbyak::CodeGenerator {
public:
    EmitX64(JitRegisters& regs, MemoryInterface& mem)
        : regs(regs), mem(mem) {}

    using RunCodeFuncType = void (*)();
    RunCodeFuncType run_code{};

    std::unordered_map<u32, void(*)()> code_blocks;

    void EmitDispatcher() {
        align();
        run_code = getCurr<RunCodeFuncType>();

        // This serves two purposes:
        // 1. It saves all the registers we as a callee need to save.
        // 2. It aligns the stack so that the code the JIT emits can assume
        //    that the stack is appropriately aligned for CALLs.
        ABI_PushRegistersAndAdjustStack(*this, ABI_ALL_CALLEE_SAVED, sizeof(StackLayout));


    }

    void PushPC() {
        u16 l = (u16)(regs.pc & 0xFFFF);
        u16 h = (u16)(regs.pc >> 16);
        if (regs.cpc == 1) {
            mem.DataWrite(--regs.sp, h);
            mem.DataWrite(--regs.sp, l);
        } else {
            mem.DataWrite(--regs.sp, l);
            mem.DataWrite(--regs.sp, h);
        }
    }

    void PopPC() {
        u16 h, l;
        if (regs.cpc == 1) {
            l = mem.DataRead(regs.sp++);
            h = mem.DataRead(regs.sp++);
        } else {
            h = mem.DataRead(regs.sp++);
            l = mem.DataRead(regs.sp++);
        }
        SetPC(l | ((u32)h << 16));
    }

    void SetPC(u32 new_pc) {
        ASSERT(new_pc < 0x40000);
        regs.pc = new_pc;
    }

    void undefined(u16 opcode) {
        UNREACHABLE();
    }

    void Run(u64 cycles) {
        u16 opcode = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
        auto& decoder = decoders[opcode];
        u16 expand_value = 0;
        if (decoder.NeedExpansion()) {
            expand_value = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
        }

        if (regs.rep) {
            if (regs.repc == 0) {
                regs.rep = false;
            } else {
                --regs.repc;
                --regs.pc;
            }
        }

        const u32 index = regs.bcn - 1;
        if (regs.lp && regs.bkrep_stack[index].end == regs.pc - 1) {
            if (regs.bkrep_stack[index].lc == 0) {
                --regs.bcn;
                regs.lp = regs.bcn != 0;
            } else {
                --regs.bkrep_stack[index].lc;
                regs.pc = regs.bkrep_stack[index].start;
            }
        }

        decoder.call(*this, opcode, expand_value);

        // I am not sure if a single-instruction loop is interruptable and how it is handled,
        // so just disable interrupt for it for now.
        if (regs.ie && !regs.rep) {
            bool interrupt_handled = false;
            for (u32 i = 0; i < regs.im.size(); ++i) {
                if (regs.im[i] && regs.ip[i]) {
                    regs.ip[i] = 0;
                    regs.ie = 0;
                    PushPC();
                    regs.pc = 0x0006 + i * 8;
                    interrupt_handled = true;
                    if (regs.ic[i]) {
                        ContextStore();
                    }
                    break;
                }
            }
            if (!interrupt_handled && regs.imv && regs.ipv) {
                regs.ipv = 0;
                regs.ie = 0;
                PushPC();
                regs.pc = vinterrupt_address;
                idle = false;
                if (vinterrupt_context_switch) {
                    ContextStore();
                }
            }
        }
    }

    void SignalInterrupt(u32 i) {
        interrupt_pending[i] = true;
    }
    void SignalVectoredInterrupt(u32 address, bool context_switch) {
        vinterrupt_address = address;
        vinterrupt_pending = true;
        vinterrupt_context_switch = context_switch;
    }

    using instruction_return_type = void;

    std::array<std::atomic<bool>, 3> interrupt_pending{{false, false, false}};
    std::atomic<bool> vinterrupt_pending{false};
    std::atomic<bool> vinterrupt_context_switch;
    std::atomic<u32> vinterrupt_address;

    void nop() {
        // literally nothing
    }

private:
    JitRegisters& regs;
    MemoryInterface& mem;

    Reg64 GetAcc(RegName name) const {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            return A[0];
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            return A[1];
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            return B[0];
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            return B[1];
        default:
            UNREACHABLE();
        }
    }

    template <typename T>
    void SignExtend(T value, u32 bit_count) {
        if (bit_count == 16) {
            movsx(value, value.cvt16());
            return;
        }
        if (bit_count == 32) {
            movsx(value, value.cvt32());
            return;
        }
        const u64 mask = (1ULL << bit_count) - 1;
        const u64 bit = 1ULL << (bit_count - 1);
        test(value, bit);
        Xbyak::Label not_sign, end_label;
        jz(not_sign);
        or_(value, ~mask);
        jmp(end_label);
        L(not_sign);
        and(value, mask);
        L(end_label);
    }

    void SaturateAccNoFlag(Reg64 value) const {
        mov(rbx, value);
        SignExtend(rbx, 32);
        cmp(rbx, value);

        if (value != SignExtend<32>(value)) {
            if ((value >> 39) != 0)
                return 0xFFFF'FFFF'8000'0000;
            else
                return 0x0000'0000'7FFF'FFFF;
        }
        return value;
    }

    u64 SaturateAcc(u64 value) {
        if (value != SignExtend<32>(value)) {
            regs.flm = 1;
            if ((value >> 39) != 0)
                return 0xFFFF'FFFF'8000'0000;
            else
                return 0x0000'0000'7FFF'FFFF;
        }
        // note: flm doesn't change value otherwise
        return value;
    }

    Reg64 GetAndSatAcc(RegName name) {
        Reg64 value = GetAcc(name);
        if (!regs.sat) {
            return SaturateAcc(value);
        }
        return value;
    }

    u64 GetAndSatAccNoFlag(RegName name) const {
        u64 value = GetAcc(name);
        if (!regs.sat) {
            return SaturateAccNoFlag(value);
        }
        return value;
    }

    void ConditionPass(Cond cond, auto&& func) const {
        Xbyak::Label end_cond;
        switch (cond.GetName()) {
        case CondValue::True:
            func();
            return;
        case CondValue::Eq:
            mov(ax, word[REGS + offsetof(RegisterState, fz)]);
            cmp(ax, 1);
            jne(end_cond);
            break;
        case CondValue::Neq:
            mov(ax, word[REGS + offsetof(RegisterState, fz)]);
            test(ax, ax);
            jne(end_cond);
            break;
        case CondValue::Gt:
            // Load fz and fm together as a u32
            static_assert(offsetof(RegisterState, fz) + sizeof(u16) == offsetof(RegisterState, fm));
            mov(eax, dword[REGS + offsetof(RegisterState, fz)]);
            test(eax, eax);
            jne(end_cond);
            break;
        case CondValue::Ge:
            mov(ax, word[REGS + offsetof(RegisterState, fm)]);
            test(ax, ax);
            jne(end_cond);
            break;
        case CondValue::Lt:
            mov(ax, word[REGS + offsetof(RegisterState, fm)]);
            cmp(ax, 1);
            jne(end_cond);
            break;
        case CondValue::Le:
            // Load fz and fm together as a u32
            mov(eax, dword[REGS + offsetof(RegisterState, fz)]);
            test(eax, eax);
            je(end_cond);
            break;
        case CondValue::Nn:
            mov(ax, word[REGS + offsetof(RegisterState, fn)]);
            test(ax, ax);
            jne(end_cond);
            break;
        case CondValue::C:
            mov(ax, word[REGS + offsetof(RegisterState, fc0)]);
            cmp(ax, 1);
            jne(end_cond);
            break;
        case CondValue::V:
            mov(ax, word[REGS + offsetof(RegisterState, fv)]);
            cmp(ax, 1);
            jne(end_cond);
            break;
        case CondValue::E:
            mov(ax, word[REGS + offsetof(RegisterState, fe)]);
            cmp(ax, 1);
            jne(end_cond);
            break;
        case CondValue::L:
            // Load flm and fvl together as a u32
            static_assert(offsetof(RegisterState, flm) + sizeof(u16) == offsetof(RegisterState, fvl));
            mov(eax, dword[REGS + offsetof(RegisterState, flm)]);
            test(eax, eax);
            je(end_cond);
            break;
        case CondValue::Nr:
            mov(ax, word[REGS + offsetof(RegisterState, fr)]);
            test(ax, ax);
            jne(end_cond);
            break;
        case CondValue::Niu0:
            mov(ax, word[REGS + offsetof(RegisterState, iu)]);
            test(ax, ax);
            jne(end_cond);
            break;
        case CondValue::Iu0:
            mov(ax, word[REGS + offsetof(RegisterState, iu)]);
            cmp(ax, 1);
            jne(end_cond);
            break;
        case CondValue::Iu1:
            mov(ax, word[REGS + offsetof(RegisterState, iu) + sizeof(u16)]);
            cmp(ax, 1);
            jne(end_cond);
            break;
        default:
            UNREACHABLE();
        }
        func();
        L(end_cond);
    }

    Reg16 RegToBus16(RegName reg, bool enable_sat_for_mov = false) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            // get aXl, but unlike using RegName::aXl, this does never saturate.
            // This only happen to insturctions using "Register" operand,
            // and doesn't apply to all instructions. Need test and special check.
            mov(ax, GetAcc(reg).cvt16());
            break;
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            if (enable_sat_for_mov) {
                mov(ax, GetAndSatAcc(reg).cvt16());
            } else {
                mov(ax, GetAcc(reg).cvt16());
            }
            break;
        case RegName::a0h:
        case RegName::a1h:
        case RegName::b0h:
        case RegName::b1h:
            if (enable_sat_for_mov) {
                mov(eax, GetAndSatAcc(reg).cvt32());
            } else {
                mov(eax, GetAcc(reg).cvt32());
            }
            shr(eax, 16);
            break;
        case RegName::a0e:
        case RegName::a1e:
        case RegName::b0e:
        case RegName::b1e:
            UNREACHABLE();

        case RegName::r0:
            mov(ax, R[0]);
            break;
        case RegName::r1:
            mov(ax, R[1]);
            break;
        case RegName::r2:
            mov(ax, R[2]);
            break;
        case RegName::r3:
            mov(ax, R[3]);
            break;
        case RegName::r4:
            mov(ax, R[4]);
            break;
        case RegName::r5:
            mov(ax, R[5]);
            break;
        case RegName::r6:
            // R6 is very rarely used, it would be a waste to assign a register to it
            mov(ax, word[REGS + offsetof(RegisterState, r[6])]);
            break;
        case RegName::r7:
            mov(ax, R[7]);
            break;

        case RegName::y0:
            mov(ax, Y0);
            break;
        case RegName::p:
            // This only happen to insturctions using "Register" operand,
            // and doesn't apply to all instructions. Need test and special check.
            return (ProductToBus40(Px{0}) >> 16) & 0xFFFF;

        case RegName::pc:
            UNREACHABLE();
        case RegName::sp:
            mov(ax, word[REGS + offsetof(RegisterState, sp)]);
            break;
        case RegName::sv:
            mov(ax, word[REGS + offsetof(RegisterState, sv)]);
            break;
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
            return regs.Get<Teakra::st0>();
        case RegName::st1:
            return regs.Get<Teakra::st1>();
        case RegName::st2:
            return regs.Get<Teakra::st2>();

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
        return ax;
    }

    void SetAccFlag(u64 value) {
        regs.fz = value == 0;
        regs.fm = (value >> 39) != 0;
        regs.fe = value != SignExtend<32>(value);
        u64 bit31 = (value >> 31) & 1;
        u64 bit30 = (value >> 30) & 1;
        regs.fn = regs.fz || (!regs.fe && (bit31 ^ bit30) != 0);
    }

    void SetAcc(RegName name, Reg64 value) {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            mov(A[0], value);
            break;
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            mov(A[1], value);
            break;
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            mov(B[0], value);
            break;
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            mov(B[1], value);
            break;
        default:
            UNREACHABLE();
        }
    }

    void SatAndSetAccAndFlag(RegName name, u64 value) {
        SetAccFlag(value);
        if (!regs.sata) {
            value = SaturateAcc(value);
        }
        SetAcc(name, value);
    }

    void SetAccAndFlag(RegName name, u64 value) {
        SetAccFlag(value);
        SetAcc(name, value);
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
            regs.Set<Teakra::st0>(value);
            break;
        case RegName::st1:
            regs.Set<Teakra::st1>(value);
            break;
        case RegName::st2:
            regs.Set<Teakra::st2>(value);
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
    void GetArRnUnit(Reg32& out, ArRnX arrn) const {
        static_assert(std::is_same_v<ArRnX, ArRn1> || std::is_same_v<ArRnX, ArRn2>);
        mov(out, dword[REGS + offsetof(RegisterState, arrn) + arrn * sizeof(arrn[0])]);
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

    u16 RnAddress(u32 unit, u32 value) {
        u16 ret = value;
        if (regs.br[unit] && !regs.m[unit]) {
            ret = BitReverse(ret);
        }
        return ret;
    }

    u16 RnAddressAndModify(unsigned unit, StepValue step, bool dmod = false) {
        return RnAddress(unit, RnAndModify(unit, step, dmod));
    }

    void DoOffsetAddress(u32 unit, Reg16 address, OffsetValue offset, bool emod) {
        ASSERT(address == ax);
        if (offset == OffsetValue::Zero) {
            ret();
            return;
        }
        if (offset == OffsetValue::MinusOneDmod) {
            sub(address, 1);
            ret();
            return;
        }
        if (offset == OffsetValue::PlusOne && !emod) {
            add(address, 1);
            ret();
            return;
        }
        if (offset == OffsetValue::MinusOne && !emod) {
            sub(address, 1);
            ret();
            return;
        }
        const u32 offset = unit < 4 ? offsetof(RegisterState, modi) : offsetof(RegisterState, modj);
        const Reg16 mod = bx;
        mov(mod, word[REGS + offset]);
        const Reg16 mask = cx;
        mov(mask, 1); // mod = 0 still have one bit mask
        for (u32 i = 0; i < 9; ++i) {
            or_(mask, mod);
            shr(mod, 1);
        }
        if (offset == OffsetValue::PlusOne) {
            Xbyak::Label not_equal;
            mov(dx, address);
            and_(dx, mask);
            cmp(dx, mod);
            jne(not_equal);
            not_(mask);
            and_(address, mask);
            ret();
            L(not_equal);
            add(address, 1);
            ret();
            return address + 1;
        } else { // OffsetValue::MinusOne
            throw UnimplementedException();
            // TODO: sometimes this would return two addresses,
            // neither of which is the original Rn value.
            // This only happens for memory writing, but not for memory reading.
            // Might be some undefined behaviour.
            ASSERT(false);
        }
    }

    void DoMask(Reg16 mask, Reg16 m) {
        // We know m != 0 so don't have to check for it.
        bsr(m, m);
        add(m, 1) // m = std::bit_width(m);
        mov(mask, 1);
        sal(mask, m);
        sub_(mask, 1); // u16 mask = (1 << m) - 1;
    }

    void DoPulseStepAddress(u32 unit, Reg32 address, StepValue step, bool legacy,
                            bool br, bool m, bool stp16, bool dmod = false) {
        const Reg16 s = bx;

        if (br && !m) {
            const u32 offset = unit < 4 ? offsetof(RegisterState, stepi0) : offsetof(RegisterState, stepj0);
            mov(s, word[REGS + offset]);
        } else {
            const u32 offset = unit < 4 ? offsetof(RegisterState, stepi) : offsetof(RegisterState, stepj);
            mov(s, word[REGS + offset]);
            SignExtend(s, 7);
        }
        if (stp16 && !legacy) {
            const u32 offset = unit < 4 ? offsetof(RegisterState, stepi0) : offsetof(RegisterState, stepj0);
            mov(s, word[REGS + offset]);
            if (m) {
                SignExtend(s, 9);
            }
        }

        Xbyak::Label return_label;
        test(s, s);
        je(return_label);

        if (!dmod && !br && m) {
            const u32 offset = unit < 4 ? offsetof(RegisterState, modi) : offsetof(RegisterState, modj);
            const Reg16 mod = cx;
            mov(mod, word[REGS + offset]);
            push(rdx);          // Following code will use rdx which contains REGS, preserve it!
            test(mod, mod);
            je(return_label); // if (mod == 0) return address;

            if (legacy) {
                const Reg16 m = mod;
                mov(dx, mod);        // tmp = mod;
                shl(mod.cvt64(), 32); // mod <<= 32;
                mov(mod, dx);        // mod[0,16] = tmp;
                Xbyak::Label no_sign, end_label;
                // Lower 32-bit of mov contained m which we no longer need after the mask is generated, use it as scratch.
                const Reg32 scratch = mod.cvt32();
                test(s, 1 << 15);
                jz(no_sign); // if (s >> 15)
                {
                    not_(s);
                    or_(m, s); // m |= ~s;
                    not_(s);
                    const Reg16 mask = dx;
                    DoMask(mask, m); // u16 mask = (1 << std::bit_width(m)) - 1;
                    // next = (address + s) & mask;
                    const Reg16 next = s;
                    add(next, address);
                    and_(next, mask);
                    // scratch = mask & address;
                    mov(scratch, address);
                    and_(scratch, mask);
                    andn(address, mask, address); // BMI instruction for address &= ~mask;
                    test(scratch, scratch); // Check if (cond == 0)
                    shr(mod.cvt64(), 32); // Restore mod from upper 32-bit. This does not affect ZF
                    cmovz(next, mod); // next = cond == 0 ? mod : next;
                    or_(address, next);  // address |= next;
                    jmp(end_label);
                }
                L(no_sign); // else
                {
                    or_(m, s); // m |= s;
                    const Reg16 mask = dx;
                    DoMask(mask, m); // u16 mask = (1 << std::bit_width(m)) - 1;
                    // next = (address + s) & mask;
                    const Reg16 next = s;
                    add(next, address);
                    and_(next, mask);
                    // scratch = mask & address;
                    mov(scratch, address);
                    and_(scratch, mask);
                    andn(address, mask, address); // BMI instruction for address &= ~mask;
                    const Reg32 cond = mask;
                    mov(cond, scratch);   // mask is no longer needed, move the condition to it.
                    shr(mod.cvt64(), 32); // Restore mod from the upper 32-bits
                    cmp(cond, mod); // Check if (address & mask) == mod;
                    mov(mod, 0);    // mod = 0
                    cmove(next, mod);  // s = (Address & mask) == mod ? 0 : next;
                    or_(address, next);    // address |= next;
                }
                L(end_label);
            } else {
                const Reg16 mask = dx;
                DoMask(mask, mod); // u16 mask = (1 << std::bit_width(mod)) - 1;
                Xbyak::Label has_sign, end_label;
                test(s, s);
                js(has_sign); // if (s < 0x8000)
                {
                    // next = (address + s) & mask;
                    const Reg16 next = s;
                    add(next, address);
                    and_(next, mask);
                    // cond = (mod + 1) & mask;
                    const Reg16 cond = mod;
                    add_(cond, 1);
                    and_(cond, mask);
                    // next = (next == cond) ? 0 : next;
                    cmp(next, cond);
                    mov(cond, 0);
                    cmove(next, cond);
                    andn(address, mask, address); // BMI instruction for address &= ~mask;
                    or_(address, next);    // address |= next;
                    jmp(end_label);
                }
                L(has_sign); // else
                {

                }
                L(end_label);
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
            pop(rdx);
        } else {
            add(address, s);
        }
        L(return_label);
    }

    union StepAddressKey {
        BitField<0, 3, u16> unit;
        BitField<3, 3, u16> step;
        BitField<6, 1, u16> legacy;
        BitField<7, 1, u16> br;
        BitField<8, 1, u16> m;
        BitField<9, 1, u16> stp16;
        BitField<10, 1, u16> dmod;
    };

    void DoStepAddress(u32 unit, Reg32 address, StepValue step, bool legacy,
                      bool br, bool m, bool stp16, bool dmod = false) {
        u16 s;
        bool step2_mode1 = false;
        bool step2_mode2 = false;
        switch (step) {
        case StepValue::Zero:
            return;
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
            return DoPulseStepAddress(unit, address, step, legacy, br, m, stp16, dmod);
        }
        default:
            UNREACHABLE();
        }

        Xbyak::Label return_label;

        if (!dmod && !br && m) {
            const u32 offset = unit < 4 ? offsetof(RegisterState, modi) : offsetof(RegisterState, modj);
            const Reg16 mod = bx;
            mov(mod, word[REGS + offset]);
            cmp(mod, 0);
            je(return_label);

            if (step2_mode2) {
                cmp(mod, 1);
                je(return_label);
            }

            u32 iteration = 1;
            if (step2_mode1) {
                iteration = 2;
                s = SignExtend<15, u16>(s >> 1);
            }

            for (u32 i = 0; i < iteration; ++i) {
                if (legacy || step2_mode2) {
                    bool negative = false;
                    const Reg16 m = cx;
                    mov(m, mod);
                    if (s >> 15) {
                        negative = true;
                        or_(m, ~s);
                    } else {
                        or_(m, s);
                    }

                    const Reg16 mask = dx;
                    // We know m != 0 so don't have to check for it.
                    bsr(m, m);
                    add(m, 1) // m = std::bit_width(m);
                    mov(mask, 1);
                    sal(mask, m);
                    sub_(mask, 1); // u16 mask = (1 << m) - 1;

                    // m is no longer needed so we can use it as scratch
                    const Reg32 scratch = m.cvt32();
                    mov(scratch, address);
                    and_(scratch, mask); // scratch = address & mask

                    Xbyak::Label do_step, end_step;
                    if (!negative) {
                        if ((address & mask) == mod && (!step2_mode2 || mod != mask)) {
                        } else {
                            address |= (address + s) & mask;
                        }
                    } else {
                        if ((address & mask) == 0 && (!step2_mode2 || mod != mask)) {
                            address |= mod;
                        } else {
                            address |= (address + s) & mask;
                        }
                    }
                    L(end_step);
                    not_(mask);
                    and_(address, mask); // address &= ~mask;
                } else {
                    u16 mask = (1 << std::bit_width(mod)) - 1;
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
        L(return_label);
        ret();
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

    u16 RnAndModify(u32 unit, StepValue step, bool dmod = false) {
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

    Reg64 ProductToBus40(Px reg) const {
        // TODO: Load this with a single mov
        u16 unit = reg.Index();
        mov(ax, word[REGS + offsetof(RegisterState, pe) + unit * sizeof(u16)]);
        shl(rax, 32);
        mov(eax, dword[REGS + offsetof(RegisterState, p) + unit * sizeof(u32)]);
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

    const std::vector<Matcher<EmitX64>> decoders = GetDecoderTable<EmitX64>();
};

} // namespace Teakra
