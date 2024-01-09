#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <vector>
#include "crash.h"
#include "bit_field.h"
#include "common_types.h"
#include <xbyak/xbyak.h>

namespace Teakra {

using namespace Xbyak::util;
using Xbyak::Reg64;
using Xbyak::Reg32;
using Xbyak::Reg16;

// The following is used to alias some commonly used registers.

/// Quads of XpertTeak GPRs packed in a single host register
constexpr Reg64 R0_1_2_3 = r8;
constexpr Reg64 R4_5_6_7 = r9;

/// XpertTeak accumulators
constexpr Reg64 A[] = {r10, r11};
constexpr Reg64 B[] = {r12, r13};
/// Holds multiplication factor registers x0/x1/y0/y1.
constexpr Reg64 FACTORS = r14;
/// Holds the register structure pointer
constexpr Reg64 REGS = r15;
/// Holds commonly used status flags
constexpr Reg32 FLAGS = edi;

// Most frequently accessed status registers, or registers with no cross refrences are stored directly by the JIT for speed
// These registers are also used in a static manner as it greatly reduces the amount of emitted assembly per block.

// This doesn't represent an actual register but the current status flags of the JIT
union Flags {
    BitField<0, 1, u16> fr;
    BitField<1, 1, u16> flm; // set on saturation
    BitField<2, 1, u16> fvl; // Rn zero flag
    BitField<3, 1, u16> fe; // extension flag
    BitField<4, 1, u16> fc0; // carry flag
    BitField<5, 1, u16> fv; // overflow flag
    BitField<6, 1, u16> fn; // normalized flag
    BitField<7, 1, u16> fm; // negative flag
    BitField<8, 1, u16> fz; // zero flag
    BitField<12, 1, u16> fc1; // another carry flag
};

union ArU {
    u16 raw;
    BitField<0, 3, u16> arstep1;
    BitField<3, 2, u16> aroffset1;
    BitField<5, 3, u16> arstep0;
    BitField<8, 2, u16> aroffset0;
    BitField<10, 3, u16> arrn1;
    BitField<13, 3, u16> arrn0;
};

union ArpU {
    u16 raw;
    BitField<0, 3, u16> arpstepi;
    BitField<3, 2, u16> arpoffseti;
    BitField<5, 3, u16> arpstepj;
    BitField<8, 2, u16> arpoffsetj;
    BitField<10, 2, u16> arprni;
    BitField<13, 2, u16> arprnj;
};

union Mod0 {
    Mod0() {
        mod0_unk_const.Assign(1);
        sata.Assign(1);
    }

    u16 raw;
    BitField<0, 1, u16> sat; // 1-bit, disable saturation when moving from acc
    BitField<1, 1, u16> sata; // 1-bit, disable saturation when moving to acc
    BitField<2, 3, u16> mod0_unk_const; // = 1, read only
    BitField<5, 2, u16> hwm; // 2-bit, half word mode, modify y on multiplication
    BitField<7, 1, u16> s; // 1-bit, shift mode. 0 - arithmetic, 1 - logic
    BitField<8, 1, u16> ou0; // user output pins (always zero in firmware?)
    BitField<9, 1, u16> ou1;
    BitField<10, 2, u16> ps0; // 2-bit, product shift mode
    BitField<13, 2, u16> ps1;
};

union Mod1 {
    Mod1() {
        cmd.Assign(1);
    }

    u16 raw;
    BitField<0, 8, u16> page; // 8-bit, higher part of MemImm8 address
    BitField<12, 1, u16> stp16; // 1 bit. If set, stepi0/j0 will be exchanged along with cfgi/j in banke, and use
                                // stepi0/j0 for steping
    BitField<13, 1, u16> cmd; // 1-bit, step/mod method. 0 - Teak; 1 - TeakLite
    BitField<14, 1, u16> epi; // 1-bit. If set, cause r3 = 0 when steping r3
    BitField<15, 1, u16> epj; // 1-bit. If set, cause r7 = 0 when steping r7
};

union Mod2 {
    u16 raw;
    BitField<0, 1, u16> m0; // 1-bit each, enable modulo arithmetic for Rn
    BitField<1, 1, u16> m1;
    BitField<2, 1, u16> m2;
    BitField<3, 1, u16> m3;
    BitField<4, 1, u16> m4;
    BitField<5, 1, u16> m5;
    BitField<6, 1, u16> m6;
    BitField<7, 1, u16> m7;
    BitField<8, 1, u16> br0; // 1-bit each, use bit-reversed value from Rn as address
    BitField<9, 1, u16> br1;
    BitField<10, 1, u16> br2;
    BitField<11, 1, u16> br3;
    BitField<12, 1, u16> br4;
    BitField<13, 1, u16> br5;
    BitField<14, 1, u16> br6;
    BitField<15, 1, u16> br7;
};

// Note: This registers owns none of its members
union Mod3 {
    u16 raw;
    BitField<0, 1, u16> nimc;
    BitField<1, 1, u16> ic0;
    BitField<2, 1, u16> ic1;
    BitField<3, 1, u16> ic2;
    BitField<4, 1, u16> ou2;
    BitField<5, 1, u16> ou3;
    BitField<6, 1, u16> ou4;
    BitField<7, 1, u16> ie;
    BitField<8, 1, u16> im0;
    BitField<9, 1, u16> im1;
    BitField<10, 1, u16> im2;
    BitField<11, 1, u16> imv;
    BitField<13, 1, u16> ccnta;
    BitField<14, 1, u16> cpc;
    BitField<15, 1, u16> crep;

    u64 IcQword() const {
        return nimc.Value()
        | (static_cast<u64>(ic0.Value()) << 16)
        | (static_cast<u64>(ic1.Value()) << 32)
        | (static_cast<u64>(ic2.Value()) << 48);
    }

    u32 OuDword() const {
        return ou2.Value() | (static_cast<u32>(ou3.Value()) << 16);
    }

    u64 ImQword() const {
        return im0.Value()
        | (static_cast<u64>(im1.Value()) << 16)
        | (static_cast<u64>(im2.Value()) << 32)
        | (static_cast<u64>(imv.Value()) << 48);
    }

    u64 CQword() const {
        return ccnta.Value()
        | (static_cast<u64>(cpc.Value()) << 16)
        | (static_cast<u64>(crep.Value()) << 32);
    }
};

union Stt0 {
    BitField<0, 1, u16> flm; // set on saturation
    BitField<1, 1, u16> fvl; // Rn zero flag
    BitField<2, 1, u16> fe; // extension flag
    BitField<3, 1, u16> fc0; // carry flag
    BitField<4, 1, u16> fv; // overflow flag
    BitField<5, 1, u16> fn; // normalized flag
    BitField<6, 1, u16> fm; // negative flag
    BitField<7, 1, u16> fz; // zero flag
    BitField<11, 1, u16> fc1; // another carry flag
    BitField<2, 6, u16> st0_flags;
};

union Stt1 {
    BitField<4, 1, u16> fr;
    BitField<10, 1, u16> iu0;
    BitField<11, 1, u16> iu1;
    BitField<14, 1, u16> pe0;
    BitField<15, 1, u16> pe1;
};

union St1 {
    BitField<0, 8, u16> page_alias;
    BitField<10, 2, u16> ps0_alias;
    BitField<12, 4, u16> a1e_alias;
};

union Cfg {
    Cfg(u16 value = 0) : raw{value} {}
    u16 raw;
    BitField<0, 7, u16> step;
    BitField<7, 9, u16> mod;
};

struct JitRegisters {
    void Reset() {
        *this = JitRegisters();
    }

    /** Program control unit **/

    u32 pc = 0;     // 18-bit, program counter
    u32 idle = 0;
    u16 pad0{};
    u16 prpage = 0; // 4-bit, program page

    u16 repc = 0;     // 16-bit rep loop counter
    u16 repcs = 0;    // repc shadow
    bool rep = false; // true when in rep loop

    u16 bcn = 0; // 3-bit, nest loop counter
    u16 lp = 0;  // 1-bit, set when in a loop

    struct BlockRepeatFrame {
        u32 start = 0;
        u32 end = 0;
        u16 lc = 0;
    };

    std::array<BlockRepeatFrame, 4> bkrep_stack;
    u16& Lc() {
        if (lp)
            return bkrep_stack[bcn - 1].lc;
        return bkrep_stack[0].lc;
    }

    /** Computation unit **/

    // 40-bit 2's comp accumulators.
    // Use 64-bit 2's comp here. The upper 24 bits are always sign extension
    std::array<u64, 2> a{};
    std::array<u64, 2> b{};

    u64 a1s = 0, b1s = 0; // shadows for a1 and b1
    u16 ccnta = 1;        // 1-bit. If clear, store/restore a1/b1 to shadows on context switch
    u16 cpc = 1;    // 1-bit, change word order when push/pop pc
    u16 crep = 1;     // 1-bit. If clear, store/restore repc to shadows on context switch
    u16 pad = 0;

    ///< Swap register list
    u16 pcmhi = 0; // 2-bit, higher part of program address for movp/movd
    Mod1 mod1{};
    Mod0 mod0{};
    Mod2 mod2{};
    std::array<u16, 3> im{}; // interrupt enable bit
    u16 imv = 0;
    ///< Swap register list end

    u16 sv = 0;   // 16-bit two's complement shift value

    Flags flags{};  // Not a register, but used to store host flags register.

    // Shadows
    Flags flagsb{};

    // Viterbi
    u16 vtr0 = 0;
    u16 vtr1 = 0;

    /** Multiplication unit **/

    std::array<u16, 2> x{};  // factor
    std::array<u16, 2> y{};  // factor
    std::array<u32, 2> p{};  // product
    std::array<u16, 2> pe{}; // 1-bit product extension
    u16 p0h_cbs = 0;         // 16-bit hidden state for codebook search (CBS) opcode

    /** Address unit **/

    std::array<u16, 8> r{}; // 16-bit general and address registers
    u16 mixp = 0;           // 16-bit, stores result of min/max instructions
    u16 sp = 0;             // 16-bit stack pointer

    // shadows for bank exchange;
    u16 r0b = 0, r1b = 0, r4b = 0, r7b = 0;

    /** Address step/mod unit **/

    // step/modulo
    Cfg cfgi{}, cfgj{};
    u16 stepi0 = 0;
    u16 stepj0 = 0; // 16-bit step

    // shadows for bank exchange
    Cfg cfgib{}, cfgjb{};
    u16 stepi0b = 0, stepj0b = 0;

    JitRegisters() {
        arp[0].arpoffseti.Assign(0);
        arp[1].arpoffseti.Assign(1);
        arp[2].arpoffseti.Assign(2);
        arp[3].arpoffseti.Assign(0);
    }

    /** Indirect address unit **/
    std::array<ArpU, 4> arp{};
    std::array<ArU, 2> ar{};
    u32 pad3; // SSE padding

    // Shadows for bank exchange
    std::array<ArpU, 4> arpb{};
    std::array<ArU, 2> arb{};
    u32 pad4; // SSE padding

    /** Interrupt unit **/

    // interrupt pending bit
    std::array<u16, 3> ip{};
    u16 ipv = 0;

    // interrupt context switching bit
    u16 nimc = 0;
    std::array<u16, 3> ic{};

    // interrupt enable master bit
    u16 ie = 0;

    /** Extension unit **/

    std::array<u16, 5> ou{}; // user output pins
    std::array<u16, 2> iu{}; // user input pins
    std::array<u16, 4> ext{};

    // shadow swap registers
    u16 pcmhib = 0; // 2-bit, higher part of program address for movp/movd
    Mod1 mod1b{};
    Mod0 mod0b{};
    Mod2 mod2b{};
    std::array<u16, 3> imb{}; // interrupt enable bit
    u16 imvb = 0;

    void ShadowStore(Xbyak::CodeGenerator& c, Xbyak::Reg32 scratch) {
        c.mov(scratch, dword[REGS + offsetof(JitRegisters, flags)]);
        c.mov(dword[REGS + offsetof(JitRegisters, flagsb)], scratch);
    }

    void ShadowRestore(Xbyak::CodeGenerator& c, Xbyak::Reg32 scratch) {
        c.mov(scratch, dword[REGS + offsetof(JitRegisters, flagsb)]);
        c.mov(dword[REGS + offsetof(JitRegisters, flags)], scratch);
    }

    void SwapAllArArp(Xbyak::CodeGenerator& c) {
        // std::swap(ar, arb);
        // std::swap(arp, arpb);
        c.movdqu(xmm0, xword[REGS + offsetof(JitRegisters, arp)]);
        c.movdqu(xmm1, xword[REGS + offsetof(JitRegisters, arpb)]);
        c.movdqu(xword[REGS + offsetof(JitRegisters, arp)], xmm1);
        c.movdqu(xword[REGS + offsetof(JitRegisters, arpb)], xmm0);
    }

    void ShadowSwap(Xbyak::CodeGenerator& c) {
        //shadow_swap_registers.Swap(this);
        static_assert(offsetof(JitRegisters, imv) + sizeof(imv) - offsetof(JitRegisters, pcmhi) == sizeof(u64) * 2);
        c.movdqu(xmm0, xword[REGS + offsetof(JitRegisters, pcmhi)]);
        c.movdqu(xmm1, xword[REGS + offsetof(JitRegisters, pcmhib)]);
        c.movdqu(xword[REGS + offsetof(JitRegisters, pcmhi)], xmm1);
        c.movdqu(xword[REGS + offsetof(JitRegisters, pcmhib)], xmm0);
        SwapAllArArp(c);
    }

    void SwapAr(Xbyak::CodeGenerator& c, u16 index) {
        // std::swap(ar[index], arb[index]);
        c.mov(ax, word[REGS + offsetof(JitRegisters, ar) + index * sizeof(ArU)]);
        c.mov(bx, word[REGS + offsetof(JitRegisters, arb) + index * sizeof(ArU)]);
        c.mov(word[REGS + offsetof(JitRegisters, ar) + index * sizeof(ArU)], bx);
        c.mov(word[REGS + offsetof(JitRegisters, arb) + index * sizeof(ArU)], ax);
    }

    void SwapArp(Xbyak::CodeGenerator& c, u16 index) {
        // std::swap(arp[index], arpb[index]);
        c.mov(ax, word[REGS + offsetof(JitRegisters, ar) + index * sizeof(ArpU)]);
        c.mov(bx, word[REGS + offsetof(JitRegisters, arb) + index * sizeof(ArpU)]);
        c.mov(word[REGS + offsetof(JitRegisters, ar) + index * sizeof(ArpU)], bx);
        c.mov(word[REGS + offsetof(JitRegisters, arb) + index * sizeof(ArpU)], ax);
    }

    void GetCfgi(Xbyak::CodeGenerator& c, Xbyak::Reg16 out) {
        c.mov(out, word[REGS + offsetof(JitRegisters, cfgi)]);
    }

    void SetCfgi(Xbyak::CodeGenerator& c, auto value) {
        c.mov(word[REGS + offsetof(JitRegisters, cfgi)], value);
    }

    void GetCfgj(Xbyak::CodeGenerator& c, Xbyak::Reg16 out) {
        c.mov(out, word[REGS + offsetof(JitRegisters, cfgj)]);
    }

    void SetCfgj(Xbyak::CodeGenerator& c, auto value) {
        c.mov(word[REGS + offsetof(JitRegisters, cfgj)], value);
    }

    void GetStt0(Xbyak::CodeGenerator& c, Xbyak::Reg16 out) {
        c.mov(out, FLAGS);
        c.shr(out, 1); // Flags is the same as Stt0 but has fz in bit0, shift it out.
    }

    template <typename T>
    void SetStt0(Xbyak::CodeGenerator& c, Xbyak::Reg16 scratch, T value) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            UNREACHABLE();
        } else {
            c.and_(FLAGS, decltype(Flags::fz)::mask);
            c.or_(FLAGS, value << 1);
        }
    }

    void GetMod0(Xbyak::CodeGenerator& c, Xbyak::Reg16 out) {
        c.mov(out, word[REGS + offsetof(JitRegisters, mod0)]);
    }

    template <typename T>
    void SetMod0(Xbyak::CodeGenerator& c, T value) {
        // mod0_unk_const is read only, always set it to 1
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            c.and_(value, ~0x1C);
            c.or_(value, 0x4);
        } else {
            value &= ~0x1C;
            value |= 0x4;
        }
        c.mov(word[REGS + offsetof(JitRegisters, mod0)], value);
    }

    void GetMod1(Xbyak::CodeGenerator& c, Xbyak::Reg16 out) {
        c.mov(out, word[REGS + offsetof(JitRegisters, mod1)]);
    }

    template <typename T>
    void SetMod1(Xbyak::CodeGenerator& c, T value) {
        c.mov(word[REGS + offsetof(JitRegisters, mod1)], value);
    }

    void GetMod2(Xbyak::CodeGenerator& c, Xbyak::Reg16 out) {
        c.mov(out, word[REGS + offsetof(JitRegisters, mod2)]);
    }

    template <typename T>
    void SetMod2(Xbyak::CodeGenerator& c, T value) {
        c.mov(word[REGS + offsetof(JitRegisters, mod2)], value);
    }

    void GetSt0(Xbyak::CodeGenerator& c, Xbyak::Reg64 scratch, Xbyak::Reg16 out) {
        /**
         * using st0 = PseudoRegister< // Dynamic
           ProxySlot<Redirector<&JitRegisters::sat>, 0, 1>,
           ProxySlot<Redirector<&JitRegisters::ie>, 1, 1>,
           ProxySlot<ArrayRedirector<3, &JitRegisters::im, 0>, 2, 1>,
           ProxySlot<ArrayRedirector<3, &JitRegisters::im, 1>, 3, 1>,
           ProxySlot<Redirector<&JitRegisters::fr>, 4, 1>,
           ProxySlot<DoubleRedirector<&JitRegisters::flm, &JitRegisters::fvl>, 5, 1>,
           ProxySlot<Redirector<&JitRegisters::fe>, 6, 1>,
           ProxySlot<Redirector<&JitRegisters::fc0>, 7, 1>,
           ProxySlot<Redirector<&JitRegisters::fv>, 8, 1>,
           ProxySlot<Redirector<&JitRegisters::fn>, 9, 1>,
           ProxySlot<Redirector<&JitRegisters::fm>, 10, 1>,
           ProxySlot<Redirector<&JitRegisters::fz>, 11, 1>,
           ProxySlot<AccEProxy<0>, 12, 4>
           >;
         **/
        UNREACHABLE();
        c.xor_(out, out);

        // Load flags and handle fvl latch
        //c.mov(scratch.cvt8(), byte[REGS + offsetof(JitRegisters, stt0)]);
        c.mov(out.cvt8(), scratch.cvt8());
        c.and_(out.cvt8(), 1);
        c.shr(scratch.cvt8(), 1);
        c.or_(out.cvt8(), scratch.cvt8());
        c.shl(out.cvt16(), 5);

        // Load a0e and place it in out.
        c.mov(scratch, A[0]);
        c.shr(scratch, 32);
        c.shl(scratch.cvt16(), 12);
        c.or_(out.cvt16(), scratch.cvt16()); // out |= (a0e << 12)
    }

    void GetSt1(Xbyak::CodeGenerator& c, Xbyak::Reg32 scratch, Xbyak::Reg16 out) {
        // Load mod1 and mod0 in a single 32-bit load.
        static_assert(offsetof(JitRegisters, mod0) - offsetof(JitRegisters, mod1) == sizeof(u16));
        c.mov(scratch, dword[REGS + offsetof(JitRegisters, mod1)]);

        // Copy lowest byte to out, which is page
        c.xor_(out, out);
        c.mov(out.cvt8(), scratch.cvt8());

        // Extract ps0 and place it in out.
        c.shr(scratch, 16); // ps0 is now in bit 10
        c.and_(scratch.cvt16(), 0xC00);
        c.or_(out.cvt16(), scratch.cvt16());

        // Load a1e and place it in out.
        c.mov(scratch.cvt64(), A[1]);
        c.shr(scratch.cvt64(), 32);
        c.shl(scratch.cvt16(), 12);
        c.or_(out.cvt16(), scratch.cvt16()); // out |= (a1e << 12)
    }

    template <typename T>
    void SetSt1(Xbyak::CodeGenerator& c, Xbyak::Reg32 scratch, T value) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            // Load mod1 and mod0 in a single 32-bit load
            c.mov(scratch, dword[REGS + offsetof(JitRegisters, mod1)]);
            // Copy to lower byte of mod1 which is page
            c.mov(scratch.cvt8(), value.cvt8());
            c.and_(value, ~0xFF);

            // Replace ps0 and store back
            c.shl(value, 32 - decltype(St1::a1e_alias)::position);
            c.shr(value.cvt32(), 4);
            c.and_(scratch, ~0xC000000); // Clear existing ps0
            c.or_(scratch, value.cvt32());
            c.mov(dword[REGS + offsetof(JitRegisters, mod1)], scratch);

            // Replace upper word of a[1] with sign extended a1e.
            c.shl(value, 32 - decltype(St1::a1e_alias)::bits);
            c.sar(value, 32 - decltype(St1::a1e_alias)::bits);
            c.mov(value.cvt32(), A[1].cvt32());
            c.mov(A[1], value);
        } else {
            const St1 reg{value};
            // Load mod1 and mod0 in a single 32-bit load
            c.mov(scratch, dword[REGS + offsetof(JitRegisters, mod1)]);
            // Copy to lower byte of mod1 which is page
            c.mov(scratch.cvt8(), reg.page_alias.Value());

            // Replace ps0 and store back
            const u32 mask = static_cast<u32>(reg.ps0_alias.Value()) << 26;
            c.and_(scratch, ~0xC000000); // Clear existing ps0
            c.or_(scratch, mask);
            c.mov(dword[REGS + offsetof(JitRegisters, mod1)], scratch);

            // Replace upper word of a[1] with sign extended a1e.
            const u32 value32 = SignExtend<4, u32>(reg.a1e_alias.Value());
            c.ror(A[1], 32);
            c.mov(A[1].cvt32(), value32);
            c.rol(A[1], 32);
        }
    }

    template <typename T>
    void SetMod3(Xbyak::CodeGenerator& c, T value) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            UNREACHABLE();
        } else {
            // This is awful but the register is seldom accessed anyway...
            const Mod3 reg{.raw = value};
            c.mov(qword[REGS + offsetof(JitRegisters, nimc)], reg.IcQword());
            c.mov(dword[REGS + offsetof(JitRegisters, ou) + sizeof(u16) * 2], reg.OuDword());
            c.mov(word[REGS + offsetof(JitRegisters, ou) + sizeof(u16) * 4], reg.ou4.Value());
            c.mov(word[REGS + offsetof(JitRegisters, ie)], reg.ie.Value());
            c.mov(qword[REGS + offsetof(JitRegisters, im)], reg.ImQword());
            c.mov(qword[REGS + offsetof(JitRegisters, ccnta)], reg.CQword());
        }
    }
};

} // namespace Teakra
