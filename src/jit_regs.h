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
/// Holds multiplication factor registers y0/y1/x0/x1.
constexpr Reg64 FACTORS = r14;
/// Holds the register structure pointer
constexpr Reg64 REGS = r15;
/// Holds commonly used status flags
constexpr Reg32 FLAGS = edi;

// Most frequently accessed status registers, or registers with no cross refrences are stored directly by the JIT for speed
// These registers are also used in a static manner as it greatly reduces the amount of emitted assembly per block.

// This doesn't represent an actual register but the current status flags of the JIT
union Flags {
    u16 raw;
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
    BitField<0, 9, u16> st0_flags; // flags to clear when being set from st0
};

union Factors {
    u64 raw;
    BitField<0, 16, u64> y0;
    BitField<16, 16, u64> y1;
    BitField<32, 16, u64> x0;
    BitField<48, 16, u64> x1;
};

union ArU {
    u16 raw;
    BitField<0, 3, u16> arstep1;
    BitField<3, 2, u16> aroffset1;
    BitField<5, 3, u16> arstep0;
    BitField<8, 2, u16> aroffset0;
    BitField<10, 3, u16> arrn1;
    BitField<13, 3, u16> arrn0;

    static u16 Mask() {
        return decltype(arstep1)::mask | decltype(aroffset1)::mask | decltype(arstep0)::mask
             | decltype(aroffset0)::mask | decltype(arrn1)::mask | decltype(arrn0)::mask;
    }
};

union ArpU {
    u16 raw;
    BitField<0, 3, u16> arpstepi;
    BitField<3, 2, u16> arpoffseti;
    BitField<5, 3, u16> arpstepj;
    BitField<8, 2, u16> arpoffsetj;
    BitField<10, 2, u16> arprni;
    BitField<13, 2, u16> arprnj;

    static u16 Mask() {
        return decltype(arpstepi)::mask | decltype(arpoffseti)::mask | decltype(arpstepj)::mask
               | decltype(arpoffsetj)::mask | decltype(arprni)::mask | decltype(arprnj)::mask;
    }
};

union Mod0 {
    Mod0() {
        mod0_unk_const.Assign(1);
        sata.Assign(1);
    }

    u16 raw{0};
    BitField<0, 1, u16> sat; // 1-bit, disable saturation when moving from acc
    BitField<1, 1, u16> sata; // 1-bit, disable saturation when moving to acc
    BitField<2, 3, u16> mod0_unk_const; // = 1, read only
    BitField<5, 2, u16> hwm; // 2-bit, half word mode, modify y on multiplication
    BitField<7, 1, u16> s; // 1-bit, shift mode. 0 - arithmetic, 1 - logic
    BitField<8, 1, u16> ou0; // user output pins (always zero in firmware?)
    BitField<9, 1, u16> ou1;
    BitField<10, 2, u16> ps0; // 2-bit, product shift mode
    BitField<13, 2, u16> ps1;

    static u16 Mask() {
        return decltype(sat)::mask | decltype(sata)::mask | decltype(mod0_unk_const)::mask
               | decltype(hwm)::mask | decltype(s)::mask | decltype(ou0)::mask | decltype(ou1)::mask | decltype(ps0)::mask | decltype(ps1)::mask;
    }
};

union Mod1 {
    Mod1() {
        cmd.Assign(1);
    }

    u16 raw{0};
    BitField<0, 8, u16> page; // 8-bit, higher part of MemImm8 address
    BitField<12, 1, u16> stp16; // 1 bit. If set, stepi0/j0 will be exchanged along with cfgi/j in banke, and use
                                // stepi0/j0 for steping
    BitField<13, 1, u16> cmd; // 1-bit, step/mod method. 0 - Teak; 1 - TeakLite
    BitField<14, 1, u16> epi; // 1-bit. If set, cause r3 = 0 when steping r3
    BitField<15, 1, u16> epj; // 1-bit. If set, cause r7 = 0 when steping r7

    static u16 Mask() {
        return decltype(page)::mask | decltype(stp16)::mask | decltype(cmd)::mask
               | decltype(epi)::mask | decltype(epj)::mask;
    }
};

union Mod2 {
    u16 raw{0};
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
    BitField<0, 6, u16> m1_5;

    bool IsBr(u32 unit) const {
        return raw & (1 << (unit + decltype(br0)::position));
    }

    bool IsM(u32 unit) const {
        return raw & (1 << (unit + decltype(m0)::position));
    }
};

// Note: This register owns none of its members
union Mod3 {
    u16 raw{0};
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

    u64 OuIeQword() const {
        return ou2.Value()
        | (static_cast<u64>(ou3.Value()) << 16)
        | (static_cast<u64>(ou4.Value()) << 32)
        | (static_cast<u64>(ie.Value()) << 48);
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

    static u16 Mask() {
        return decltype(flm)::mask | decltype(fvl)::mask | decltype(fe)::mask
               | decltype(fc0)::mask | decltype(fv)::mask | decltype(fn)::mask
               | decltype(fm)::mask | decltype(fz)::mask | decltype(fc1)::mask;
    }
};

union Stt1 {
    u16 raw;
    BitField<4, 1, u16> fr;
    BitField<10, 1, u16> iu0;
    BitField<11, 1, u16> iu1;
    BitField<14, 1, u16> pe0;
    BitField<15, 1, u16> pe1;
};

union Stt2 {
    u16 raw;
    BitField<0, 1, u16> ip0;
    BitField<1, 1, u16> ip1;
    BitField<2, 1, u16> ip2;
    BitField<3, 1, u16> ipv;
    BitField<6, 2, u16> pcmhi;
    BitField<12, 3, u16> bcn;
    BitField<15, 1, u16> lp;
};

union St0 {
    u16 raw;
    BitField<0, 1, u16> sat;
    BitField<1, 1, u16> ie;
    BitField<2, 1, u16> im0;
    BitField<3, 1, u16> im1;
    BitField<4, 1, u16> fr;
    BitField<5, 1, u16> flm_fvl;
    BitField<6, 1, u16> fe;
    BitField<7, 1, u16> fc0;
    BitField<8, 1, u16> fv;
    BitField<9, 1, u16> fn;
    BitField<10, 1, u16> fm;
    BitField<11, 1, u16> fz;
    BitField<12, 4, u16> a0e_alias;
    BitField<6, 6, u16> flags_upper;

    u16 ToHostMask() const {
        return fr.Value() | (flm_fvl.Value() << 1) | (flm_fvl.Value() << 2) | (flags_upper.Value() << 3);
    }

    u32 Im0Im1Mask() const {
        return im0.Value() | (im1.Value() << 16);
    }
};

union St1 {
    u16 raw;
    BitField<0, 8, u16> page_alias;
    BitField<10, 2, u16> ps0_alias;
    BitField<12, 4, u16> a1e_alias;
};

union St2 {
    u16 raw;
    BitField<0, 1, u16> m0;
    BitField<1, 1, u16> m1;
    BitField<2, 1, u16> m2;
    BitField<3, 1, u16> m3;
    BitField<4, 1, u16> m4;
    BitField<5, 1, u16> m5;
    BitField<6, 1, u16> im2;
    BitField<7, 1, u16> s;
    BitField<8, 1, u16> ou0;
    BitField<9, 1, u16> ou1;
    BitField<10, 1, u16> iu0;
    BitField<11, 1, u16> iu1;
    BitField<13, 1, u16> ip2;
    BitField<14, 1, u16> ip0;
    BitField<15, 1, u16> ip1;
    BitField<0, 6, u16> m1_5;

    u64 IpMask() const {
        return ip0.Value() | (static_cast<u64>(ip1.Value()) << 16) | (static_cast<u64>(ip2.Value()) << 32);
    }
};

union Cfg {
    Cfg(u16 value = 0) : raw{value} {}
    u16 raw;
    BitField<0, 7, u16> step;
    BitField<7, 9, u16> mod;
};

// NOTE: The jit relies heavily on the member order, do not swap!
struct JitRegisters {
    JitRegisters() {
        mod0b.sata.Assign(0);
        mod1b.cmd.Assign(0);
        ar[0].raw = 0x102c;
        ar[1].raw = 0x5aa3;
        arp[0].raw = 0x21;
        arp[1].raw = 0x258c;
        arp[2].raw = 0x4ab5;
        arp[3].raw = 0x6c63;
    }
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

    void GetLc(Xbyak::CodeGenerator& c, Reg64 out) {
        c.movzx(rsi, word[REGS + offsetof(JitRegisters, bcn)]);
        c.sub(rsi, 1);
        c.lea(rsi, ptr[rsi + rsi * 2]);
        c.mov(out, word[REGS + offsetof(JitRegisters, bkrep_stack) + offsetof(BlockRepeatFrame, lc)]);
        c.test(word[REGS + offsetof(JitRegisters, lp)], 0x1);
        c.cmovnz(out, word[REGS + offsetof(JitRegisters, bkrep_stack) + offsetof(BlockRepeatFrame, lc) + rsi * 4]);
    }

    void SetLc(Xbyak::CodeGenerator& c, Reg64 value) {
        Xbyak::Label end_label, not_in_loop;
        c.test(word[REGS + offsetof(JitRegisters, lp)], 0x1);
        c.jz(not_in_loop);
        c.movzx(rsi, word[REGS + offsetof(JitRegisters, bcn)]);
        c.sub(rsi, 1);
        c.lea(rsi, ptr[rsi + rsi * 2]);
        c.mov(word[REGS + offsetof(JitRegisters, bkrep_stack) + offsetof(BlockRepeatFrame, lc) + rsi * 4], value.cvt16());
        c.jmp(end_label);
        c.L(not_in_loop);
        c.mov(word[REGS + offsetof(JitRegisters, bkrep_stack) + offsetof(BlockRepeatFrame, lc)], value.cvt16());
        c.L(end_label);
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
    u16 pad7 = 0;

    // Shadows
    Flags flagsb{};
    u16 pad8 = 0;

    // Viterbi
    u16 vtr0 = 0;
    u16 vtr1 = 0;

    /** Multiplication unit **/

    std::array<u16, 2> y{};  // factor
    std::array<u16, 2> x{};  // factor
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

    /** Extension unit **/

    std::array<u16, 5> ou{}; // user output pins
    // interrupt enable master bit
    u16 ie = 0;
    std::array<u16, 2> iu{}; // user input pins
    std::array<u16, 4> ext{};

    // shadow swap registers
    u16 pcmhib = 0; // 2-bit, higher part of program address for movp/movd
    Mod1 mod1b{};
    Mod0 mod0b{};
    Mod2 mod2b{};
    std::array<u16, 3> imb{}; // interrupt enable bit
    u16 imvb = 0;

    void ShadowStore(Xbyak::CodeGenerator& c) {
        c.mov(word[REGS + offsetof(JitRegisters, flagsb)], FLAGS);
    }

    void ShadowRestore(Xbyak::CodeGenerator& c) {
        c.mov(FLAGS, word[REGS + offsetof(JitRegisters, flagsb)]);
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
        c.and_(out, Stt0::Mask());
    }

    template <typename T>
    void SetStt0(Xbyak::CodeGenerator& c, T value) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            c.and_(FLAGS, decltype(Flags::fr)::mask);
            c.shl(value, 1);
            c.or_(FLAGS.cvt16(), value.cvt16());
        } else {
            c.and_(FLAGS, decltype(Flags::fr)::mask);
            c.or_(FLAGS, value << 1);
        }
    }

    void GetStt1(Xbyak::CodeGenerator& c, Xbyak::Reg16 out) {
        c.mov(out, FLAGS.cvt16());
        c.mov(esi, dword[REGS + offsetof(JitRegisters, iu)]);
        c.and_(out, 1);
        c.ror(out, 6);
        c.or_(out.cvt8(), esi.cvt8());
        c.shr(esi, 15);
        c.or_(out.cvt8(), esi.cvt8());
        c.ror(out, 4);
        c.mov(esi, dword[REGS + offsetof(JitRegisters, pe)]);
        c.or_(out.cvt8(), esi.cvt8());
        c.shr(esi, 15);
        c.or_(out.cvt8(), esi.cvt8());
        c.ror(out, 2);
    }

    template <typename T>
    void SetStt1(Xbyak::CodeGenerator& c, T value) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            c.and_(value, decltype(Stt1::fr)::mask | decltype(Stt1::pe0)::mask
                              | decltype(Stt1::pe1)::mask); // mask out read only bits
            c.shr(value, decltype(Stt1::fr)::position);
            c.and_(FLAGS, ~decltype(Flags::fr)::mask); // clear fr
            c.or_(FLAGS.cvt8(), value.cvt8());
            c.bt(value, decltype(Stt1::pe0)::position - decltype(Stt1::fr)::position);
            c.setc(byte[REGS + offsetof(JitRegisters, pe)]);
            c.bt(value, decltype(Stt1::pe1)::position - decltype(Stt1::fr)::position);
            c.setc(byte[REGS + offsetof(JitRegisters, pe) + sizeof(u16)]);
        } else {
            UNREACHABLE();
        }
    }

    void GetStt2(Xbyak::CodeGenerator& c, Xbyak::Reg16 out) {
        c.xor_(out, out);
        c.mov(si, word[REGS + offsetof(JitRegisters, bcn)]);
        c.mov(out, word[REGS + offsetof(JitRegisters, lp)]); // Load lp
        c.shl(out, 3);
        c.or_(out, si); // Load bcn
        c.shl(out, 6);
        c.mov(si, word[REGS + offsetof(JitRegisters, pcmhi)]);
        c.or_(out, si); // Load bcn
        c.shl(out, 6);

        c.mov(rsi, qword[REGS + offsetof(JitRegisters, ip)]);
        c.or_(out, si); // Load ip and ipv
        c.shr(rsi, 15);
        c.or_(out, si);
        c.shr(rsi, 15);
        c.or_(out, si);
        c.shr(rsi, 15);
        c.or_(out, si);
    }

    template <typename T>
    void SetStt2(Xbyak::CodeGenerator& c, T value) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            // Most members are ready only, great for us :)
            // Mask out read only bits
            c.and_(value, decltype(Stt2::lp)::mask | decltype(Stt2::pcmhi)::mask);
            c.bt(value, decltype(Stt2::lp)::position);
            c.setc(byte[REGS + offsetof(JitRegisters, lp)]);
            c.shr(value, decltype(Stt2::pcmhi)::position);
            c.mov(byte[REGS + offsetof(JitRegisters, pcmhi)], value.cvt8());
        } else {
            UNREACHABLE();
        }
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

    template <typename T>
    void SetMod1(Xbyak::CodeGenerator& c, T value) {
        c.mov(word[REGS + offsetof(JitRegisters, mod1)], value);
    }

    template <typename T>
    void SetMod2(Xbyak::CodeGenerator& c, T value) {
        c.mov(word[REGS + offsetof(JitRegisters, mod2)], value);
    }

    void GetSt0(Xbyak::CodeGenerator& c, Xbyak::Reg16 out, Mod0& mod0_const) {
        c.mov(out, mod0_const.sat.Value() << 15);
        c.mov(out.cvt8(), byte[REGS + offsetof(JitRegisters, ie)]);
        c.ror(out, 1);
        c.mov(esi, dword[REGS + offsetof(JitRegisters, im)]);
        c.or_(out.cvt8(), sil);
        c.shr(rsi, 15);
        c.or_(out.cvt8(), sil);
        c.ror(out, 2);
        c.mov(si, FLAGS.cvt16());
        c.shr(si, 2);
        c.shl(si, 1);
        c.or_(out.cvt8(), FLAGS.cvt8());
        c.and_(out.cvt8(), 0b11);
        c.or_(out.cvt8(), sil);
        c.ror(out, 8);
        c.rorx(rsi, A[0], 32);
        c.and_(rsi, 0xFF);
        c.or_(out, sil);
        c.ror(out, 4);
    }

    template <typename T>
    void SetSt0(Xbyak::CodeGenerator& c, T value, Mod0& mod0_const) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            // Set sat in mod0
            c.and_(byte[REGS + offsetof(JitRegisters, mod0)], ~decltype(Mod0::sat)::mask);
            c.bt(value, decltype(St0::sat)::position);
            c.setc(sil);
            c.or_(word[REGS + offsetof(JitRegisters, mod0)], sil);

            // Set ie, im0, im1
            c.bt(value, decltype(St0::ie)::position);
            c.setc(byte[REGS + offsetof(JitRegisters, ie)]);
            c.bt(value, decltype(St0::im0)::position);
            c.setc(byte[REGS + offsetof(JitRegisters, im)]);
            c.bt(value, decltype(St0::im1)::position);
            c.setc(byte[REGS + offsetof(JitRegisters, im) + sizeof(u16)]);

            // Update flags.
            c.and_(FLAGS, ~decltype(Flags::st0_flags)::mask);
            c.rorx(rsi, value, decltype(St0::fr)::position);
            c.and_(rsi, 0x3);
            c.or_(FLAGS.cvt8(), sil);
            c.rorx(rsi, value, decltype(St0::flm_fvl)::position);
            c.shl(rsi, 2);
            c.and_(rsi, 0x1fc);
            c.or_(FLAGS.cvt16(), si);

            // Replace upper word of a[0] with sign extended a0e.
            c.and_(value, decltype(St0::a0e_alias)::mask);
            c.shl(value, 64 - decltype(St0::a0e_alias)::position - decltype(St0::a0e_alias)::bits);
            c.sar(value, 64 - decltype(St0::a0e_alias)::bits);
            c.mov(value.cvt32(), A[0].cvt32());
            c.mov(A[0], value);
        } else {
            const St0 reg{.raw = value};

            // Update flags.
            c.and_(FLAGS, ~decltype(Flags::st0_flags)::mask);
            c.or_(FLAGS, reg.ToHostMask());

            // Set ie and im0/im1
            c.mov(word[REGS + offsetof(JitRegisters, ie)], reg.ie.Value());
            c.mov(dword[REGS + offsetof(JitRegisters, im)], reg.Im0Im1Mask());

            // Update sat field of mod0
            mod0_const.sat.Assign(reg.sat.Value());
            c.mov(word[REGS + offsetof(JitRegisters, mod0)], mod0_const.raw);

            // Replace upper word of a[0] with sign extended a0e.
            const u64 value32 = SignExtend<4, u32>(reg.a0e_alias.Value());
            c.mov(rsi, value32 << 32);
            c.mov(rsi.cvt32(), A[0].cvt32());
            c.mov(A[0], rsi);
        }
    }

    void GetSt1(Xbyak::CodeGenerator& c, Xbyak::Reg16 out, Mod0& mod0_const, Mod1& mod1_const) {
        // Copy lowest byte to out, which is page
        c.xor_(out, out);
        c.mov(out.cvt8(), mod1_const.page.Value());

        // Extract ps0 and place it in out.
        c.or_(out.cvt16(), mod0_const.ps0.Value() << decltype(St1::ps0_alias)::position);

        // Load a1e and place it in out.
        c.rorx(rsi, A[1], 32);
        c.shl(rsi, decltype(St1::a1e_alias)::position);
        c.or_(out.cvt16(), si);
    }

    template <typename T>
    void SetSt1(Xbyak::CodeGenerator& c, T value, Mod0& mod0_const, Mod1& mod1_const) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            Mod0 mod0 = mod0_const;
            mod0.ps0.Assign(0);
            const u32 mod1mod0 = mod1_const.raw | (static_cast<u32>(mod0.raw) << 16);

            // Load mod1 and mod0 in a single 32-bit load
            c.mov(rsi.cvt32(), mod1mod0);
            c.mov(rsi.cvt8(), value.cvt8());

            // Replace ps0 and store back
            c.shr(value, decltype(St1::ps0_alias)::position);
            c.shl(value, 32 - decltype(St1::ps0_alias)::bits);
            c.shr(value.cvt32(), 4);
            c.or_(rsi, value.cvt32());
            c.mov(dword[REGS + offsetof(JitRegisters, mod1)], rsi.cvt32());

            // Replace upper word of a[1] with sign extended a1e.
            c.shl(value, 32 - decltype(St1::a1e_alias)::bits);
            c.sar(value, 32 - decltype(St1::a1e_alias)::bits);
            c.mov(value.cvt32(), A[1].cvt32());
            c.mov(A[1], value);
        } else {
            const St1 reg{.raw = value};
            // Copy to lower byte of mod1 which is page
            // Replace ps0 and store to both mod1 and mod0
            mod1_const.page.Assign(reg.page_alias.Value());
            mod0_const.ps0.Assign(reg.ps0_alias.Value());
            const u32 value = mod1_const.raw | (static_cast<u32>(mod0_const.raw) << 16);
            c.mov(dword[REGS + offsetof(JitRegisters, mod1)], value);

            // Replace upper word of a[1] with sign extended a1e.
            const u64 value32 = SignExtend<4, u32>(reg.a1e_alias.Value());
            c.mov(rsi, value32 << 32);
            c.mov(rsi.cvt32(), A[1].cvt32());
            c.mov(A[1], rsi);
        }
    }

    template <typename T>
    void SetSt2(Xbyak::CodeGenerator& c, T value, Mod0& mod0_const, Mod2& mod2_const) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            UNREACHABLE();
        } else {
            const St2 reg{.raw = value};

            // Update mod0 and mod2
            mod0_const.s.Assign(reg.s.Value());
            mod2_const.m1_5.Assign(reg.m1_5.Value());

            // Update im, ou. iu and ip are read only
            c.mov(word[REGS + offsetof(JitRegisters, im) + sizeof(u16) * 2], reg.im2.Value());
            c.mov(dword[REGS + offsetof(JitRegisters, ou)], reg.ou0.Value() | (static_cast<u32>(reg.ou1.Value()) << 16));
        }
    }

    void GetMod3(Xbyak::CodeGenerator& c, Xbyak::Reg64 out) {
        // TODO: Use pext, my laptop is Zen 2 :<
        c.xor_(out, out);
        c.mov(rsi, qword[REGS + offsetof(JitRegisters, nimc)]);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8()); // nimc-ic2

        c.mov(rsi, qword[REGS + offsetof(JitRegisters, ou) + sizeof(u16) * 2]);
        c.shl(rsi, 4);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8()); // ou2-ie

        c.rol(out.cvt16(), 8);

        c.mov(rsi, qword[REGS + offsetof(JitRegisters, im)]);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8()); // im0-imv

        c.mov(rsi, qword[REGS + offsetof(JitRegisters, ccnta)]);
        c.shl(rsi, 5);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8());
        c.shr(rsi, 15);
        c.or_(out.cvt8(), rsi.cvt8()); // ccnta-crep

        c.rol(out.cvt16(), 8);
    }

    template <typename T>
    void SetMod3(Xbyak::CodeGenerator& c, T value) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            // TODO: Use pdep, my laptop is Zen 2 :<
            const auto nibble_to_qword = [&](u32 offset) {
                c.bt(value, 3 + offset);
                c.setc(rsi.cvt8());
                c.shl(rsi, 16);
                c.bt(value, 2 + offset);
                c.setc(rsi.cvt8());
                c.shl(rsi, 16);
                c.bt(value, 1 + offset);
                c.setc(rsi.cvt8());
                c.shl(rsi, 16);
                c.bt(value, 0 + offset);
                c.setc(rsi.cvt8());
            };

            c.xor_(rsi, rsi);
            nibble_to_qword(0);
            c.mov(qword[REGS + offsetof(JitRegisters, nimc)], rsi); // nimc-ic2
            nibble_to_qword(4);
            c.mov(qword[REGS + offsetof(JitRegisters, ou) + sizeof(u16) * 2], rsi); // ou2-ie
            nibble_to_qword(8);
            c.mov(qword[REGS + offsetof(JitRegisters, im)], rsi); // im0-imv
            nibble_to_qword(13);
            c.mov(qword[REGS + offsetof(JitRegisters, ccnta)], rsi);
        } else {
            const Mod3 reg{.raw = value};
            c.mov(qword[REGS + offsetof(JitRegisters, nimc)], reg.IcQword());
            c.mov(qword[REGS + offsetof(JitRegisters, ou) + sizeof(u16) * 2], reg.OuIeQword());
            c.mov(qword[REGS + offsetof(JitRegisters, im)], reg.ImQword());
            c.mov(qword[REGS + offsetof(JitRegisters, ccnta)], reg.CQword());
        }
    }
};

} // namespace Teakra
