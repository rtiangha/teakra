#pragma once
#pragma clang optimize off
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
#include "interpreter.h"
#include "operand.h"
#include "hash.h"
#include "jit_regs.h"
#include "register.h"
#include "xbyak_abi.h"

namespace Teakra {

constexpr size_t MAX_CODE_SIZE = 4 * 1024 * 1024;

struct alignas(16) StackLayout {
    s64 cycles_remaining;
    s64 cycles_to_run;
};

class EmitX64 {
public:
    EmitX64(CoreTiming& core_timing, JitRegisters& regs, MemoryInterface& mem)
        : core_timing(core_timing), regs(regs), mem(mem), interp(core_timing, iregs, mem), c(MAX_CODE_SIZE) {
        EmitDispatcher();
    }

    using RunCodeFuncType = void (*)(EmitX64*);
    RunCodeFuncType run_code{};

    using BlockFunc = void(*)(JitRegisters*);

    struct BlockSwapState {
        Cfg cfgi, cfgj;
        u16 stepi0, stepj0;
        Mod0 mod0;
        Mod1 mod1;
        Mod2 mod2;
        std::array<ArpU, 4> arp;
        std::array<ArU, 2> ar;
    };

    struct BlockKey {
        BlockSwapState curr;
        BlockSwapState shadow;
        u32 pc;
    };

    struct Block {
        BlockFunc func;
        BlockKey entry_key;
        s32 cycles;
    };

    CoreTiming& core_timing;
    JitRegisters& regs;
    RegisterState iregs;
    MemoryInterface& mem;
    Interpreter interp;
    Xbyak::CodeGenerator c;
    s32 cycles_remaining;
    Xbyak::Label dispatcher_start;
    const std::vector<Matcher<EmitX64>> decoders = GetDecoderTable<EmitX64>();
    bool compiling = false;
    std::unordered_map<size_t, Block> code_blocks;
    Block* current_blk{};
    BlockKey blk_key{};

    void Run(s64 cycles) {
        cycles_remaining = cycles;
        run_code(this);
    }

    void EmitDispatcher() {
        run_code = c.getCurr<RunCodeFuncType>();
        ABI_PushRegistersAndAdjustStack(c, ABI_ALL_CALLEE_SAVED, 8, 16);

        Xbyak::Label dispatcher_end;
        c.L(dispatcher_start);

        // Call LookupBlock. This will compile a new block and increment timers for us
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(this));
        c.mov(ABI_RETURN, reinterpret_cast<uintptr_t>(LookupNewBlockThunk));
        c.call(ABI_RETURN);

        c.test(ABI_RETURN, ABI_RETURN);
        c.jz(dispatcher_end);
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(&regs));
        c.jmp(ABI_RETURN);
        c.L(dispatcher_end);
    }

    static BlockFunc LookupNewBlockThunk(void* this_ptr) {
        return reinterpret_cast<EmitX64*>(this_ptr)->LookupNewBlock();
    }

    BlockFunc LookupNewBlock() {
        if (regs.idle) {
            u64 skipped = core_timing.Skip(cycles_remaining - 1);
            cycles_remaining -= skipped;
            // Skip additional tick so to let components fire interrupts
            if (cycles_remaining > 0) {
                cycles_remaining--;
                core_timing.Tick();
            }
        }

        // Count the cycles of the previous executed block and handle any interrupts
        if (current_blk) {
            ASSERT(CompareRegisterState());
            core_timing.Tick(current_blk->cycles);
            cycles_remaining -= current_blk->cycles;
        }

        // Check for interrupts.
        for (std::size_t i = 0; i < 3; ++i) {
            if (interrupt_pending[i].exchange(false)) {
                regs.ip[i] = 1;
            }
        }

        if (vinterrupt_pending.exchange(false)) {
            regs.ipv = 1;
        }

        if (regs.ie && !regs.rep) {
            bool interrupt_handled = false;
            for (u32 i = 0; i < regs.im.size(); ++i) {
                if (regs.im[i] && regs.ip[i]) {
                    regs.ip[i] = 0;
                    regs.ie = 0;
                    PushPC();
                    regs.pc = 0x0006 + i * 8;
                    regs.idle = false;
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
                regs.idle = false;
                if (vinterrupt_context_switch) {
                    ContextStore();
                }
            }
        }

        // Lookup and compile next block
        blk_key.pc = regs.pc;

        // Current state
        blk_key.curr.mod0 = regs.mod0;
        blk_key.curr.mod1 = regs.mod1;
        blk_key.curr.mod2 = regs.mod2;
        blk_key.curr.cfgi = regs.cfgi;
        blk_key.curr.cfgj = regs.cfgj;
        blk_key.curr.stepi0 = regs.stepi0;
        blk_key.curr.stepj0 = regs.stepj0;
        blk_key.curr.ar = regs.ar;
        blk_key.curr.arp = regs.arp;

        // Shadow state.
        blk_key.shadow.mod0 = regs.mod0b;
        blk_key.shadow.mod1 = regs.mod1b;
        blk_key.shadow.mod2 = regs.mod2b;
        blk_key.shadow.cfgi = regs.cfgib;
        blk_key.shadow.cfgj = regs.cfgjb;
        blk_key.shadow.stepi0 = regs.stepi0b;
        blk_key.shadow.stepj0 = regs.stepj0b;
        blk_key.shadow.ar = regs.arb;
        blk_key.shadow.arp = regs.arpb;

        const size_t hash = Common::ComputeStructHash64(blk_key);
        auto [it, new_block] = code_blocks.try_emplace(hash);
        if (new_block) {
            // Note: They key may change during compilation, so this needs to be first.
            it->second.entry_key = blk_key;
            CompileBlock(it->second);
        }

        // Check if we have enough cycles to execute it
        current_blk = &it->second;
        if (cycles_remaining < current_blk->cycles) {
            return nullptr;
        }

        // DEBUG: Run interpreter before running block. We will compare register
        // state when the dispatcher is re-entered to ensure JIT was correct.
        interp.Run(current_blk->cycles);

        // Return block function
        return current_blk->func;
    }

    void CompileBlock(Block& blk) {
        // Load block state
        blk.func = c.getCurr<BlockFunc>();
        c.mov(REGS, ABI_PARAM1);
        c.mov(R0_1_2_3, qword[REGS + offsetof(JitRegisters, r)]);
        c.mov(R4_5_6_7, qword[REGS + offsetof(JitRegisters, r) + sizeof(u16) * 4]);
        c.mov(FACTORS, qword[REGS + offsetof(JitRegisters, x)]);
        c.mov(A[0], qword[REGS + offsetof(JitRegisters, a)]);
        c.mov(A[1], qword[REGS + offsetof(JitRegisters, a) + sizeof(u64)]);
        c.mov(B[0], qword[REGS + offsetof(JitRegisters, b)]);
        c.mov(B[1], qword[REGS + offsetof(JitRegisters, b) + sizeof(u64)]);
        c.mov(FLAGS, dword[REGS + offsetof(JitRegisters, flags)]);

        compiling = true;
        while (compiling) {
            u16 opcode = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
            auto& decoder = decoders[opcode];
            u16 expand_value = 0;
            if (decoder.NeedExpansion()) {
                expand_value = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
            }

            decoder.call(*this, opcode, expand_value);
            blk.cycles++;
            if (regs.pc == 0x0000697F) {
                compiling = false;
            }
        }

        // Flush block state
        c.mov(qword[REGS + offsetof(JitRegisters, r)], R0_1_2_3);
        c.mov(qword[REGS + offsetof(JitRegisters, r) + sizeof(u16) * 4], R4_5_6_7);
        c.mov(qword[REGS + offsetof(JitRegisters, x)], FACTORS);
        c.mov(qword[REGS + offsetof(JitRegisters, a)], A[0]);
        c.mov(qword[REGS + offsetof(JitRegisters, a) + sizeof(u64)], A[1]);
        c.mov(qword[REGS + offsetof(JitRegisters, b)], B[0]);
        c.mov(qword[REGS + offsetof(JitRegisters, b) + sizeof(u64)], B[1]);
        c.mov(dword[REGS + offsetof(JitRegisters, flags)], FLAGS);
        c.jmp(dispatcher_start);
    }

    void EmitPushPC() {
        u16 l = (u16)(regs.pc & 0xFFFF);
        u16 h = (u16)(regs.pc >> 16);
        const Reg16 sp = ax;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        if (regs.cpc == 1) {
            UNREACHABLE();
        } else {
            StoreToMemory(sp, l);
            c.sub(sp, 1);
            StoreToMemory(sp, h);
        }
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp);
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

    void EmitPopPC() {
        const Reg16 sp = bx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 pc = rcx;
        if (regs.cpc == 1) {
            UNREACHABLE();
        } else {
            LoadFromMemory(pc, sp);
            c.add(sp, 1);
            c.shl(pc, 16);
            LoadFromMemory(pc, sp);
            c.add(sp, 1);
        }
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp);
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], pc.cvt32());
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
        compiling = false;
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

    void undefined(u16 opcode) {
        std::printf("Undefined opcode: 0x%x\n", opcode);
        UNREACHABLE();
    }

    void ContextStore() {
        UNREACHABLE();
    }

    void ContextRestore() {
        UNREACHABLE();
    }

    void norm(Ax a, Rn b, StepZIDS bs) {
        UNREACHABLE();
    }
    void swap(SwapType swap) {
        UNREACHABLE();
    }
    void trap() {
        UNREACHABLE();
    }

    static u16 MemDataReadThunk(void* mem_ptr, u16 address) {
        return reinterpret_cast<MemoryInterface*>(mem_ptr)->DataRead(address);
    }

    void LoadFromMemory(Reg64 out, MemImm8 addr) {
        LoadFromMemory(out, addr.Unsigned16() + (blk_key.curr.mod1.page << 8));
    }

    template <typename T>
    void LoadFromMemory(Reg64 out, T addr) {
        // TODO: Non MMIO reads can be performed inside the JIT.
        ABI_PushRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(&mem));
        c.mov(ABI_PARAM2, addr);
        CallFarFunction(c, MemDataReadThunk);
        ABI_PopRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.mov(out.cvt16(), ABI_RETURN);
    }

    void DoMultiplication(u32 unit, Reg32 x, Reg32 y, bool x_sign, bool y_sign) {
        c.mov(y.cvt64(), FACTORS);
        if (unit == 0) {
            c.movzx(x, FACTORS.cvt16());
            c.shr(y.cvt64(), 32);
            c.and_(y, 0xFFFF);
        } else {
            c.shr(y.cvt64(), 16);
            c.movzx(x, y.cvt16());
            c.shr(y.cvt64(), 32);
        }
        if (blk_key.curr.mod0.hwm == 1 || (blk_key.curr.mod0.hwm == 3 && unit == 0)) {
            c.shr(y, 8);
        } else if (blk_key.curr.mod0.hwm == 2 || (blk_key.curr.mod0.hwm == 3 && unit == 1)) {
            c.and_(y, 0xFF);
        }
        if (x_sign) {
            SignExtend(x, 16);
        }
        if (y_sign) {
            SignExtend(y, 16);
        }
        c.imul(x, y);
        c.mov(dword[REGS + offsetof(JitRegisters, p) + sizeof(u32) * unit], x);
        if (x_sign || y_sign) {
            c.bt(x, 31);
            c.setc(x.cvt16());
            c.mov(word[REGS + offsetof(JitRegisters, pe) + sizeof(u16) * unit], x.cvt16());
        } else {
            c.mov(word[REGS + offsetof(JitRegisters, pe) + sizeof(u16) * unit], 0);
        }
    }

    void AlmGeneric(AlmOp op, Reg64 a, Ax b) {
        const Reg64 value = rax;
        switch (op) {
        case AlmOp::Or: {
            GetAcc(value, b.GetName());
            c.or_(value, a);
            SignExtend(value, 40);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::And: {
            GetAcc(value, b.GetName());
            c.and_(value, a);
            SignExtend(value, 40);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::Xor: {
            GetAcc(value, b.GetName());
            c.xor_(value, a);
            SignExtend(value, 40);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::Tst0: {
            GetAcc(value, b.GetName());
            c.and_(value, 0xFFFF);
            c.and_(FLAGS, ~decltype(Flags::fz)::mask); // clear fz
            c.test(value, a);
            c.setz(a);
            c.shl(a, decltype(Flags::fz)::position);
            c.or_(FLAGS, a); // regs.fz = (value & a) == 0;
            break;
        }
        case AlmOp::Tst1: {
            GetAcc(value, b.GetName());
            c.and_(value, 0xFFFF);
            c.and_(FLAGS, ~decltype(Flags::fz)::mask); // clear fz
            c.not_(a);
            c.test(value, a);
            c.setz(a);
            c.shl(a, decltype(Flags::fz)::position);
            c.or_(FLAGS, a); // regs.fz = (value & ~a) == 0;
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
            GetAcc(value, b.GetName());
            const bool sub = !(op == AlmOp::Add || op == AlmOp::Addl || op == AlmOp::Addh);
            const Reg64 result = rcx;
            AddSub(value, a, result, sub);
            if (op == AlmOp::Cmp || op == AlmOp::Cmpu) {
                SetAccFlag(result);
            } else {
                SatAndSetAccAndFlag(b.GetName(), result);
            }
            break;
        }
        case AlmOp::Msu: {
            GetAcc(value, b.GetName());
            const Reg64 product = rcx;
            ProductToBus40(product, Px{0});
            const Reg64 result = rdx;
            AddSub(value, product, result, true);
            SatAndSetAccAndFlag(b.GetName(), result);

            c.mov(FACTORS.cvt16(), a.cvt16()); // regs.x[0] = a & 0xFFFF;
            DoMultiplication(0, eax, ebx, true, true);
            break;
        }
        case AlmOp::Sqra: {
            GetAcc(value, b.GetName());
            const Reg64 product = rcx;
            ProductToBus40(product, Px{0});
            const Reg64 result = rdx;
            AddSub(value, product, result, false);
            SatAndSetAccAndFlag(b.GetName(), result);
        }
        [[fallthrough]];
        case AlmOp::Sqr: {
            c.and_(a, 0xFFFF);
            c.mov(FACTORS.cvt16(), a.cvt16());
            c.ror(FACTORS, 32);
            c.mov(FACTORS.cvt16(), a.cvt16());
            c.rol(FACTORS, 32);
            DoMultiplication(0, eax, ebx, true, true);
            break;
        }

        default:
            UNREACHABLE();
        }
    }

    void ExtendOperandForAlm(AlmOp op, Reg64 a) {
        switch (op) {
        case AlmOp::Cmp:
        case AlmOp::Sub:
        case AlmOp::Add:
            SignExtend(a, 16);
            break;
        case AlmOp::Addh:
        case AlmOp::Subh:
            c.shl(a, 16);
            SignExtend(a, 32);
            break;
        default:
            c.movzx(a.cvt64(), a.cvt16());
            break;
        }
    }

    u64 ExtendOperandForAlm(AlmOp op, u16 a) {
        switch (op) {
        case AlmOp::Cmp:
        case AlmOp::Sub:
        case AlmOp::Add:
            return ::SignExtend<16, u64>(a);
        case AlmOp::Addh:
        case AlmOp::Subh:
            return ::SignExtend<32, u64>((u64)a << 16);
        default:
            return a;
        }
    }

    void alm(Alm op, MemImm8 a, Ax b) {
        const Reg64 value = rbx;
        LoadFromMemory(value, a);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }
    void alm(Alm op, Rn a, StepZIDS as, Ax b) {
        UNREACHABLE();
    }
    void alm(Alm op, Register a, Ax b) {
        const Reg64 value = rbx;
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
            ProductToBus40(value, Px{0});
            break;
        case RegName::a0:
        case RegName::a1:
            CheckBus40OperandAllowed();
            GetAcc(value, a.GetName());
            break;
        default:
            RegToBus16(a.GetName(), value);
            ExtendOperandForAlm(op.GetName(), value);
            break;
        }
        AlmGeneric(op.GetName(), value, b);
    }
    void alm_r6(Alm op, Ax b) {
        UNREACHABLE();
    }

    void alu(Alu op, MemImm16 a, Ax b) {
        UNREACHABLE();
    }
    void alu(Alu op, MemR7Imm16 a, Ax b) {
        UNREACHABLE();
    }
    void alu(Alu op, Imm16 a, Ax b) {
        u16 value = a.Unsigned16();
        c.mov(rbx, ExtendOperandForAlm(op.GetName(), value));
        AlmGeneric(op.GetName(), rbx, b);
    }
    void alu(Alu op, Imm8 a, Ax b) {
        u16 value = a.Unsigned16();
        const Reg64 and_backup = rsi;
        c.xor_(and_backup, and_backup);
        if (op.GetName() == AlmOp::And) {
            // AND instruction has a special treatment:
            // bit 8~15 are unaffected in the accumulator, but the flags are set as if they are
            // affected
            GetAcc(and_backup, b.GetName());
            c.and_(and_backup, 0xFF00);
        }
        value = ExtendOperandForAlm(op.GetName(), value);
        c.mov(rbx, value);
        AlmGeneric(op.GetName(), rbx, b);
        if (op.GetName() == AlmOp::And) {
            const Reg64 and_new = rax;
            GetAcc(and_new, b.GetName());
            c.and_(and_new.cvt16(), 0x00FF);
            c.or_(and_new, and_backup);
            SetAcc(b.GetName(), and_new);
        }
    }
    void alu(Alu op, MemR7Imm7s a, Ax b) {
        UNREACHABLE();
    }

    void or_(Ab a, Ax b, Ax c) {
        GetAcc(rax, a.GetName());
        GetAcc(rbx, b.GetName());
        this->c.or_(rax, rbx);
        SetAccAndFlag(c.GetName(), rax);
    }
    void or_(Ax a, Bx b, Ax c) {
        UNREACHABLE();
    }
    void or_(Bx a, Bx b, Ax c) {
        UNREACHABLE();
    }

    u16 GenericAlbConst(Alb op, u16 a, u16 b) {
        u16 result;
        switch (op.GetName()) {
        case AlbOp::Set: {
            result = a | b;
            if (result >> 15) {
                c.bts(FLAGS, decltype(Flags::fm)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fm)::position);
            }
            break;
        }
        case AlbOp::Rst: {
            result = ~a & b;
            if (result >> 15) {
                c.bts(FLAGS, decltype(Flags::fm)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fm)::position);
            }
            break;
        }
        case AlbOp::Chng: {
            result = a ^ b;
            if (result >> 15) {
                c.bts(FLAGS, decltype(Flags::fm)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fm)::position);
            }
            break;
        }
        case AlbOp::Addv: {
            u32 r = a + b;
            if ((r >> 16) != 0) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            if ((::SignExtend<16, u32>(b) + ::SignExtend<16, u32>(a)) >> 31) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
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
            if ((r >> 16) != 0) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            if ((::SignExtend<16, u32>(b) + ::SignExtend<16, u32>(a)) >> 31) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            result = r & 0xFFFF;
            break;
        }
        default:
            UNREACHABLE();
        }
        if (result == 0) {
            c.bts(FLAGS, decltype(Flags::fz)::position);
        } else {
            c.btr(FLAGS, decltype(Flags::fz)::position);
        }
        return result;
    }

    void GenericAlb(Alb op, u16 a, Reg16 b, Reg16 result) {
        switch (op.GetName()) {
        case AlbOp::Set: {
            c.or_(b, a);
            c.mov(result, b);
            c.shr(b, 15);
            c.shl(b, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~decltype(Flags::fm)::mask);
            c.or_(FLAGS, b);
            break;
        }
        case AlbOp::Rst: {
            c.and_(b, ~a);
            c.mov(result, b);
            c.shr(b, 15);
            c.shl(b, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~decltype(Flags::fm)::mask);
            c.or_(FLAGS, b);
            break;
        }
        case AlbOp::Chng: {
            c.xor_(b, a);
            c.mov(result, b);
            c.shr(b, 15);
            c.shl(b, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~decltype(Flags::fm)::mask);
            c.or_(FLAGS, b);
            break;
        }
        case AlbOp::Addv: {
            c.movsx(result.cvt32(), b);
            c.add(result.cvt32(), ::SignExtend<16, u32>(a));
            c.shr(result.cvt32(), 31);
            c.shl(result, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~(decltype(Flags::fm)::mask | decltype(Flags::fc0)::mask));
            c.or_(FLAGS, result);

            c.movzx(result.cvt32(), b);
            c.add(result.cvt32(), static_cast<u32>(a));
            c.mov(b.cvt32(), result.cvt32());
            c.test(b.cvt32(), ~0xFFFF);
            c.setnz(b.cvt8());
            c.shl(b.cvt8(), decltype(Flags::fc0)::position);
            c.or_(FLAGS.cvt8(), b.cvt8());
            break;
        }
        case AlbOp::Tst0: {
            c.xor_(result, result);
            c.and_(b, a);
            c.test(b, b);
            c.setnz(result.cvt8());
            break;
        }
        case AlbOp::Tst1: {
            c.xor_(result, result);
            c.not_(b);
            c.and_(b, a);
            c.test(b, b);
            c.setnz(result.cvt8());
            break;
        }
        case AlbOp::Cmpv:
        case AlbOp::Subv: {
            c.movsx(result.cvt32(), b);
            c.add(result.cvt32(), ::SignExtend<16, u32>(a));
            c.shr(result.cvt32(), 31);
            c.shl(result, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~(decltype(Flags::fm)::mask | decltype(Flags::fc0)::mask));
            c.or_(FLAGS, result);

            c.movzx(result.cvt32(), b);
            c.sub(result.cvt32(), static_cast<u32>(a));
            c.mov(b.cvt32(), result.cvt32());
            c.test(b.cvt32(), ~0xFFFF);
            c.setnz(b.cvt8());
            c.shl(b.cvt8(), decltype(Flags::fc0)::position);
            c.or_(FLAGS.cvt8(), b.cvt8());
            break;
        }
        default:
            UNREACHABLE();
        }
        c.xor_(b, b);
        c.test(result.cvt16(), result.cvt16());
        c.setz(b.cvt8());
        c.shl(b, decltype(Flags::fz)::position);
        c.and_(FLAGS, ~decltype(Flags::fz)::mask);
        c.or_(FLAGS, b);
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

    static bool IsAlbConst(SttMod b) {
        switch (b.GetName()) {
        case RegName::mod0:
        case RegName::mod1:
        case RegName::mod2:
            return true;
        default:
            return false;
        }
    }

    void alb(Alb op, Imm16 a, MemImm8 b) {
        UNREACHABLE();
    }
    void alb(Alb op, Imm16 a, Rn b, StepZIDS bs) {
        UNREACHABLE();
    }
    void alb(Alb op, Imm16 a, Register b) {
        const Reg64 bv = rax;
        if (b.GetName() == RegName::p) {
            ProductToBus40(bv, Px{0});
            c.shr(bv, 16);
        } else if (b.GetName() == RegName::a0 || b.GetName() == RegName::a1) {
            throw UnimplementedException(); // weird effect;
        } else {
            RegToBus16(b.GetName(), bv);
        }
        const Reg16 result = bx;
        GenericAlb(op, a.Unsigned16(), bv.cvt16(), result);
        if (IsAlbModifying(op)) {
            switch (b.GetName()) {
            case RegName::a0:
            case RegName::a1:
                UNREACHABLE();
            // operation on accumulators doesn't go through regular bus with flag and saturation
            case RegName::a0l:
                c.mov(A[0].cvt16(), result.cvt16());
                break;
            case RegName::a1l:
                c.mov(A[1].cvt16(), result.cvt16());
                break;
            case RegName::b0l:
                c.mov(B[0].cvt16(), result.cvt16());
                break;
            case RegName::b1l:
                c.mov(B[1].cvt16(), result.cvt16());
                break;
            case RegName::a0h:
                c.shl(result.cvt32(), 16);
                c.mov(result.cvt16(), A[0].cvt16());
                c.mov(A[0].cvt32(), result.cvt32());
                break;
            case RegName::a1h:
                c.shl(result.cvt32(), 16);
                c.mov(result.cvt16(), A[1].cvt16());
                c.mov(A[1].cvt32(), result.cvt32());
                break;
            case RegName::b0h:
                c.shl(result.cvt32(), 16);
                c.mov(result.cvt16(), B[0].cvt16());
                c.mov(B[0].cvt32(), result.cvt32());
                break;
            case RegName::b1h:
                c.shl(result.cvt32(), 16);
                c.mov(result.cvt16(), B[1].cvt16());
                c.mov(B[1].cvt32(), result.cvt32());
                break;
            default:
                RegFromBus16(b.GetName(), result.cvt64()); // including RegName:p (p0h)
            }
        }
    }
    void alb_r6(Alb op, Imm16 a) {
        UNREACHABLE();
    }
    void alb(Alb op, Imm16 a, SttMod b) {
        if (IsAlbConst(b)) {
            u16 bv;
            switch (b.GetName()) {
            case RegName::mod0:
                bv = blk_key.curr.mod0.raw;
                break;
            case RegName::mod1:
                bv = blk_key.curr.mod1.raw;
                break;
            case RegName::mod2:
                bv = blk_key.curr.mod2.raw;
                break;
            default:
                UNREACHABLE();
            }
            u16 result = GenericAlbConst(op, a.Unsigned16(), bv);
            if (IsAlbModifying(op)) {
                RegFromBus16(b.GetName(), result);
            }
            return;
        }
        const Reg64 bv = rax;
        RegToBus16(b.GetName(), bv);
        const Reg16 result = bx;
        GenericAlb(op, a.Unsigned16(), bv.cvt16(), result);
        if (IsAlbModifying(op)) {
            RegFromBus16(b.GetName(), result.cvt64());
        }
    }

    void add(Ab a, Bx b) {
        UNREACHABLE();
    }
    void add(Bx a, Ax b) {
        UNREACHABLE();
    }
    void add_p1(Ax b) {
        UNREACHABLE();
    }
    void add(Px a, Bx b) {
        UNREACHABLE();
    }

    void sub(Ab a, Bx b) {
        UNREACHABLE();
    }
    void sub(Bx a, Ax b) {
        UNREACHABLE();
    }
    void sub_p1(Ax b) {
        UNREACHABLE();
    }
    void sub(Px a, Bx b) {
        UNREACHABLE();
    }

    void app(Ab c, SumBase base, bool sub_p0, bool p0_align, bool sub_p1, bool p1_align) {
        UNREACHABLE();
    }

    void add_add(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        UNREACHABLE();
    }
    void add_sub(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        UNREACHABLE();
    }
    void sub_add(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        UNREACHABLE();
    }
    void sub_sub(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        UNREACHABLE();
    }
    void add_sub_sv(ArRn1 a, ArStep1 as, Ab b) {
        UNREACHABLE();
    }
    void sub_add_sv(ArRn1 a, ArStep1 as, Ab b) {
        UNREACHABLE();
    }
    void sub_add_i_mov_j_sv(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        UNREACHABLE();
    }
    void sub_add_j_mov_i_sv(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        UNREACHABLE();
    }
    void add_sub_i_mov_j(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        UNREACHABLE();
    }
    void add_sub_j_mov_i(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        UNREACHABLE();
    }

    template <typename T>
    constexpr size_t BitSize() {
        if constexpr (std::is_same_v<T, Reg64>) {
            return 64;
        } else if constexpr (std::is_same_v<T, Reg32>) {
            return 32;
        } else if constexpr (std::is_same_v<T, Reg16>) {
            return 16;
        } else {
            return sizeof(std::make_unsigned_t<T>) * CHAR_BIT;
        }
    }

    template <typename T>
    void SignExtend(T value, u32 bit_count) {
        if (bit_count == 16) {
            c.movsx(value, value.cvt16());
            return;
        }
        if (bit_count == 32) {
            if constexpr (std::is_same_v<T, Xbyak::Reg64>) {
                c.movsxd(value, value.cvt32());
                return;
            }
            UNREACHABLE();
        }
        c.shl(value, BitSize<T>() - bit_count);
        c.sar(value, BitSize<T>() - bit_count);
    }

    void ShiftBus40(Reg64 value, u16 sv, RegName dest) {
        c.xor_(rcx, rcx);
        c.mov(rbx, 0xFF'FFFF'FFFF);
        c.and_(value, rbx);
        const Reg64 original_sign = rbx;
        c.mov(original_sign, value);
        c.shr(original_sign, 39);
        if ((sv >> 15) == 0) {
            // left shift
            if (sv >= 40) {
                if (blk_key.curr.mod0.s == 0) {
                    // regs.fv = value != 0;
                    c.and_(FLAGS, ~decltype(Flags::fv)::mask); // clear fv
                    c.test(value, value);
                    c.setne(cx); // u32 mask = (value != 0) ? 1 : 0
                    c.shl(cx, decltype(Flags::fv)::position); // mask <<= fv_pos
                    c.or_(FLAGS, cx); // flags |= mask
                    // if (regs.fv) {
                    //    regs.fvl = 1;
                    // }
                    static_assert(decltype(Flags::fv)::position - decltype(Flags::fvl)::position == 3);
                    // mask is either 0 or 1. If it was 0, then the or_ wont have any effect.
                    // If it was 1, it will set fvl, which is what we want.
                    c.shr(cx, 3); // mask >>= 3;
                    c.or_(FLAGS, cx);
                }
                c.xor_(value, value); // value = 0;
                c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // regs.fc0 = 0;
            } else {
                if (blk_key.curr.mod0.s == 0) {
                    // TODO: This can probably be done better
                    c.mov(rcx, value);
                    c.mov(rdx, value);
                    SignExtend(rcx, 40);
                    SignExtend(rdx, 40 - sv);
                    // regs.fv = SignExtend<40>(value) != SignExtend(value, 40 - sv);
                    c.and_(FLAGS, ~decltype(Flags::fv)::mask); // clear fv
                    c.cmp(rcx, rdx);
                    c.setne(cl); // u32 mask = (SignExtend<40>(value) != SignExtend(value, 40 - sv) ? 1 : 0
                    c.shl(cl, decltype(Flags::fv)::position); // mask <<= fv_pos
                    c.or_(FLAGS, cl); // flags |= mask
                    // if (regs.fv) {
                    //     regs.fvl = 1;
                    // }
                    c.shr(cl, 3); // mask >>= 3;
                    c.or_(FLAGS, cl);
                }
                c.shl(value, sv); // value <<= sv;
                // regs.fc0 = (value & ((u64)1 << 40)) != 0;
                c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
                c.bt(value, 40);
                c.setc(cl); // u32 mask = (value & ((u64)1 << 40));
                c.shl(cl, decltype(Flags::fc0)::position); // mask <<= fc0_pos;
                c.or_(FLAGS, cl);
            }
        } else {
            // right shift
            u16 nsv = ~sv + 1;
            if (nsv >= 40) {
                if (blk_key.curr.mod0.s == 0) {
                    UNREACHABLE();
                    // regs.fc0 = (value >> 39) & 1;
                    // value = regs.fc0 ? 0xFF'FFFF'FFFF : 0;
                } else {
                    c.xor_(value, value); // value = 0;
                    c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // regs.fc0 = 0;
                }
            } else {
                // regs.fc0 = (value & ((u64)1 << (nsv - 1))) != 0;
                c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
                c.bt(value, nsv - 1);
                c.setc(cl); // u32 mask = (value & ((u64)1 << (nsv - 1)));
                c.shl(cx, decltype(Flags::fc0)::position); // mask <<= fc0_pos;
                c.or_(FLAGS, cx);
                c.shr(value, nsv); // value >>= nsv;
                if (blk_key.curr.mod0.s == 0) {
                    SignExtend(value, 40 - nsv);
                }
            }

            if (blk_key.curr.mod0.s == 0) {
                c.and_(FLAGS, ~decltype(Flags::fv)::mask); // regs.fv = 0;
            }
        }

        SignExtend(value, 40);
        SetAccFlag(value);
        if (blk_key.curr.mod0.s == 0 && blk_key.curr.mod0.sata == 0) {
            UNREACHABLE();
            /*if (regs.fv || SignExtend<32>(value) != value) {
                regs.flm = 1;
                value = original_sign == 1 ? 0xFFFF'FFFF'8000'0000 : 0x7FFF'FFFF;
            }*/
        }
        SetAcc(dest, value);
    }

    static void PrintValue(u64 value) {
        printf("jit: 0x%lx\n", value);
    }

    void EmitPrint(Reg64 value) {
        c.push(rax);
        c.push(rbx);
        ABI_PushRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.mov(ABI_PARAM1, value);
        CallFarFunction(c, PrintValue);
        ABI_PopRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.pop(rbx);
        c.pop(rax);
    }

    void AddSub(Reg64 a, Reg64 b, Reg64 result, bool sub) {
        c.mov(rsi, 0xFF'FFFF'FFFF);
        c.and_(a, rsi);
        c.and_(b, rsi);
        c.mov(result, a);
        if (sub) {
            c.sub(result, b);
        } else {
            c.add(result, b);
        }
        c.and_(FLAGS, ~(decltype(Flags::fc0)::mask | decltype(Flags::fv)::mask)); // clear fc0, fv
        c.xor_(rsi, rsi);
        c.bt(result, 40);
        c.setc(rsi.cvt8());
        c.shl(rsi, decltype(Flags::fc0)::position);
        c.or_(FLAGS, rsi.cvt16()); // regs.fc0 = (result >> 40) & 1;
        if (sub) {
            c.not_(b);
        }
        c.xor_(b, a);
        c.not_(b);
        c.xor_(a, result);
        c.and_(a, b);
        c.bt(a, 39); // ((~(a ^ b) & (a ^ result)) >> 39) & 1;
        c.setc(rsi.cvt8());
        c.shl(rsi, decltype(Flags::fv)::position);
        c.or_(FLAGS, rsi.cvt16());
        // if (regs.fv) {
        //     regs.fvl = 1;
        // }
        static_assert(decltype(Flags::fv)::position - decltype(Flags::fvl)::position == 3);
        c.shr(rsi, 3);
        c.or_(FLAGS, rsi.cvt16());
        SignExtend(result, 40);
    }

    void Moda(ModaOp op, RegName a, Cond cond) {
        ConditionPass(cond, [&] {
            Xbyak::Reg64 acc = rax;
            switch (op) {
            case ModaOp::Shr: {
                GetAcc(acc, a);
                ShiftBus40(acc, 0xFFFF, a);
                break;
            }
            case ModaOp::Shr4: {
                GetAcc(acc, a);
                ShiftBus40(acc, 0xFFFC, a);
                break;
            }
            case ModaOp::Shl: {
                GetAcc(acc, a);
                ShiftBus40(acc, 1, a);
                break;
            }
            case ModaOp::Shl4: {
                GetAcc(acc, a);
                ShiftBus40(acc, 4, a);
                break;
            }
            case ModaOp::Ror: {
                GetAcc(acc, a);
                c.mov(rbx, 0xFF'FFFF'FFFF); // Maybe use BZHI?
                c.and_(acc, rbx);
                c.btr(FLAGS, decltype(Flags::fc0)::position); // test and clear fc0
                c.setc(rbx); // u16 old_fc = regs.fc0;
                c.shl(rbx, 40);
                c.or_(acc, rbx); // value |= (u64)old_fc << 40;
                c.bt(acc, 1); // u32 mask = value & 1;
                c.setc(rbx); // mask <<= fc0_pos;
                c.shl(rbx, decltype(Flags::fc0)::position);
                c.or_(FLAGS, rbx); // flags |= mask;
                c.shr(acc, 1); // value >>= 1;
                SignExtend(acc, 40);
                SetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Rol: {
                GetAcc(acc, a);
                c.btr(FLAGS, decltype(Flags::fc0)::position); // test and clear fc0
                c.setc(rbx); // u16 old_fc = regs.fc0;
                c.shl(acc, 1); // value <<= 1;
                c.or_(acc, rbx); // value |= old_fc;
                c.bt(acc, 40);
                c.setc(rbx);
                c.shl(rbx, decltype(Flags::fc0)::position);
                c.or_(FLAGS, rbx); // regs.fc0 = (value >> 40) & 1;
                SignExtend(acc, 40);
                SetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Clr: {
                SatAndSetAccAndFlag(a, 0ULL);
                break;
            }
            case ModaOp::Not: {
                GetAcc(acc, a);
                c.not_(acc);
                SetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Neg: {
                /*GetAcc(acc, a);
                regs.fc0 = value != 0;                    // ?
                regs.fv = value == 0xFFFF'FF80'0000'0000; // ?
                if (regs.fv)
                    regs.fvl = 1;
                u64 result = SignExtend<40, u64>(~GetAcc(a) + 1);
                SatAndSetAccAndFlag(a, result);
                break;*/
                UNREACHABLE();
            }
            case ModaOp::Rnd: {
                GetAcc(acc, a);
                c.mov(rbx, 0x8000);
                AddSub(acc, rbx, rcx, false);
                SatAndSetAccAndFlag(a, rcx);
                break;
            }
            case ModaOp::Pacr: {
                /*u64 value = ProductToBus40(Px{0});
                u64 result = AddSub(value, 0x8000, false);
                SatAndSetAccAndFlag(a, result);
                break;*/
                UNREACHABLE();
            }
            case ModaOp::Clrr: {
                SatAndSetAccAndFlag(a, 0x8000ULL);
                break;
            }
            case ModaOp::Inc: {
                GetAcc(acc, a);
                c.mov(rbx, 1);
                AddSub(acc, rbx, rcx, false);
                SatAndSetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Dec: {
                GetAcc(acc, a);
                c.mov(rbx, 1);
                AddSub(acc, rbx, rcx, true);
                SatAndSetAccAndFlag(a, rcx);
                break;
            }
            case ModaOp::Copy: {
                // note: bX doesn't support
                GetAcc(acc, a == RegName::a0 ? RegName::a1 : RegName::a0);
                SatAndSetAccAndFlag(a, acc);
                break;
            }
            default:
                UNREACHABLE();
            }
        });
    }

    void moda4(Moda4 op, Ax a, Cond cond) {
        Moda(op.GetName(), a.GetName(), cond);
    }

    void moda3(Moda3 op, Bx a, Cond cond) {
        UNREACHABLE();
    }

    void pacr1(Ax a) {
        UNREACHABLE();
    }

    void clr(Ab a, Ab b) {
        UNREACHABLE();
    }
    void clrr(Ab a, Ab b) {
        UNREACHABLE();
    }

    void bkrep(Imm8 a, Address16 addr) {
        UNREACHABLE();
    }
    void bkrep(Register a, Address18_16 addr_low, Address18_2 addr_high) {
        UNREACHABLE();
    }
    void bkrep_r6(Address18_16 addr_low, Address18_2 addr_high) {
        UNREACHABLE();
    }

    void bkreprst(ArRn2 a) {
        UNREACHABLE();
    }
    void bkreprst_memsp() {
        UNREACHABLE();
    }
    void bkrepsto(ArRn2 a) {
        UNREACHABLE();
    }
    void bkrepsto_memsp() {
        UNREACHABLE();
    }

    void banke(BankFlags flags) {
        UNREACHABLE();
    }
    void bankr() {
         UNREACHABLE();
    }
    void bankr(Ar a) {
        UNREACHABLE();
    }
    void bankr(Ar a, Arp b) {
        UNREACHABLE();
    }
    void bankr(Arp a) {
        UNREACHABLE();
    }

    void bitrev(Rn a) {
        UNREACHABLE();
    }
    void bitrev_dbrv(Rn a) {
        UNREACHABLE();
    }
    void bitrev_ebrv(Rn a) {
         UNREACHABLE();
    }

    void br(Address18_16 addr_low, Address18_2 addr_high, Cond cond) {
        ConditionPass(cond, [&] {
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], Address32(addr_low, addr_high));
        });
        compiling = false;
    }

    void brr(RelAddr7 addr, Cond cond) {
        ConditionPass(cond, [&] {
            // note: pc is the address of the NEXT instruction
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc + addr.Relative32());
            compiling = false;
            if (addr.Relative32() == 0xFFFFFFFF) {
                c.mov(dword[REGS + offsetof(JitRegisters, idle)], true);
            }
        });
    }

    void break_() {
         UNREACHABLE();
    }

    void call(Address18_16 addr_low, Address18_2 addr_high, Cond cond) {
         ConditionPass(cond, [&] {
            EmitPushPC();
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], Address32(addr_low, addr_high));
         });
         compiling = false;
    }
    void calla(Axl a) {
        UNREACHABLE();
    }
    void calla(Ax a) {
        UNREACHABLE();
    }
    void callr(RelAddr7 addr, Cond cond) {
        UNREACHABLE();
    }

    void cntx_s() {
        regs.ShadowStore(c);
        regs.ShadowSwap(c);
        std::swap(blk_key.curr, blk_key.shadow);
        // if (!regs.crep) {
        //     regs.repcs = regs.repc;
        // }
        c.mov(ax, word[REGS + offsetof(JitRegisters, repcs)]);
        c.test(word[REGS + offsetof(JitRegisters, crep)], 0x1);
        c.cmovz(ax, word[REGS + offsetof(JitRegisters, repc)]);
        c.mov(word[REGS + offsetof(JitRegisters, repcs)], ax);

        Xbyak::Label ccnta_label, end_label;
        c.test(word[REGS + offsetof(JitRegisters, ccnta)], 0x1);
        c.jnz(ccnta_label);
        c.mov(qword[REGS + offsetof(JitRegisters, a1s)], A[1]);
        c.mov(qword[REGS + offsetof(JitRegisters, b1s)], B[1]);
        c.jmp(end_label);
        c.L(ccnta_label);
        c.mov(rax, B[1]);
        c.mov(B[1], A[1]);
        SetAccAndFlag(RegName::a1, rax); // Flag set on b1->a1
        c.L(end_label);
    }
    void cntx_r() {
        regs.ShadowRestore(c);
        regs.ShadowSwap(c);
        std::swap(blk_key.curr, blk_key.shadow);

        // if (!regs.crep) {
        //     regs.repc = regs.repcs;
        // }
        c.mov(ax, word[REGS + offsetof(JitRegisters, repc)]);
        c.test(word[REGS + offsetof(JitRegisters, crep)], 0x1);
        c.cmovz(ax, word[REGS + offsetof(JitRegisters, repcs)]);
        c.mov(word[REGS + offsetof(JitRegisters, repc)], ax);

        Xbyak::Label ccnta_label, end_label;
        c.test(word[REGS + offsetof(JitRegisters, ccnta)], 0x1);
        c.jnz(ccnta_label);
        c.mov(A[1], qword[REGS + offsetof(JitRegisters, a1s)]);
        c.mov(B[1], qword[REGS + offsetof(JitRegisters, b1s)]);
        c.jmp(end_label);
        c.L(ccnta_label);
        c.xchg(A[1], B[1]);
        c.L(end_label);
    }

    void ret(Cond c) {
        ConditionPass(c, [&] {
            EmitPopPC();
        });
        compiling = false;
    }
    void retd() {
        UNREACHABLE();
    }
    void reti(Cond c) {
        UNREACHABLE();
    }
    void retic(Cond c) {
        UNREACHABLE();
    }
    void retid() {
        UNREACHABLE();
    }
    void retidc() {
        UNREACHABLE();
    }
    void rets(Imm8 a) {
        UNREACHABLE();
    }

    void load_ps(Imm2 a) {
        const u16 mask = (a.Unsigned16() & 3) << decltype(Mod0::ps0)::position;
        c.mov(ax, word[REGS + offsetof(JitRegisters, mod0)]);
        c.and_(ax, ~decltype(Mod0::ps0)::mask);
        c.or_(ax, mask);
        c.mov(word[REGS + offsetof(JitRegisters, mod0)], ax);
        blk_key.curr.mod0.ps0.Assign(a.Unsigned16());
    }
    void load_stepi(Imm7s a) {
        // Although this is signed, we still only store the lower 7 bits
        const u8 stepi = static_cast<u8>(a.Signed16() & 0x7F);
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgi)]);
        c.and_(ax, ~decltype(Cfg::step)::mask);
        c.or_(ax, stepi);
        c.mov(word[REGS + offsetof(JitRegisters, cfgi)], ax);
        blk_key.curr.cfgi.step.Assign(stepi);
    }
    void load_stepj(Imm7s a) {
        const u8 stepj = static_cast<u8>(a.Signed16() & 0x7F);
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgj)]);
        c.and_(ax, ~decltype(Cfg::step)::mask);
        c.or_(ax, stepj);
        c.mov(word[REGS + offsetof(JitRegisters, cfgj)], ax);
        blk_key.curr.cfgj.step.Assign(stepj);
    }
    void load_page(Imm8 a) {
        const u8 page = static_cast<u8>(a.Unsigned16());
        c.mov(byte[REGS + offsetof(JitRegisters, mod1)], page);
        blk_key.curr.mod1.page.Assign(page);
    }
    void load_modi(Imm9 a) {
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgi)]);
        c.and_(ax, ~decltype(Cfg::mod)::mask);
        c.or_(ax, a.Unsigned16() << decltype(Cfg::mod)::position);
        c.mov(word[REGS + offsetof(JitRegisters, cfgi)], ax);
        blk_key.curr.cfgi.mod.Assign(a.Unsigned16());
    }
    void load_modj(Imm9 a) {
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgj)]);
        c.and_(ax, ~decltype(Cfg::mod)::mask);
        c.or_(ax, a.Unsigned16() << decltype(Cfg::mod)::position);
        c.mov(word[REGS + offsetof(JitRegisters, cfgj)], ax);
        blk_key.curr.cfgj.mod.Assign(a.Unsigned16());
    }
    void load_movpd(Imm2 a) {
        c.mov(word[REGS + offsetof(JitRegisters, pcmhi)], a.Unsigned16());
    }
    void load_ps01(Imm4 a) {
        const u16 mask = (a.Unsigned16() & 0xF) << decltype(Mod0::ps0)::position;
        c.mov(ax, word[REGS + offsetof(JitRegisters, mod0)]);
        c.and_(ax, ~(decltype(Mod0::ps0)::mask | decltype(Mod0::ps1)::mask));
        c.or_(ax, mask);
        c.mov(word[REGS + offsetof(JitRegisters, mod0)], ax);
        blk_key.curr.mod0.ps0.Assign(a.Unsigned16() & 3);
        blk_key.curr.mod0.ps1.Assign(a.Unsigned16() >> 2);
    }

    void push(Imm16 a) {
        UNREACHABLE();
    }
    void push(Register a) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value, true);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp);
    }
    void push(Abe a) {
        UNREACHABLE();
    }
    void push(ArArpSttMod a) {
        UNREACHABLE();
    }
    void push_prpage() {
        UNREACHABLE();
    }
    void push(Px a) {
        UNREACHABLE();
    }
    void push_r6() {
        UNREACHABLE();
    }
    void push_repc() {
        UNREACHABLE();
    }
    void push_x0() {
        UNREACHABLE();
    }
    void push_x1() {
        UNREACHABLE();
    }
    void push_y1() {
        UNREACHABLE();
    }
    void pusha(Ax a) {
        UNREACHABLE();
    }
    void pusha(Bx a) {
        UNREACHABLE();
    }

    void pop(Register a) {
        UNREACHABLE();
    }
    void pop(Abe a) {
        UNREACHABLE();
    }
    void pop(ArArpSttMod a) {
        UNREACHABLE();
    }
    void pop(Bx a) {
        UNREACHABLE();
    }
    void pop_prpage() {
        UNREACHABLE();
    }
    void pop(Px a) {
        UNREACHABLE();
    }
    void pop_r6() {
        UNREACHABLE();
    }
    void pop_repc() {
        UNREACHABLE();
    }
    void pop_x0() {
        UNREACHABLE();
    }
    void pop_x1() {
        UNREACHABLE();
    }
    void pop_y1() {
        UNREACHABLE();
    }
    void popa(Ab a) {
        UNREACHABLE();
    }

    void rep(Imm8 a) {
        UNREACHABLE();
    }
    void rep(Register a) {
        UNREACHABLE();
    }
    void rep_r6() {
        UNREACHABLE();
    }

    void shfc(Ab a, Ab b, Cond cond) {
        UNREACHABLE();
    }
    void shfi(Ab a, Ab b, Imm6s s) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        u16 sv = s.Signed16();
        ShiftBus40(value, sv, b.GetName());
    }

    void tst4b(ArRn2 b, ArStep2 bs) {
        UNREACHABLE();
    }
    void tst4b(ArRn2 b, ArStep2 bs, Ax c) {
        UNREACHABLE();
    }
    void tstb(MemImm8 a, Imm4 b) {
        UNREACHABLE();
    }
    void tstb(Rn a, StepZIDS as, Imm4 b) {
        UNREACHABLE();
    }
    void tstb(Register a, Imm4 b) {
        UNREACHABLE();
    }
    void tstb_r6(Imm4 b) {
        UNREACHABLE();
    }
    void tstb(SttMod a, Imm16 b) {
        UNREACHABLE();
    }

    void and_(Ab a, Ab b, Ax c) {
        UNREACHABLE();
    }

    void dint() {
        c.mov(word[REGS + offsetof(JitRegisters, ie)], 0);
    }
    void eint() {
        c.mov(word[REGS + offsetof(JitRegisters, ie)], 1);
    }

    void mul(Mul3 op, Rn y, StepZIDS ys, Imm16 x, Ax a) {
        UNREACHABLE();
    }
    void mul_y0(Mul3 op, Rn x, StepZIDS xs, Ax a) {
        UNREACHABLE();
    }
    void mul_y0(Mul3 op, Register x, Ax a) {
        UNREACHABLE();
    }
    void mul(Mul3 op, R45 y, StepZIDS ys, R0123 x, StepZIDS xs, Ax a) {
        UNREACHABLE();
    }
    void mul_y0_r6(Mul3 op, Ax a) {
        UNREACHABLE();
    }
    void mul_y0(Mul2 op, MemImm8 x, Ax a) {
        UNREACHABLE();
    }

    void mpyi(Imm8s x) {
        UNREACHABLE();
    }

    void msu(R45 y, StepZIDS ys, R0123 x, StepZIDS xs, Ax a) {
        UNREACHABLE();
    }
    void msu(Rn y, StepZIDS ys, Imm16 x, Ax a) {
        UNREACHABLE();
    }
    void msusu(ArRn2 x, ArStep2 xs, Ax a) {
        UNREACHABLE();
    }
    void mac_x1to0(Ax a) {
        UNREACHABLE();
    }
    void mac1(ArpRn1 xy, ArpStep1 xis, ArpStep1 yjs, Ax a) {
        UNREACHABLE();
    }

    void modr(Rn a, StepZIDS as) {
         UNREACHABLE();
    }
    void modr_dmod(Rn a, StepZIDS as) {
        UNREACHABLE();
    }
    void modr_i2(Rn a) {
        UNREACHABLE();
    }
    void modr_i2_dmod(Rn a) {
        UNREACHABLE();
    }
    void modr_d2(Rn a) {
        UNREACHABLE();
    }
    void modr_d2_dmod(Rn a) {
        UNREACHABLE();
    }
    void modr_eemod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        UNREACHABLE();
    }
    void modr_edmod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        UNREACHABLE();
    }
    void modr_demod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        UNREACHABLE();
    }
    void modr_ddmod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        UNREACHABLE();
    }

    void movd(R0123 a, StepZIDS as, R45 b, StepZIDS bs) {
        UNREACHABLE();
    }
    void movp(Axl a, Register b) {
        UNREACHABLE();
    }
    void movp(Ax a, Register b) {
        UNREACHABLE();
    }
    void movp(Rn a, StepZIDS as, R0123 b, StepZIDS bs) {
        UNREACHABLE();
    }
    void movpdw(Ax a) {
        UNREACHABLE();
    }

    void mov(Ab a, Ab b) {
        UNREACHABLE();
    }
    void mov_dvm(Abl a) {
        UNREACHABLE();
    }
    void mov_x0(Abl a) {
        UNREACHABLE();
    }
    void mov_x1(Abl a) {
        UNREACHABLE();
    }
    void mov_y1(Abl a) {
        UNREACHABLE();
    }

    static void MemDataWriteThunk(void* mem_ptr, u16 address, u16 data) {
        reinterpret_cast<MemoryInterface*>(mem_ptr)->DataWrite(address, data);
    }

    void StoreToMemory(MemImm8 addr, Reg64 value) {
        StoreToMemory(addr.Unsigned16() + (blk_key.curr.mod1.page << 8), value);
    }

    template <typename T1, typename T2>
    void StoreToMemory(T1 addr, T2 value) {
        // TODO: Non MMIO writes can be performed inside the JIT.
        ABI_PushRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(&mem));
        c.mov(ABI_PARAM2, addr);
        c.mov(ABI_PARAM3, value);
        CallFarFunction(c, MemDataWriteThunk);
        ABI_PopRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
    }

    void mov(Ablh a, MemImm8 b) {
        const Reg64 value16 = rax;
        RegToBus16(a.GetName(), value16, true);
        StoreToMemory(b, value16);
    }
    void mov(Axl a, MemImm16 b) {
         UNREACHABLE();
    }
    void mov(Axl a, MemR7Imm16 b) {
        UNREACHABLE();
    }
    void mov(Axl a, MemR7Imm7s b) {
        UNREACHABLE();
    }

    void mov(MemImm16 a, Ax b) {
        UNREACHABLE();
    }
    void mov(MemImm8 a, Ab b) {
        UNREACHABLE();
    }
    void mov(MemImm8 a, Ablh b) {
        UNREACHABLE();
    }
    void mov_eu(MemImm8 a, Axh b) {
        UNREACHABLE();
    }
    void mov(MemImm8 a, RnOld b) {
        UNREACHABLE();
    }
    void mov_sv(MemImm8 a) {
        UNREACHABLE();
    }
    void mov_dvm_to(Ab b) {
        UNREACHABLE();
    }
    void mov_icr_to(Ab b) {
        UNREACHABLE();
    }
    void mov(Imm16 a, Bx b) {
        UNREACHABLE();
    }
    void mov(Imm16 a, Register b) {
        const u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_icr(Imm5 a) {
        UNREACHABLE();
    }
    void mov(Imm8s a, Axh b) {
        UNREACHABLE();
    }
    void mov(Imm8s a, RnOld b) {
        UNREACHABLE();
    }
    void mov_sv(Imm8s a) {
        UNREACHABLE();
    }
    void mov(Imm8 a, Axl b) {
        UNREACHABLE();
    }
    void mov(MemR7Imm16 a, Ax b) {
        UNREACHABLE();
    }
    void mov(MemR7Imm7s a, Ax b) {
        UNREACHABLE();
    }
    void mov(Rn a, StepZIDS as, Bx b) {
        UNREACHABLE();
    }
    void mov(Rn a, StepZIDS as, Register b) {
        UNREACHABLE();
    }
    void mov_memsp_to(Register b) {
        UNREACHABLE();
    }
    void mov_mixp_to(Register b) {
        UNREACHABLE();
    }
    void mov(RnOld a, MemImm8 b) {
        UNREACHABLE();
    }
    void mov_icr(Register a) {
        UNREACHABLE();
    }
    void mov_mixp(Register a) {
        UNREACHABLE();
    }
    void mov(Register a, Rn b, StepZIDS bs) {
        UNREACHABLE();
    }
    void mov(Register a, Bx b) {
        UNREACHABLE();
    }
    void mov(Register a, Register b) {
        // a = a0 or a1 is overrided
        if (a.GetName() == RegName::p) {
            // b loses its typical meaning in this case
            const RegName b_name = b.GetNameForMovFromP();
            const Reg64 value = rax;
            ProductToBus40(value, Px{0});
            SatAndSetAccAndFlag(b_name, value);
        } else if (a.GetName() == RegName::pc) {
            if (b.GetName() == RegName::a0 || b.GetName() == RegName::a1) {
                SatAndSetAccAndFlag(b.GetName(), static_cast<u64>(regs.pc));
            } else {
                RegFromBus16(b.GetName(), regs.pc & 0xFFFF);
            }
        } else {
            const Reg64 value = rax;
            RegToBus16(a.GetName(), value, true);
            RegFromBus16(b.GetName(), value);
        }
    }
    void mov_repc_to(Ab b) {
        UNREACHABLE();
    }
    void mov_sv_to(MemImm8 b) {
        UNREACHABLE();
    }
    void mov_x0_to(Ab b) {
        UNREACHABLE();
    }
    void mov_x1_to(Ab b) {
        UNREACHABLE();
    }
    void mov_y1_to(Ab b) {
        UNREACHABLE();
    }
    void mov(Imm16 a, ArArp b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_r6(Imm16 a) {
        UNREACHABLE();
    }
    void mov_repc(Imm16 a) {
        UNREACHABLE();
    }
    void mov_stepi0(Imm16 a) {
        u16 value = a.Unsigned16();
        c.mov(word[REGS + offsetof(JitRegisters, stepi0)], value);
        blk_key.curr.stepi0 = value;
    }
    void mov_stepj0(Imm16 a) {
        u16 value = a.Unsigned16();
        c.mov(word[REGS + offsetof(JitRegisters, stepj0)], value);
        blk_key.curr.stepj0 = value;
    }
    void mov(Imm16 a, SttMod b) {
        const u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_prpage(Imm4 a) {
        UNREACHABLE();
    }

    void mov_a0h_stepi0() {
        UNREACHABLE();
    }
    void mov_a0h_stepj0() {
        UNREACHABLE();
    }
    void mov_stepi0_a0h() {
        UNREACHABLE();
    }
    void mov_stepj0_a0h() {
        UNREACHABLE();
    }

    void mov_prpage(Abl a) {
        UNREACHABLE();
    }
    void mov_repc(Abl a) {
        UNREACHABLE();
    }
    void mov(Abl a, ArArp b) {
        UNREACHABLE();
    }
    void mov(Abl a, SttMod b) {
        UNREACHABLE();
    }

    void mov_prpage_to(Abl b) {
        UNREACHABLE();
    }
    void mov_repc_to(Abl b) {
        UNREACHABLE();
    }
    void mov(ArArp a, Abl b) {
        UNREACHABLE();
    }
    void mov(SttMod a, Abl b) {
        UNREACHABLE();
    }

    void mov_repc_to(ArRn1 b, ArStep1 bs) {
         UNREACHABLE();
    }
    void mov(ArArp a, ArRn1 b, ArStep1 bs) {
        UNREACHABLE();
    }
    void mov(SttMod a, ArRn1 b, ArStep1 bs) {
        UNREACHABLE();
    }

    void mov_repc(ArRn1 a, ArStep1 as) {
        UNREACHABLE();
    }
    void mov(ArRn1 a, ArStep1 as, ArArp b) {
        UNREACHABLE();
    }
    void mov(ArRn1 a, ArStep1 as, SttMod b) {
        UNREACHABLE();
    }

    void mov_repc_to(MemR7Imm16 b) {
        UNREACHABLE();
    }
    void mov(ArArpSttMod a, MemR7Imm16 b) {
        UNREACHABLE();
    }

    void mov_repc(MemR7Imm16 a) {
        UNREACHABLE();
    }
    void mov(MemR7Imm16 a, ArArpSttMod b) {
        UNREACHABLE();
    }

    void mov_pc(Ax a) {
        UNREACHABLE();
    }
    void mov_pc(Bx a) {
        UNREACHABLE();
    }

    void mov_mixp_to(Bx b) {
        UNREACHABLE();
    }
    void mov_mixp_r6() {
        UNREACHABLE();
    }
    void mov_p0h_to(Bx b) {
        UNREACHABLE();
    }
    void mov_p0h_r6() {
        UNREACHABLE();
    }
    void mov_p0h_to(Register b) {
        UNREACHABLE();
    }
    void mov_p0(Ab a) {
        UNREACHABLE();
    }
    void mov_p1_to(Ab b) {
        UNREACHABLE();
    }

    void mov2(Px a, ArRn2 b, ArStep2 bs) {
        UNREACHABLE();
    }
    void mov2s(Px a, ArRn2 b, ArStep2 bs) {
        UNREACHABLE();
    }
    void mov2(ArRn2 a, ArStep2 as, Px b) {
        UNREACHABLE();
    }
    void mova(Ab a, ArRn2 b, ArStep2 bs) {
        UNREACHABLE();
    }
    void mova(ArRn2 a, ArStep2 as, Ab b) {
        UNREACHABLE();
    }

    void mov_r6_to(Bx b) {
        UNREACHABLE();
    }
    void mov_r6_mixp() {
        UNREACHABLE();
    }
    void mov_r6_to(Register b) {
        UNREACHABLE();
    }
    void mov_r6(Register a) {
        UNREACHABLE();
    }
    void mov_memsp_r6() {
        UNREACHABLE();
    }
    void mov_r6_to(Rn b, StepZIDS bs) {
        UNREACHABLE();
    }
    void mov_r6(Rn a, StepZIDS as) {
        UNREACHABLE();
    }

    void mov2_axh_m_y0_m(Axh a, ArRn2 b, ArStep2 bs) {
        UNREACHABLE();
    }

    void mov2_ax_mij(Ab a, ArpRn1 b, ArpStep1 bsi, ArpStep1 bsj) {
        UNREACHABLE();
    }
    void mov2_ax_mji(Ab a, ArpRn1 b, ArpStep1 bsi, ArpStep1 bsj) {
        UNREACHABLE();
    }
    void mov2_mij_ax(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        UNREACHABLE();
    }
    void mov2_mji_ax(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        UNREACHABLE();
    }
    void mov2_abh_m(Abh ax, Abh ay, ArRn1 b, ArStep1 bs) {
        UNREACHABLE();
    }
    void exchange_iaj(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        UNREACHABLE();
    }
    void exchange_riaj(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        UNREACHABLE();
    }
    void exchange_jai(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        UNREACHABLE();
    }
    void exchange_rjai(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        UNREACHABLE();
    }

    void movs(MemImm8 a, Ab b) {
        UNREACHABLE();
    }
    void movs(Rn a, StepZIDS as, Ab b) {
        UNREACHABLE();
    }
    void movs(Register a, Ab b) {
        UNREACHABLE();
    }
    void movs_r6_to(Ax b) {
        UNREACHABLE();
    }
    void movsi(RnOld a, Ab b, Imm5s s) {
        UNREACHABLE();
    }

    void movr(ArRn2 a, ArStep2 as, Abh b) {
        UNREACHABLE();
    }
    void movr(Rn a, StepZIDS as, Ax b) {
        UNREACHABLE();
    }
    void movr(Register a, Ax b) {
        UNREACHABLE();
    }
    void movr(Bx a, Ax b) {
        UNREACHABLE();
    }
    void movr_r6_to(Ax b) {
        UNREACHABLE();
    }

    void exp(Bx a) {
        UNREACHABLE();
    }
    void exp(Bx a, Ax b) {
        UNREACHABLE();
    }
    void exp(Rn a, StepZIDS as) {
        UNREACHABLE();
    }
    void exp(Rn a, StepZIDS as, Ax b) {
        UNREACHABLE();
    }
    void exp(Register a) {
        UNREACHABLE();
    }
    void exp(Register a, Ax b) {
        UNREACHABLE();
    }
    void exp_r6() {
        UNREACHABLE();
    }
    void exp_r6(Ax b) {
        UNREACHABLE();
    }

    void lim(Ax a, Ax b) {
        UNREACHABLE();
    }

    void vtrclr0() {
        UNREACHABLE();
    }
    void vtrclr1() {
        UNREACHABLE();
    }
    void vtrclr() {
        UNREACHABLE();
    }
    void vtrmov0(Axl a) {
        UNREACHABLE();
    }
    void vtrmov1(Axl a) {
        UNREACHABLE();
    }
    void vtrmov(Axl a) {
        UNREACHABLE();
    }
    void vtrshr() {
        UNREACHABLE();
    }

    void clrp0() {
        UNREACHABLE();
    }
    void clrp1() {
        UNREACHABLE();
    }
    void clrp() {
        UNREACHABLE();
    }

    void max_ge(Ax a, StepZIDS bs) {
        UNREACHABLE();
    }
    void max_gt(Ax a, StepZIDS bs) {
        UNREACHABLE();
    }
    void min_le(Ax a, StepZIDS bs) {
        UNREACHABLE();
    }
    void min_lt(Ax a, StepZIDS bs) {
        UNREACHABLE();
    }

    void max_ge_r0(Ax a, StepZIDS bs) {
        UNREACHABLE();
    }
    void max_gt_r0(Ax a, StepZIDS bs) {
        UNREACHABLE();
    }
    void min_le_r0(Ax a, StepZIDS bs) {
        UNREACHABLE();
    }
    void min_lt_r0(Ax a, StepZIDS bs) {
        UNREACHABLE();
    }

    void divs(MemImm8 a, Ax b) {
        UNREACHABLE();
    }

    void sqr_sqr_add3(Ab a, Ab b) {
        UNREACHABLE();
    }

    void sqr_sqr_add3(ArRn2 a, ArStep2 as, Ab b) {
        UNREACHABLE();
    }

    void sqr_mpysu_add3a(Ab a, Ab b) {
        UNREACHABLE();
    }

    void cmp(Ax a, Bx b) {
        UNREACHABLE();
    }
    void cmp_b0_b1() {
        UNREACHABLE();
    }
    void cmp_b1_b0() {
        UNREACHABLE();
    }
    void cmp(Bx a, Ax b) {
        UNREACHABLE();
    }
    void cmp_p1_to(Ax b) {
        UNREACHABLE();
    }

    void max2_vtr(Ax a) {
        UNREACHABLE();
    }
    void min2_vtr(Ax a) {
        UNREACHABLE();
    }
    void max2_vtr(Ax a, Bx b) {
        UNREACHABLE();
    }
    void min2_vtr(Ax a, Bx b) {
        UNREACHABLE();
    }
    void max2_vtr_movl(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        UNREACHABLE();
    }
    void max2_vtr_movh(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        UNREACHABLE();
    }
    void max2_vtr_movl(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        UNREACHABLE();
    }
    void max2_vtr_movh(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        UNREACHABLE();
    }
    void min2_vtr_movl(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        UNREACHABLE();
    }
    void min2_vtr_movh(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        UNREACHABLE();
    }
    void min2_vtr_movl(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        UNREACHABLE();
    }
    void min2_vtr_movh(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        UNREACHABLE();
    }
    void max2_vtr_movij(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        UNREACHABLE();
    }
    void max2_vtr_movji(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        UNREACHABLE();
    }
    void min2_vtr_movij(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        UNREACHABLE();
    }
    void min2_vtr_movji(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        UNREACHABLE();
    }

    template <typename ArpStepX>
    void mov_sv_app(ArRn1 a, ArpStepX as, Bx b, SumBase base, bool sub_p0, bool p0_align,
                    bool sub_p1, bool p1_align) {
        UNREACHABLE();
    }

    void cbs(Axh a, CbsCond c) {
        UNREACHABLE();
    }
    void cbs(Axh a, Bxh b, CbsCond c) {
        UNREACHABLE();
    }
    void cbs(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, CbsCond c) {
        UNREACHABLE();
    }

    void mma(RegName a, bool x0_sign, bool y0_sign, bool x1_sign, bool y1_sign, SumBase base,
             bool sub_p0, bool p0_align, bool sub_p1, bool p1_align) {
        UNREACHABLE();
    }

    template <typename ArpRnX, typename ArpStepX>
    void mma(ArpRnX xy, ArpStepX i, ArpStepX j, bool dmodi, bool dmodj, RegName a, bool x0_sign,
             bool y0_sign, bool x1_sign, bool y1_sign, SumBase base, bool sub_p0, bool p0_align,
             bool sub_p1, bool p1_align) {
        UNREACHABLE();
    }

    void mma_mx_xy(ArRn1 y, ArStep1 ys, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {
        UNREACHABLE();
    }

    void mma_xy_mx(ArRn1 y, ArStep1 ys, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {
        UNREACHABLE();
    }

    void mma_my_my(ArRn1 x, ArStep1 xs, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {
        UNREACHABLE();
    }

    void mma_mov(Axh u, Bxh v, ArRn1 w, ArStep1 ws, RegName a, bool x0_sign, bool y0_sign,
                 bool x1_sign, bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                 bool p1_align) {
        UNREACHABLE();
    }

    void mma_mov(ArRn2 w, ArStep1 ws, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                 bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                 bool p1_align) {
        UNREACHABLE();
    }

    void addhp(ArRn2 a, ArStep2 as, Px b, Ax c) {
        UNREACHABLE();
    }

    void mov_ext0(Imm8s a) {
        UNREACHABLE();
    }
    void mov_ext1(Imm8s a) {
        UNREACHABLE();
    }
    void mov_ext2(Imm8s a) {
        UNREACHABLE();
    }
    void mov_ext3(Imm8s a) {
        UNREACHABLE();
    }

private:
    void ConditionPass(Cond cond, auto&& func) {
        Xbyak::Label end_cond;
        switch (cond.GetName()) {
        case CondValue::True:
            func();
            return;
        case CondValue::Eq:
            // return fz == 1;
            c.test(FLAGS, decltype(Flags::fz)::mask);
            c.jz(end_cond);
            break;
        case CondValue::Neq:
            // return fz == 0;
            c.test(FLAGS, decltype(Flags::fz)::mask);
            c.jnz(end_cond);
            break;
        case CondValue::Gt:
            // return fz == 0 && fm == 0;
            c.test(FLAGS, decltype(Flags::fz)::mask | decltype(Flags::fm)::mask);
            c.jnz(end_cond);
            break;
        case CondValue::Ge:
            // return fm == 0;
            c.test(FLAGS, decltype(Flags::fm)::mask);
            c.jnz(end_cond);
            break;
        case CondValue::Lt:
            // return fm == 1;
            c.test(FLAGS, decltype(Flags::fm)::mask);
            c.jz(end_cond);
            break;
        case CondValue::Le:
            // return fm == 1 || fz == 1;
            c.test(FLAGS, decltype(Flags::fm)::mask | decltype(Flags::fz)::mask);
            c.jz(end_cond);
            break;
        case CondValue::Nn:
            // return fn == 0;
            c.test(FLAGS, decltype(Flags::fn)::mask);
            c.jnz(end_cond);
            break;
        case CondValue::C:
            // return fc0 == 1;
            c.test(FLAGS, decltype(Flags::fc0)::mask);
            c.jz(end_cond);
            break;
        case CondValue::V:
            // return fv == 1;
            c.test(FLAGS, decltype(Flags::fv)::mask);
            c.jz(end_cond);
            break;
        case CondValue::E:
            // return fe == 1;
            c.test(FLAGS, decltype(Flags::fe)::mask);
            c.jz(end_cond);
            break;
        case CondValue::L:
            // return flm == 1 || fvl == 1;
            c.test(FLAGS, decltype(Flags::flm)::mask | decltype(Flags::fvl)::mask);
            c.jz(end_cond);
            break;
        case CondValue::Nr:
            // return fr == 0;
            c.test(FLAGS, decltype(Flags::fr)::mask);
            c.jnz(end_cond);
            break;
        case CondValue::Niu0:
        case CondValue::Iu0:
        case CondValue::Iu1:
        default:
            UNREACHABLE();
        }
        func();
        c.L(end_cond);
    }

    void RegToBus16(RegName reg, Reg64 out, bool enable_sat_for_mov = false) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            GetAcc(out, reg);
            c.and_(out, 0xFFFF);
            break;
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            if (enable_sat_for_mov) {
                GetAndSatAcc(out, reg);
            } else {
                GetAcc(out, reg);
            }
            c.and_(out, 0xFFFF);
            break;
        case RegName::a0h:
        case RegName::a1h:
        case RegName::b0h:
        case RegName::b1h:
            if (enable_sat_for_mov) {
                UNREACHABLE();
            } else {
                GetAcc(out, reg);
                c.shr(out, 16);
            }
            c.and_(out, 0xFFFF);
            break;
        case RegName::a0e:
        case RegName::a1e:
        case RegName::b0e:
        case RegName::b1e:
            UNREACHABLE();

        case RegName::r0:
            c.movzx(out, R0_1_2_3.cvt16());
            break;
        case RegName::r1:
        case RegName::r2:
        case RegName::r3:
            UNREACHABLE();
            break;
        case RegName::r4:
            c.movzx(out, R4_5_6_7.cvt16());
            break;
        case RegName::r5:
        case RegName::r6:
        case RegName::r7:
            UNREACHABLE();
            break;
        case RegName::y0:
            c.mov(out, FACTORS);
            c.shr(out, 32);
            break;
        case RegName::sp:
            c.movzx(out, word[REGS + offsetof(JitRegisters, sp)]);
            break;
        case RegName::mod0:
            UNREACHABLE();
            break;
        default:
            UNREACHABLE();
        }
    }

    void GetAcc(Xbyak::Reg64 out, RegName name) {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            c.mov(out, A[0]);
            break;
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            c.mov(out, A[1]);
            break;
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            c.mov(out, B[0]);
            break;
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            c.mov(out, B[1]);
            break;
        default:
            UNREACHABLE();
        }
    }

    void RegFromBus16(RegName reg, Reg64 value) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            SignExtend(value, 16);
            SatAndSetAccAndFlag(reg, value);
            break;
        case RegName::st1:
            regs.SetSt1(c, value, blk_key.curr.mod0, blk_key.curr.mod1);
            compiling = false; // Modifies static state, end block
            break;
        case RegName::sp:
            c.mov(word[REGS + offsetof(JitRegisters, sp)], value.cvt16());
            break;
        default:
            UNREACHABLE();
        }
    }

    void RegFromBus16(RegName reg, u16 value) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            SatAndSetAccAndFlag(reg, ::SignExtend<16, u64>(value));
            break;
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            SatAndSetAccAndFlag(reg, (u64)value);
            break;
        case RegName::a0h:
            SatAndSetAccAndFlag(reg, ::SignExtend<32, u64>(value << 16));
            break;
        case RegName::a1h:
            SatAndSetAccAndFlag(reg, ::SignExtend<32, u64>(value << 16));
            break;
        case RegName::b0h:
            SatAndSetAccAndFlag(reg, ::SignExtend<32, u64>(value << 16));
            break;
        case RegName::b1h:
            SatAndSetAccAndFlag(reg, ::SignExtend<32, u64>(value << 16));
            break;
        case RegName::a0e:
        case RegName::a1e:
        case RegName::b0e:
        case RegName::b1e:
            UNREACHABLE();

        case RegName::r0:
            c.mov(R0_1_2_3.cvt16(), value);
            break;
        case RegName::r1:
        case RegName::r2:
        case RegName::r3:
            UNREACHABLE();
            break;
        case RegName::r4:
            c.mov(R4_5_6_7.cvt16(), value);
            break;
        case RegName::r5:
        case RegName::r6:
        case RegName::r7:
            UNREACHABLE();
            break;

        case RegName::y0:
            c.rorx(FACTORS, FACTORS, 32);
            c.mov(FACTORS.cvt16(), value);
            c.rorx(FACTORS, FACTORS, 32);
            break;

        case RegName::sp:
            c.mov(word[REGS + offsetof(JitRegisters, sp)], value);
            break;
        case RegName::mod0:
            blk_key.curr.mod0.raw = value;
            regs.SetMod0(c, value);
            break;
        case RegName::mod1:
            blk_key.curr.mod1.raw = value;
            regs.SetMod1(c, value);
            break;
        case RegName::mod2:
            blk_key.curr.mod2.raw = value;
            regs.SetMod2(c, value);
            break;
        case RegName::mod3:
            regs.SetMod3(c, value);
            break;

        case RegName::cfgi:
            blk_key.curr.cfgi.raw = value;
            regs.SetCfgi(c, value);
            break;
        case RegName::cfgj:
            blk_key.curr.cfgj.raw = value;
            regs.SetCfgj(c, value);
            break;

        case RegName::st0:
            regs.SetSt0(c, value, blk_key.curr.mod0);
            break;
        case RegName::st1:
            regs.SetSt1(c, value, blk_key.curr.mod0, blk_key.curr.mod1);
            break;
        case RegName::st2:
            regs.SetSt2(c, value, blk_key.curr.mod0, blk_key.curr.mod2);
            break;

        case RegName::ar0:
            blk_key.curr.ar[0].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, ar)], value);
            break;
        case RegName::ar1:
            blk_key.curr.ar[1].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, ar) + sizeof(u16)], value);
            break;

        case RegName::arp0:
            blk_key.curr.arp[0].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp)], value);
            break;
        case RegName::arp1:
            blk_key.curr.arp[1].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16)], value);
            break;
        case RegName::arp2:
            blk_key.curr.arp[2].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16) * 2], value);
            break;
        case RegName::arp3:
            blk_key.curr.arp[3].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16) * 3], value);
            break;

        default:
            UNREACHABLE();
        }
    }

    template <typename T>
    void SetAccFlag(T value) {
        constexpr u16 ACC_MASK = ~(decltype(Flags::fz)::mask | decltype(Flags::fm)::mask |
                                   decltype(Flags::fe)::mask | decltype(Flags::fn)::mask);
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            const Reg64 scratch = rdx;
            c.xor_(scratch, scratch);
            c.and_(FLAGS.cvt32(), ACC_MASK); // clear fz, fm, fe, fn
            c.test(value, value);
            c.setz(scratch.cvt8()); // mask = (value == 0) ? 1 : 0;
            c.shl(scratch, decltype(Flags::fz)::position); // mask <<= fz_pos
            // regs.fz = value == 0;
            c.or_(FLAGS, scratch); // flags |= mask
            static_assert(decltype(Flags::fz)::position - decltype(Flags::fn)::position == 2);
            c.shr(scratch, 2); // mask >>= (fz_pos - fn_pos);
            c.or_(FLAGS, scratch);
            // regs.fm = (value >> 39) != 0;
            c.mov(scratch, value);
            c.shr(scratch, 39);
            c.test(scratch, scratch);
            c.setnz(scratch.cvt8());
            c.shl(scratch, decltype(Flags::fm)::position);
            c.or_(FLAGS.cvt8(), scratch.cvt8());
            // regs.fe = value != SignExtend<32>(value);
            c.movsxd(scratch, value.cvt32());
            c.cmp(scratch, value);
            c.setne(scratch.cvt8());
            c.shl(scratch, decltype(Flags::fe)::position);
            c.or_(FLAGS.cvt8(), scratch.cvt8());
            c.shl(scratch, 16 - decltype(Flags::fe)::position);
            c.btc(scratch, 16);
            c.bt(value, 31);  // u64 bit31 = (value >> 31) & 1;
            c.setc(dl);
            c.bt(value, 30); // u64 bit30 = (value >> 30) & 1;
            c.setc(dh);
            c.xor_(dh, dl);
            c.shr(scratch, 8);
            c.and_(scratch, 0x101);
            c.and_(dl, dh);
            static_assert(decltype(Flags::fn)::position < 8);
            c.shl(dl, decltype(Flags::fn)::position);
            c.or_(FLAGS.cvt8(), dl);
            // regs.fn = regs.fz || (!regs.fe && (bit31 ^ bit30) != 0);
        } else {
            const u64 bit31 = (value >> 31) & 1;
            const u64 bit30 = (value >> 30) & 1;
            const u16 fz = value == 0;
            const u16 fm = (value >> 39) != 0;
            const u16 fe = value != ::SignExtend<32>(value);
            const u16 fn = fz || (!fe && (bit31 ^ bit30) != 0);

            u16 mask = 0;
            mask |= fz << decltype(Flags::fz)::position;
            mask |= fm << decltype(Flags::fm)::position;
            mask |= fe << decltype(Flags::fe)::position;
            mask |= fn << decltype(Flags::fn)::position;

            c.and_(FLAGS, ACC_MASK); // clear fz, fm, fe, fn
            c.or_(FLAGS, mask);
        }
    }

    template <typename T>
    void SetAcc(RegName name, T value) {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            c.mov(A[0], value);
            break;
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            c.mov(A[1], value);
            break;
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            c.mov(B[0], value);
            break;
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            c.mov(B[1], value);
            break;
        default:
            UNREACHABLE();
        }
    }

    template <typename T>
    void SaturateAcc(T& value) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            Xbyak::Label end_saturate;
            c.movsxd(rsi, value.cvt32()); // rbx = SignExtend<32>(value);
            c.cmp(value, rsi);
            c.je(end_saturate);
            c.or_(FLAGS, decltype(Flags::flm)::mask); // regs.flm = 1;
            c.shr(value, 39);
            c.mov(rsi, 0x0000'0000'7FFF'FFFF);
            c.test(value, value);
            c.mov(value, 0xFFFF'FFFF'8000'0000);
            c.cmovz(value, rsi);
            // note: flm doesn't change value otherwise
            c.L(end_saturate);
        } else {
            if (value != ::SignExtend<32>(value)) {
                c.or_(FLAGS, decltype(Flags::flm)::mask); // regs.flm = 1;
                if ((value >> 39) != 0)
                    value = 0xFFFF'FFFF'8000'0000;
                else
                    value = 0x0000'0000'7FFF'FFFF;
            }
            // note: flm doesn't change value otherwise
        }
    }

    template <typename T>
    void SatAndSetAccAndFlag(RegName name, T value) {
        SetAccFlag(value);
        if (!blk_key.curr.mod0.sata) {
            SaturateAcc(value);
        }
        SetAcc(name, value);
    }

    void SetAccAndFlag(RegName name, Reg64 value) {
        SetAccFlag(value);
        SetAcc(name, value);
    }

    void GetAndSatAcc(Reg64 out, RegName name) {
        GetAcc(out, name);
        if (!blk_key.curr.mod0.sat) {
            SaturateAcc(out);
        }
    }

    void ProductToBus40(Reg64 value, Px reg) {
        const u16 unit = reg.Index();
        c.mov(value.cvt16(), word[REGS + offsetof(JitRegisters, pe) + sizeof(u16) * unit]);
        c.shl(value, 32);
        c.mov(value.cvt32(), dword[REGS + offsetof(JitRegisters, p) + sizeof(u32) * unit]);
        switch (unit == 0 ? blk_key.curr.mod0.ps0.Value() : blk_key.curr.mod0.ps1.Value()) {
        case 0:
            SignExtend(value, 33);
            break;
        case 1:
            c.shr(value, 1);
            SignExtend(value, 32);
            break;
        case 2:
            c.shl(value, 1);
            SignExtend(value, 34);
            break;
        case 3:
            c.shl(value, 2);
            SignExtend(value, 35);
            break;
        }
    }

    bool CompareRegisterState() {
        bool result = true;
        if (!(regs.pc == iregs.pc && regs.r == iregs.r && regs.cpc == iregs.cpc && regs.prpage == iregs.prpage && regs.repc == iregs.repc && regs.repcs == iregs.repcs)) {
            printf("Failed part 0 of checks\n");
            result = false;
        }
        if (!(regs.im == iregs.im && regs.ou == iregs.ou && regs.nimc == iregs.nimc)) {
            printf("Failed part 1 of checks\n");
            result = false;
        }
        if (!(regs.rep == iregs.rep && regs.crep == iregs.crep && regs.bcn == iregs.bcn && regs.lp == iregs.lp && regs.a == iregs.a && regs.b == iregs.b)) {
            printf("Failed part 2 of checks\n");
            result = false;
        }
        if (!(regs.ccnta == iregs.ccnta && regs.mod0.sat == iregs.sat && regs.mod0.sata == iregs.sata && regs.mod0.s == iregs.s && regs.sv == iregs.sv)) {
            printf("Failed part 3 of checks\n");
            result = false;
        }
        if (!(regs.flags.fz == iregs.fz && regs.flags.fm == iregs.fm && regs.flags.fn == iregs.fn && regs.flags.fv == iregs.fv && regs.flags.fe == iregs.fe)) {
            printf("Failed part 4 of checks\n");
            result = false;
        }
        if (!(regs.flags.fc0 == iregs.fc0 && regs.flags.fc1 == iregs.fc1 && regs.flags.flm == iregs.flm && regs.flags.fvl == iregs.fvl && regs.flags.fr == iregs.fr)) {
            printf("Failed part 5 of checks\n");
            result = false;
        }
        if (!(regs.vtr0 == iregs.vtr0 && regs.vtr1 == iregs.vtr1 && regs.x == iregs.x && regs.y == iregs.y && regs.mod0.hwm == iregs.hwm && regs.p == iregs.p)) {
            printf("Failed part 6 of checks\n");
            result = false;
        }
        if (!(regs.pe == iregs.pe && regs.mod0.ps0 == iregs.ps[0] && regs.mod0.ps1 == iregs.ps[1] && regs.p0h_cbs == iregs.p0h_cbs && regs.mixp == iregs.mixp)) {
            printf("Failed part 7 of checks\n");
            result = false;
        }
        if (!(regs.sp == iregs.sp && regs.mod1.page == iregs.page && regs.pcmhi == iregs.pcmhi && regs.cfgi.step == iregs.stepi && regs.cfgj.step == iregs.stepj)) {
            printf("Failed part 8 of checks\n");
            result = false;
        }
        if (!(regs.cfgi.mod == iregs.modi && regs.cfgj.mod == iregs.modj && regs.stepi0 == iregs.stepi0 && regs.stepj0 == iregs.stepj0 && regs.mod1.stp16 == iregs.stp16)) {
            printf("Failed part 9 of checks\n");
            result = false;
        }
        if (!(regs.mod1.cmd == iregs.cmd && regs.mod1.epi == iregs.epi && regs.mod1.epj == iregs.epj)) {
            printf("Failed part 10 of checks\n");
            result = false;
        }
        return result;

    }
};

} // namespace Teakra
