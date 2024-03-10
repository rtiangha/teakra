#include "jit.h"
#include "processor.h"
#include "register.h"

namespace Teakra {

struct Processor::Impl {
    Impl(CoreTiming& core_timing, MemoryInterface& memory_interface, bool use_jit_)
        : core_timing(core_timing), interpreter(core_timing, iregs, memory_interface),
          jit(core_timing, regs, memory_interface), use_jit(use_jit_) {}
    CoreTiming& core_timing;
    JitRegisters regs;
    RegisterState iregs;
    Interpreter interpreter;
    EmitX64 jit;
    bool use_jit;
};

Processor::Processor(CoreTiming& core_timing, MemoryInterface& memory_interface, bool use_jit)
    : impl(new Impl(core_timing, memory_interface, use_jit)) {}

Processor::~Processor() = default;

void Processor::Reset() {
    if (impl->use_jit) {
        impl->jit.Reset();
    } else {
        impl->iregs.Reset();
    }
}

void Processor::Run(unsigned cycles, Interpreter* debug_interp) {
    if (impl->use_jit) {
        impl->jit.Run(cycles, debug_interp);
    } else {
        impl->interpreter.Run(cycles);
    }
}

void Processor::SignalInterrupt(u32 i) {
    if (impl->use_jit) {
        impl->jit.SignalInterrupt(i);
    } else {
        impl->interpreter.SignalInterrupt(i);
    }
}
void Processor::SignalVectoredInterrupt(u32 address, bool context_switch) {
    if (impl->use_jit) {
        impl->jit.SignalVectoredInterrupt(address, context_switch);
    } else {
        impl->interpreter.SignalVectoredInterrupt(address, context_switch);
    }
}

Interpreter& Processor::Interp() {
    return impl->interpreter;
}

} // namespace Teakra
