#pragma once

#include <memory>
#include "common_types.h"
#include "core_timing.h"

namespace Teakra {

class MemoryInterface;
class Interpreter;

class Processor {
public:
    Processor(CoreTiming& core_timing, MemoryInterface& memory_interface, bool use_jit);
    ~Processor();
    void Reset();
    u32 Run(u32 cycles, Interpreter* debug_interp);
    void SignalInterrupt(u32 i);
    void SignalVectoredInterrupt(u32 address, bool context_switch);
    Interpreter& Interp();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Teakra
