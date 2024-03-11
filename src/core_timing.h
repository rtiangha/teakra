#pragma once

#include <array>
#include "common_types.h"
#include "btdmp.h"
#include "timer.h"

namespace Teakra {

class CoreTiming {
public:
    CoreTiming(std::array<Timer, 2>& timer_, std::array<Btdmp, 2>& btdmp_) :
          timer{timer_}, btdmp{btdmp_} {}

    void Tick(u64 ticks = 1) {
        timer[0].Tick(ticks);
        timer[1].Tick(ticks);
        btdmp[0].Tick(ticks);
        btdmp[1].Tick(ticks);
    }

    u64 Skip(u64 maximum) {
        const u64 ticks = GetMaxSkip(maximum);
        timer[0].Skip(ticks);
        timer[1].Skip(ticks);
        btdmp[0].Skip(ticks);
        btdmp[1].Skip(ticks);
        return ticks;
    }

    u64 GetMaxSkip(u64 maximum) {
        u64 ticks = maximum;
        ticks = std::min(ticks, timer[0].GetMaxSkip());
        ticks = std::min(ticks, timer[1].GetMaxSkip());
        ticks = std::min(ticks, btdmp[0].GetMaxSkip());
        ticks = std::min(ticks, btdmp[1].GetMaxSkip());
        return ticks;
    }

private:
    std::array<Timer, 2>& timer;
    std::array<Btdmp, 2>& btdmp;
};

} // namespace Teakra
