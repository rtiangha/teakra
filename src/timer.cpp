#pragma clang optimize off
#include "crash.h"
#include "timer.h"
#include "crash.h"

namespace Teakra {

void Timer::Reset() {
    update_mmio = 0;
    pause = 0;
    count_mode = CountMode::Single;
    scale = 0;

    start_high = 0;
    start_low = 0;
    counter = 0;
    counter_high = 0;
    counter_low = 0;
}

void Timer::Restart() {
    ASSERT(static_cast<u16>(count_mode) < 4);
    if (count_mode != CountMode::FreeRunning) {
        counter = ((u32)start_high << 16) | start_low;
        UpdateMMIO();
    }
}

void Timer::Tick(u64 ticks) {
    ASSERT(static_cast<u16>(count_mode) < 4);
    ASSERT(scale == 0);
    if (pause)
        return;
    if (count_mode == CountMode::EventCount)
        return;
    if (ticks > counter) {
        const u16 old_counter = counter;
        if (count_mode == CountMode::AutoRestart) {
            counter = ((u32)start_high << 16) | start_low;
        } else if (count_mode == CountMode::FreeRunning) {
            counter = 0xFFFFFFFF;
        }
        counter -= ticks - old_counter - 1;
        if (old_counter != 0) {
            interrupt_handler();
        }
    } else {
        counter -= ticks;
        if (counter == 0) {
            interrupt_handler();
        }
    }

    UpdateMMIO();
}

void Timer::TickEvent() {
    if (pause)
        return;
    if (count_mode != CountMode::EventCount)
        return;
    if (counter == 0)
        return;
    --counter;
    UpdateMMIO();
    if (counter == 0)
        interrupt_handler();
}

void Timer::UpdateMMIO() {
    if (!update_mmio)
        return;
    counter_high = counter >> 16;
    counter_low = counter & 0xFFFF;
}

u64 Timer::GetMaxSkip() const {
    if (pause || count_mode == CountMode::EventCount)
        return Infinity;

    if (counter == 0) {
        if (count_mode == CountMode::AutoRestart) {
            return ((u32)start_high << 16) | start_low;
        } else if (count_mode == CountMode::FreeRunning) {
            return 0xFFFFFFFF;
        } else /*Single*/ {
            return Infinity;
        }
    }

    return counter - 1;
}

void Timer::Skip(u64 ticks) {
    if (pause || count_mode == CountMode::EventCount)
        return;

    if (counter == 0) {
        u32 reset;
        if (count_mode == CountMode::AutoRestart) {
            reset = ((u32)start_high << 16) | start_low;
        } else if (count_mode == CountMode::FreeRunning) {
            reset = 0xFFFFFFFF;
        } else {
            return;
        }
        ASSERT(reset >= ticks);
        counter = reset - ((u32)ticks - 1);
    } else {
        ASSERT(counter > ticks);
        counter -= (u32)ticks;
    }

    UpdateMMIO();
}

Timer::Timer(CoreTiming& core_timing) {
    core_timing.RegisterCallbacks(this);
}

} // namespace Teakra
