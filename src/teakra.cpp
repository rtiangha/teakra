#include <array>
#include "ahbm.h"
#include "apbp.h"
#include "btdmp.h"
#include "core_timing.h"
#include "dma.h"
#include "icu.h"
#include "memory_interface.h"
#include "mmio.h"
#include "processor.h"
#include "shared_memory.h"
#include "teakra/teakra.h"
#include "timer.h"

namespace Teakra {

struct Teakra::Impl {
    std::array<Timer, 2> timer{};
    std::array<Btdmp, 2> btdmp{};
    CoreTiming core_timing{timer, btdmp};
    SharedMemory shared_memory;
    MemoryInterfaceUnit miu;
    ICU icu;
    Apbp apbp_from_cpu, apbp_from_dsp;
    Ahbm ahbm;
    Dma dma{shared_memory, ahbm};
    MMIORegion mmio{miu, icu, apbp_from_cpu, apbp_from_dsp, timer, dma, ahbm, btdmp};
    MemoryInterface memory_interface{shared_memory, miu};
    Processor processor;

    Impl(bool use_jit) : processor(core_timing, memory_interface, use_jit) {
        memory_interface.SetMMIO(mmio);
        using namespace std::placeholders;
        icu.SetInterruptHandler(std::bind(&Processor::SignalInterrupt, &processor, _1),
                                std::bind(&Processor::SignalVectoredInterrupt, &processor, _1, _2));

        timer[0].SetInterruptHandler([this]() { icu.TriggerSingle(0xA); });
        timer[1].SetInterruptHandler([this]() { icu.TriggerSingle(0x9); });

        apbp_from_cpu.SetDataHandler(0, [this]() { icu.TriggerSingle(0xE); });
        apbp_from_cpu.SetDataHandler(1, [this]() { icu.TriggerSingle(0xE); });
        apbp_from_cpu.SetDataHandler(2, [this]() { icu.TriggerSingle(0xE); });
        apbp_from_cpu.SetSemaphoreHandler([this]() { icu.TriggerSingle(0xE); });

        btdmp[0].SetInterruptHandler([this]() { icu.TriggerSingle(0xB); });
        btdmp[1].SetInterruptHandler([this]() { icu.TriggerSingle(0xB); });

        dma.SetInterruptHandler([this]() { icu.TriggerSingle(0xF); });
    }

    void Reset() {
        shared_memory.raw.fill(0);
        miu.Reset();
        apbp_from_cpu.Reset();
        apbp_from_dsp.Reset();
        timer[0].Reset();
        timer[1].Reset();
        ahbm.Reset();
        dma.Reset();
        btdmp[0].Reset();
        btdmp[1].Reset();
        processor.Reset();
    }
};

Teakra::Teakra(bool use_jit) : impl_jit(new Impl(true)), impl_interp(new Impl(false)),
      impl(use_jit ? impl_jit.get() : impl_interp.get()), use_jit(use_jit) {}

Teakra::~Teakra() = default;

void Teakra::Reset() {
    impl->Reset();
}

Processor& Teakra::GetProcessor() {
    return impl->processor;
}

std::array<std::uint8_t, 0x80000>& Teakra::GetDspMemory() {
    return impl->shared_memory.raw;
}

const std::array<std::uint8_t, 0x80000>& Teakra::GetDspMemory() const {
    return impl->shared_memory.raw;
}

u32 Teakra::Run(unsigned cycle) {
    return impl->processor.Run(cycle, &impl_interp->processor.Interp());
}

bool Teakra::SendDataIsEmpty(std::uint8_t index) const {
    return !impl->apbp_from_cpu.IsDataReady(index);
}
void Teakra::SendData(std::uint8_t index, std::uint16_t value) {
    impl->apbp_from_cpu.SendData(index, value);
}
bool Teakra::RecvDataIsReady(std::uint8_t index) const {
    return impl->apbp_from_dsp.IsDataReady(index);
}
std::uint16_t Teakra::RecvData(std::uint8_t index) {
    return impl->apbp_from_dsp.RecvData(index);
}
std::uint16_t Teakra::PeekRecvData(std::uint8_t index) {
    return impl->apbp_from_dsp.PeekData(index);
}
void Teakra::SetRecvDataHandler(std::uint8_t index, std::function<void()> handler) {
    impl->apbp_from_dsp.SetDataHandler(index, handler);
}

void Teakra::SetSemaphore(std::uint16_t value) {
    impl->apbp_from_cpu.SetSemaphore(value);
}
void Teakra::SetSemaphoreHandler(std::function<void()> handler) {
    impl->apbp_from_dsp.SetSemaphoreHandler(handler);
}
std::uint16_t Teakra::GetSemaphore() const {
    return impl->apbp_from_dsp.GetSemaphore();
}
void Teakra::ClearSemaphore(std::uint16_t value) {
    impl->apbp_from_dsp.ClearSemaphore(value);
}
void Teakra::MaskSemaphore(std::uint16_t value) {
    impl->apbp_from_dsp.MaskSemaphore(value);
}
void Teakra::SetAHBMCallback(const AHBMCallback& callback) {
    impl->ahbm.SetExternalMemoryCallback(callback.read8, callback.write8,
        callback.read16, callback.write16,
        callback.read32, callback.write32);
}

std::uint16_t Teakra::AHBMGetUnitSize(std::uint16_t i) const {
    return impl->ahbm.GetUnitSize(i);
}
std::uint16_t Teakra::AHBMGetDirection(std::uint16_t i) const {
    impl_interp->ahbm.GetDirection(i);
    return impl->ahbm.GetDirection(i);
}
std::uint16_t Teakra::AHBMGetDmaChannel(std::uint16_t i) const {
    return impl->ahbm.GetDmaChannel(i);
}

std::uint16_t Teakra::AHBMRead16(std::uint32_t addr) {
    return impl->ahbm.Read16(0, addr);
}
void Teakra::AHBMWrite16(std::uint32_t addr, std::uint16_t value) {
    impl->ahbm.Write16(0, addr, value);
}
std::uint16_t Teakra::AHBMRead32(std::uint32_t addr) {
    return impl->ahbm.Read32(0, addr);
}
void Teakra::AHBMWrite32(std::uint32_t addr, std::uint32_t value) {
    impl->ahbm.Write32(0, addr, value);
}

void Teakra::SetAudioCallback(std::function<void(std::array<s16, 2>)> callback) {
    impl->btdmp[0].SetAudioCallback(callback);
}

std::uint16_t Teakra::ProgramRead(std::uint32_t address) const {
    return impl->memory_interface.ProgramRead(address);
}
void Teakra::ProgramWrite(std::uint32_t address, std::uint16_t value) {
    impl->memory_interface.ProgramWrite(address, value);
}
std::uint16_t Teakra::DataRead(std::uint16_t address, bool bypass_mmio) {
    return impl->memory_interface.DataRead(address, bypass_mmio);
}
void Teakra::DataWrite(std::uint16_t address, std::uint16_t value, bool bypass_mmio) {
    impl->memory_interface.DataWrite(address, value, bypass_mmio);
}
std::uint16_t Teakra::DataReadA32(std::uint32_t address) const {
    return impl->memory_interface.DataReadA32(address);
}
void Teakra::DataWriteA32(std::uint32_t address, std::uint16_t value) {
    impl->memory_interface.DataWriteA32(address, value);
}
std::uint16_t Teakra::MMIORead(std::uint16_t address) {
    return impl->memory_interface.MMIORead(address);
}
void Teakra::MMIOWrite(std::uint16_t address, std::uint16_t value) {
    impl->memory_interface.MMIOWrite(address, value);
}

std::uint16_t Teakra::DMAChan0GetSrcHigh() {
    u16 active_bak = impl->dma.GetActiveChannel();
    impl->dma.ActivateChannel(0);
    u16 ret = impl->dma.GetAddrSrcHigh();
    impl->dma.ActivateChannel(active_bak);
    return ret;
}

std::uint16_t Teakra::DMAChan0GetDstHigh() {
    u16 active_bak = impl->dma.GetActiveChannel();
    impl->dma.ActivateChannel(0);
    u16 ret = impl->dma.GetAddrDstHigh();
    impl->dma.ActivateChannel(active_bak);
    return ret;
}

} // namespace Teakra
