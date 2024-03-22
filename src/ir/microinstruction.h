#pragma once

#include <array>

#include <mcl/container/intrusive_list.hpp>
#include <mcl/stdint.hpp>

#include "ir/value.h"

namespace Teakra::IR {

enum class Opcode;
enum class Type;

/**
 * A representation of a microinstruction. A single Teak instruction may be
 * converted into zero or more microinstructions.
 */
class Inst final : public mcl::intrusive_list_node<Inst> {
    static constexpr size_t max_arg_count = 4;

public:
    explicit Inst(Opcode op) : op(op) {}

    /// Get the microop this microinstruction represents.
    Opcode GetOpcode() const {
        return op;
    }
    /// Get the number of arguments this instruction has.
    size_t NumArgs() const;

private:
    Opcode op;
    u32 use_count = 0;
    u32 name = 0;
    std::array<Value, max_arg_count> args;

    // Linked list of pseudooperations associated with this instruction.
    Inst* next_pseudoop = nullptr;
};

} // namespace Teakra::IR
