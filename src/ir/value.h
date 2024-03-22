#pragma once

#include <array>
#include <cstddef>
#include "common_types.h"
#include "crash.h"

enum class RegName;

namespace Teakra::IR {

class Inst;

/**
 * The intermediate representation is typed. These are the used by our IR.
 */
enum class Type {
    Void = 0,
    Opaque = 1 << 0,
    Reg = 1 << 1,
    U1 = 1 << 2,
    U8 = 1 << 3,
    U16 = 1 << 4,
    S16 = 1 << 5,
    U32 = 1 << 6,
    U64 = 1 << 7,
};

constexpr Type operator|(Type a, Type b) {
    return static_cast<Type>(static_cast<size_t>(a) | static_cast<size_t>(b));
}

constexpr Type operator&(Type a, Type b) {
    return static_cast<Type>(static_cast<size_t>(a) & static_cast<size_t>(b));
}

/**
 * A representation of a value in the IR.
 * A value may either be an immediate or the result of a microinstruction.
 */
class Value {
public:
    using CoprocessorInfo = std::array<u8, 8>;

    Value() : type(Type::Void) {}
    explicit Value(Inst* inst) : type(Type::Opaque) {
        inner.inst = inst;
    }
    explicit Value(RegName value) : type(Type::Reg) {
        inner.reg = value;
    }
    explicit Value(bool imm) : type(Type::U1) {
        inner.imm_u1 = imm;
    }
    explicit Value(u8 imm) : type(Type::U8) {
        inner.imm_u8 = imm;
    }
    explicit Value(u16 imm) : type(Type::U16) {
        inner.imm_u16 = imm;
    }
    explicit Value(s16 imm) : type(Type::S16) {
        inner.imm_s16 = imm;
    }
    explicit Value(u32 imm) : type(Type::U32) {
        inner.imm_u32 = imm;
    }
    explicit Value(u64 imm) : type(Type::U64) {
        inner.imm_u64 = imm;
    }

    Type GetType() const {
        return type;
    }

    RegName GetRegName() const {
        return inner.reg;
    }

    bool GetU1() const {
        return inner.imm_u1;
    }

    u16 GetU16() const {
        return inner.imm_u16;
    }

    s16 GetS16() const {
        return inner.imm_s16;
    }

private:
    Type type;

    union {
        Inst* inst;
        RegName reg;
        bool imm_u1;
        u8 imm_u8;
        u16 imm_u16;
        s16 imm_s16;
        u32 imm_u32;
        u64 imm_u64;
    } inner;
};

template <Type type_>
class TypedValue final : public Value {
public:
    TypedValue() = default;

    template <Type other_type>
    TypedValue(const TypedValue<other_type>& value) : Value(value) {
        ASSERT((value.GetType() & type_) != Type::Void);
    }

    explicit TypedValue(const Value& value) : Value(value) {
        ASSERT((value.GetType() & type_) != Type::Void);
    }

    explicit TypedValue(Inst* inst) : TypedValue(Value(inst)) {}
};

using U1 = TypedValue<Type::U1>;
using U8 = TypedValue<Type::U8>;
using U16 = TypedValue<Type::U16>;
using U32 = TypedValue<Type::U32>;
using U64 = TypedValue<Type::U64>;

} // namespace Teakra::IR
