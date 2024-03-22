#pragma once

namespace Teakra::IR {

enum class Opcode {
    GetRegister16, // U16, RegName
    GetRegister32, // U32, RegName
    GetAcc,        // U64, RegName
    GetXFactor,    // U32,
    GetYFactor,
    SetRegister16,       // Void, RegName, U16
    SetAcc,              // Void, RegName, U64
    SetProduct,          // Void, U32, U32
    SetAccFlag,          //
    SignExtend32,        // U32, U32, U8
    SaturateAcc,         // U64, U64, U1
    LogicalShiftLeft32,  // U32, U32, U8
    LogicalShiftRight32, // U32, U32, U8
    Add16,               // U16, U16, U16
    Sub16,               // U16, U16, U16
    Mul32,               // U32, U32, U32
    And32,               // U32, U32, U32
    TestBit,             // U1, U32, U8
    ExtractHalf32,       // U16, U32, U8
    ExtractHalf64,       // U16, U64, U8
    ReadMemory16,        // U16, U16
    WriteMemory16,       // Void, U16, U16
    Pack2x16To1x32       // U32, U16, U16
};

} // namespace Teakra::IR
