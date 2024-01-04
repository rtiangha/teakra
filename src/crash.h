#pragma once
#include <cstdio>
#include <cstdlib>

[[noreturn]] inline void Assert(const char* expression, const char* file, int line) {
    std::fprintf(stderr, "Assertion '%s' failed, file '%s' line '%d'.", expression, file, line);
    std::abort();
}

#ifndef ASSERT
#define ASSERT(EXPRESSION) ((EXPRESSION) ? (void)0 : Assert(#EXPRESSION, __FILE__, __LINE__))
#endif
#ifndef UNREACHABLE
#define UNREACHABLE() Assert("UNREACHABLE", __FILE__, __LINE__)
#endif
