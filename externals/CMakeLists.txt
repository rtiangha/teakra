if (TEAKRA_BUILD_UNIT_TESTS)
    add_library(catch INTERFACE)
    target_include_directories(catch INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/catch>)
endif()

set(MCL_INSTALL OFF)
add_subdirectory(mcl)

if (NOT TARGET xbyak::xbyak)
    add_subdirectory(xbyak)
endif()
