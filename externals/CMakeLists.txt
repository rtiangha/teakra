if (TEAKRA_BUILD_UNIT_TESTS)
    add_library(catch INTERFACE)
    target_include_directories(catch INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/catch>)
endif()

if (NOT TARGET xbyak::xbyak)
    add_subdirectory(xbyak)
endif()

if (NOT TARGET tsl::robin_map)
    add_subdirectory(robin-map)
endif()
