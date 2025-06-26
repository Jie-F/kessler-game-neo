cmake_minimum_required(VERSION 3.15)
project(neo_controller LANGUAGES CXX)

# Determine proper Python development module name based on CMake version
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

# Find Python 3.13 and override problematic default paths
find_package(Python 3.13 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Manually set include and library paths to avoid missing 't' (free-threaded) lib
set(Python_INCLUDE_DIR "C:/Python313/include")
set(Python_LIBRARY_RELEASE "C:/Python313/libs/python313.lib")
set(Python_LIBRARY_DEBUG "C:/Python313/libs/python313.lib")  # override to avoid missing python313t.lib

# Default to Release build if unspecified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# C++ configuration
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add nanobind subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nanobind)

# Add the actual module
nanobind_add_module(neo_controller neo_controller.cpp)

# Use manual include + conditional link logic to avoid 't' suffix problems
target_include_directories(neo_controller PRIVATE ${Python_INCLUDE_DIR})
target_link_libraries(neo_controller
  PRIVATE
    $<$<CONFIG:Debug>:${Python_LIBRARY_DEBUG}>
    $<$<NOT:$<CONFIG:Debug>>:${Python_LIBRARY_RELEASE}>
)

# Set output directories
set_target_properties(neo_controller PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_SOURCE_DIR}"
)

# MSVC-specific options
if(MSVC)
    target_compile_options(neo_controller PRIVATE /W4 /EHsc /std:c++latest)

    # Linker and compiler flags for Debug and Release modes
    target_link_options(neo_controller PRIVATE /DEBUG)
    target_compile_options(neo_controller PRIVATE
        "$<$<CONFIG:Release>:/O2>"
        "$<$<CONFIG:Debug>:/Od>"
        "$<$<CONFIG:Debug>:/Zi>"
        "$<$<CONFIG:Debug>:/RTC1>"
    )
endif()

# Clean up MSVC artifacts we don't want
if (MSVC)
    add_custom_command(TARGET neo_controller POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -f
            "${CMAKE_SOURCE_DIR}/neo_controller.lib"
            "${CMAKE_SOURCE_DIR}/neo_controller.exp"
            "${CMAKE_SOURCE_DIR}/neo_controller.obj"
            # "${CMAKE_SOURCE_DIR}/neo_controller.pdb"
    )
endif()
