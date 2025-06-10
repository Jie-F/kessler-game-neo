cmake_minimum_required(VERSION 3.15)
project(neo_controller LANGUAGES CXX)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(nanobind)

nanobind_add_module(neo_controller neo_controller.cpp)

set_target_properties(neo_controller PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(neo_controller PRIVATE /W4 /EHsc /std:c++latest)
    target_link_options(neo_controller PRIVATE /DEBUG)
    target_compile_options(neo_controller PRIVATE
        "$<$<CONFIG:Release>:/O2>"
        "$<$<CONFIG:Debug>:/Zi>"
        "$<$<CONFIG:Debug>:/RTC1>"
    )
endif()

if (MSVC)
    add_custom_command(TARGET neo_controller POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -f
            "${CMAKE_SOURCE_DIR}/neo_controller.lib"
            "${CMAKE_SOURCE_DIR}/neo_controller.exp"
            "${CMAKE_SOURCE_DIR}/neo_controller.obj"
            #"${CMAKE_SOURCE_DIR}/neo_controller.pdb"
    )
endif()
