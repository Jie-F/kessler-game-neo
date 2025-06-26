cmake_minimum_required(VERSION 3.15)
project(neo_controller LANGUAGES CXX)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.13 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
set(Python_LIBRARY "C:/Python313/libs/python313.lib")
set(Python_INCLUDE_DIR "C:/Python313/include")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(nanobind)

nanobind_add_module(neo_controller neo_controller.cpp)

target_compile_definitions(neo_controller PRIVATE Py_GIL_DISABLED=0)


target_include_directories(neo_controller PRIVATE ${Python_INCLUDE_DIR})
target_link_libraries(neo_controller PRIVATE ${Python_LIBRARY})

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
        "$<$<CONFIG:Debug>:/Od>"
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
