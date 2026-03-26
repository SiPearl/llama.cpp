# llama-bench-impl: benchmark logic, reusable by app

set(TARGET llama-bench-impl)

add_library(${TARGET} llama-bench.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})
option(GGML_CPU_HBM          "ggml: use CPU HBM buffer" OFF)

if (GGML_CPU_HBM)
	target_compile_definitions(${TARGET} PUBLIC GGML_USE_CPU_HBM)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-bench executable

set(TARGET llama-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-bench-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
