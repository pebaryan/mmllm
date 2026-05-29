cmake_minimum_required(VERSION 3.10)
project(mmllm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Collect all source files
file(GLOB_RECURSE SOURCES
    src/gl/*.cpp
    src/engine/*.cpp
    src/layers/*.cpp
    src/main.cpp
)

# Create executable
add_executable(mmllm ${SOURCES})

# Link libraries
target_link_libraries(mmllm
    ${OPENGL_gl_LIBRARY}
    GLEW::GLEW
    glfw
)

# Include directories
target_include_directories(mmllm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OPENGL_INCLUDE_DIRS}
)

# Set build type to Release by default for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable optimizations
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(mmllm PRIVATE -O2 -march=native -mtune=native)
endif()

# Copy shaders to build directory
file(COPY src/shaders/ DESTINATION ${CMAKE_BINARY_DIR}/src/shaders)

# Copy tools to build directory
file(COPY tools/ DESTINATION ${CMAKE_BINARY_DIR}/tools)

# Create models directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/models)

# Print info
message(STATUS "mmllm v${PROJECT_VERSION}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenGL: ${OPENGL_gl_LIBRARY}")
message(STATUS "GLEW: ${GLEW_LIBRARIES}")
message(STATUS "GLFW: found")
message(STATUS "")
message(STATUS "To build:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. && make -j\$(nproc)")
message(STATUS "")
message(STATUS "To run:")
message(STATUS "  ./mmllm --self-test")
message(STATUS "  ./mmllm --model ../models/tinystories-1m.mlm --tokens 100")
