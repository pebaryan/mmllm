#pragma once
#include <GL/glew.h>
#include <string>
#include <cstdint>

namespace gl {

class Shader {
public:
    Shader() = default;
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    Shader(Shader&& other) noexcept;
    Shader& operator=(Shader&& other) noexcept;

    // Compile a shader from source string
    bool compile(GLenum type, const std::string& source);

    // Load and compile a shader from file
    bool compileFromFile(GLenum type, const std::string& filepath);

    uint32_t id() const { return id_; }
    bool valid() const { return id_ != 0; }

    void destroy();

private:
    uint32_t id_ = 0;
    bool checkCompileErrors() const;
    std::string readFile(const std::string& path) const;
};

} // namespace gl
