#include "shader.h"
#include <GL/glew.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <utility>

namespace gl {

Shader::~Shader() {
    destroy();
}

bool Shader::compile(GLenum type, const std::string& source) {
    destroy();

    id_ = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id_, 1, &src, nullptr);
    glCompileShader(id_);

    return checkCompileErrors();
}

bool Shader::compileFromFile(GLenum type, const std::string& filepath) {
    std::string source = readFile(filepath);
    if (source.empty()) return false;
    return compile(type, source);
}

bool Shader::checkCompileErrors() const {
    GLint success = 0;
    glGetShaderiv(id_, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(id_, sizeof(infoLog), nullptr, infoLog);
        std::fprintf(stderr, "[mmllm] Shader compilation error:\n%s\n", infoLog);
        return false;
    }
    return true;
}

std::string Shader::readFile(const std::string& path) const {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::fprintf(stderr, "[mmllm] Failed to open shader file: %s\n", path.c_str());
        return {};
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

void Shader::destroy() {
    if (id_) {
        glDeleteShader(id_);
        id_ = 0;
    }
}

Shader::Shader(Shader&& other) noexcept
    : id_(other.id_)
{
    other.id_ = 0;
}

Shader& Shader::operator=(Shader&& other) noexcept {
    if (this != &other) {
        destroy();
        id_ = other.id_;
        other.id_ = 0;
    }
    return *this;
}

} // namespace gl
