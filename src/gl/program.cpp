#include "program.h"
#include <cstdio>
#include <utility>

namespace gl {

Program::~Program() {
    destroy();
}

bool Program::link(const Shader& vert, const Shader& frag) {
    destroy();

    id_ = glCreateProgram();
    glAttachShader(id_, vert.id());
    glAttachShader(id_, frag.id());
    glLinkProgram(id_);

    return checkLinkErrors();
}

void Program::use() const {
    glUseProgram(id_);
}

void Program::unuse() const {
    glUseProgram(0);
}

bool Program::checkLinkErrors() const {
    GLint success = 0;
    glGetProgramiv(id_, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetProgramInfoLog(id_, sizeof(infoLog), nullptr, infoLog);
        std::fprintf(stderr, "[mmllm] Program link error:\n%s\n", infoLog);
        return false;
    }
    return true;
}

GLint Program::getUniformLoc(const std::string& name) const {
    return glGetUniformLocation(id_, name.c_str());
}

void Program::setInt(const std::string& name, int val) const {
    glUniform1i(getUniformLoc(name), val);
}

void Program::setFloat(const std::string& name, float val) const {
    glUniform1f(getUniformLoc(name), val);
}

void Program::setVec2(const std::string& name, int x, int y) const {
    glUniform2i(getUniformLoc(name), x, y);
}

void Program::setBool(const std::string& name, bool val) const {
    glUniform1i(getUniformLoc(name), val ? 1 : 0);
}

void Program::destroy() {
    if (id_) {
        glDeleteProgram(id_);
        id_ = 0;
    }
}

Program::Program(Program&& other) noexcept
    : id_(other.id_)
{
    other.id_ = 0;
}

Program& Program::operator=(Program&& other) noexcept {
    if (this != &other) {
        destroy();
        id_ = other.id_;
        other.id_ = 0;
    }
    return *this;
}

} // namespace gl
