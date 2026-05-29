#pragma once
#include <GL/glew.h>
#include "shader.h"
#include <cstdint>
#include <string>

namespace gl {

class Program {
public:
    Program() = default;
    ~Program();

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;

    // Attach shaders and link program
    bool link(const Shader& vert, const Shader& frag);

    void use() const;
    void unuse() const;

    uint32_t id() const { return id_; }
    bool valid() const { return id_ != 0; }

    // Uniform setters
    void setInt(const std::string& name, int val) const;
    void setFloat(const std::string& name, float val) const;
    void setVec2(const std::string& name, int x, int y) const;
    void setBool(const std::string& name, bool val) const;

    void destroy();

private:
    uint32_t id_ = 0;
    bool checkLinkErrors() const;
    GLint getUniformLoc(const std::string& name) const;
};

} // namespace gl
