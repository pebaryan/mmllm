#pragma once
#include "../engine/tensor.h"
#include "../gl/program.h"
#include <memory>

class ElementwiseAdd {
public:
    ElementwiseAdd();
    ~ElementwiseAdd() = default;

    bool init();

    // C = A + B (element-wise)
    bool forward(const gl::Texture& texA, const gl::Texture& texB,
                 int texWidth, int texHeight, int totalElements,
                 gl::FBO& outputFBO);

private:
    std::unique_ptr<gl::Program> addProg_;
    std::unique_ptr<gl::Shader> vertShader_;
    std::unique_ptr<gl::Shader> fragShader_;
};

class ElementwiseScale {
public:
    ElementwiseScale();
    ~ElementwiseScale() = default;

    bool init();

    // out = in * scalar + bias
    bool forward(const gl::Texture& texInput, float scalar,
                 const float bias[4], int N, gl::FBO& outputFBO);

private:
    std::unique_ptr<gl::Program> scaleProg_;
    std::unique_ptr<gl::Shader> vertShader_;
    std::unique_ptr<gl::Shader> fragShader_;
};
