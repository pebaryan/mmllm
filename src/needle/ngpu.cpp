#include "ngpu.h"
#include "../engine/tensor.h"     // dispatchShader
#include <GL/glew.h>
#include <cstdio>
#include <cstring>

namespace needle {

NGpu::NGpu() = default;
NGpu::~NGpu() { shutdown(); }

bool NGpu::loadProgram(gl::Program& prog, const char* frag, const std::string& what) {
    // The engine is normally run from the build directory where CMake copied src/shaders; accept the
    // source tree layout too so the binary also works from the repository root.
    const char* dirs[] = {"src/shaders/", "../src/shaders/", nullptr};
    gl::Shader* fs = new gl::Shader();
    bool compiled = false;
    for (int i = 0; dirs[i] && !compiled; i++) {
        std::string p = std::string(dirs[i]) + frag;
        compiled = fs->compileFromFile(GL_FRAGMENT_SHADER, p);
    }
    if (!compiled) {
        err_ = what + ": cannot compile " + frag + " (run from the build directory)";
        delete fs;
        return false;
    }
    std::unique_ptr<gl::Shader> fragShader(fs);
    if (!prog.link(*vert_, *fragShader)) {
        err_ = what + ": failed to link program";
        return false;
    }
    return true;
}

bool NGpu::init(std::string& err) {
    vert_ = std::make_unique<gl::Shader>();
    const char* dirs[] = {"src/shaders/", "../src/shaders/", nullptr};
    bool ok = false;
    for (int i = 0; dirs[i] && !ok; i++) {
        ok = vert_->compileFromFile(GL_VERTEX_SHADER, std::string(dirs[i]) + "passthrough.vert");
    }
    if (!ok) {
        err = err_ = "needle-gpu: cannot compile passthrough.vert (run from the build directory)";
        return false;
    }

    partial_ = std::make_unique<gl::Program>();
    reduce_ = std::make_unique<gl::Program>();
    if (!loadProgram(*partial_, "needle_gemv2_partial.frag", "needle-gpu") ||
        !loadProgram(*reduce_, "needle_gemv2_reduce.frag", "needle-gpu")) {
        err = err_;
        return false;
    }

    ok_ = true;
    return true;
}

void NGpu::shutdown() {
    entries_.clear();
    act_.destroy();
    part_.reset();
    outFbo_.reset();
    partial_.reset();
    reduce_.reset();
    vert_.reset();
    ok_ = false;
}

void NGpu::setCodebook(const float* cb) {
    for (int i = 0; i < 4; i++) cb_[i] = cb[i];
}

int NGpu::uploadQuant(const uint8_t* packed, const float* norms, int out, int in) {
    if (!ok_ || in <= 0 || out <= 0 || in % 128 != 0) return -1;
    Entry e;
    e.out = out;
    e.in = in;
    e.kg = in / 128;
    const int rowBytes = in / 4;
    if (!e.w.create(rowBytes, out, gl::TextureFormat::R8UI)) return -1;
    e.w.uploadBytes(packed);
    if (!e.n.create(e.kg, out, gl::TextureFormat::R32F)) return -1;
    e.n.upload(norms);
    entries_.push_back(std::move(e));
    uploads++;
    return (int)entries_.size() - 1;
}

bool NGpu::beginActivation(int in) {
    const int w = in / 4;
    if (actIn_ != in) {
        if (!act_.create(w, 1, gl::TextureFormat::RGBA32F)) return false;
        actIn_ = in;
    }
    return true;
}

bool NGpu::ensurePartials(int kg, int out) {
    if (!part_ || partKg_ != kg || partRows_ != out) {
        part_ = std::make_unique<gl::FBO>();
        if (!part_->create(kg, out, gl::TextureFormat::RGBA32F)) { part_.reset(); return false; }
        partKg_ = kg;
        partRows_ = out;
    }
    return true;
}

bool NGpu::ensureOutput(int out) {
    if (!outFbo_ || outRows_ != out) {
        outFbo_ = std::make_unique<gl::FBO>();
        if (!outFbo_->create(1, out, gl::TextureFormat::RGBA32F)) { outFbo_.reset(); return false; }
        outRows_ = out;
        outBuf_.assign((size_t)out * 4, 0.f);
    }
    return true;
}

bool NGpu::matvec(int handle, const float* x, float* y) {
    if (!ok_ || handle < 0 || (size_t)handle >= entries_.size()) return false;
    Entry& e = entries_[handle];
    if (!beginActivation(e.in) || !ensurePartials(e.kg, e.out) || !ensureOutput(e.out)) return false;

    // Upload the rotated activation (already in [in/4 x 1] RGBA32F layout: 4 consecutive floats/texel).
    act_.upload(x);

    // Pass 1: per-(group, row) partial sums, already scaled by the group norm.
    dispatchShader(*partial_, *part_,
        {{0, &e.w, "texW"}, {1, &act_, "texX"}, {2, &e.n, "texN"}},
        [&](gl::Program& p) {
            glUniform4fv(glGetUniformLocation(p.id(), "cb"), 1, cb_);
        });

    // Pass 2: reduce the in/128 partials per row.
    dispatchShader(*reduce_, *outFbo_,
        {{0, &part_->colorTexture(), "texP"}},
        [&](gl::Program& p) { p.setInt("kg", e.kg); });

    outFbo_->colorTexture().download(outBuf_.data());
    for (int r = 0; r < e.out; r++) y[r] = outBuf_[(size_t)r * 4];
    calls++;
    return true;
}

} // namespace needle
