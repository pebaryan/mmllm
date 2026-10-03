#include "nmodel.h"
#include "ngpu.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace needle {

// ------------------------------------------------------------------------------------------------
// small persistent thread pool (the matrix-vector products are the hot path on a 2-core CPU)
// ------------------------------------------------------------------------------------------------
struct Model::Pool {
    explicit Pool(int n) : n_(std::max(1, n)) {
        for (int i = 1; i < n_; i++) workers_.emplace_back([this, i] { loop(i); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; gen_++; }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    int threads() const { return n_; }
    // f(tid, nthreads); the caller runs tid 0.
    void run(const std::function<void(int, int)>& f) {
        if (n_ == 1) { f(0, 1); return; }
        {
            std::lock_guard<std::mutex> lk(m_);
            task_ = &f;
            pending_ = n_ - 1;
            gen_++;
        }
        cv_.notify_all();
        f(0, n_);
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [this] { return pending_ == 0; });
    }

private:
    void loop(int tid) {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int, int)>* task;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
                task = task_;
            }
            (*task)(tid, n_);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--pending_ == 0) done_.notify_one();
            }
        }
    }
    int n_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(int, int)>* task_ = nullptr;
    int pending_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

// ------------------------------------------------------------------------------------------------
// numeric helpers
// ------------------------------------------------------------------------------------------------
static inline float dotf(const float* a, const float* b, int n) {
#if defined(__SSE2__)
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps(), s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4)));
        s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_loadu_ps(a + i + 8), _mm_loadu_ps(b + i + 8)));
        s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_loadu_ps(a + i + 12), _mm_loadu_ps(b + i + 12)));
    }
    for (; i + 4 <= n; i += 4) s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    s0 = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
    float t[4];
    _mm_storeu_ps(t, s0);
    float r = (t[0] + t[1]) + (t[2] + t[3]);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
#else
    float r = 0.f;
    for (int i = 0; i < n; i++) r += a[i] * b[i];
    return r;
#endif
}

static inline double nowSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline float sigmoidf(float x) { return (float)(1.0 / (1.0 + std::exp(-(double)x))); }

// x / sqrt(mean(x^2) + 1e-6)
static void rmsUnit(const float* x, int n, float* out) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float inv = (float)(1.0 / std::sqrt(ss / n + 1e-6));
    for (int i = 0; i < n; i++) out[i] = x[i] * inv;
}

// ZCRMSNorm: (1 + scale) * x / sqrt(mean(x^2) + 1e-6)
static void zcrms(const float* x, int n, const float* scale, float* out) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float inv = (float)(1.0 / std::sqrt(ss / n + 1e-6));
    for (int i = 0; i < n; i++) out[i] = (1.0f + scale[i]) * x[i] * inv;
}

// Symmetric int8 fake quantisation of one row (quantize.fake_quant with group = row length).
static int qround() {
    static const int m = std::getenv("NEEDLE_QROUND") ? std::atoi(std::getenv("NEEDLE_QROUND")) : 0;
    return m;
}
void fakeQuantRow(float* x, int n) {
    float amax = 0.f;
    for (int i = 0; i < n; i++) amax = std::max(amax, std::fabs(x[i]));
    const float scale = amax > 0.f ? amax / 127.0f : 1.0f;
    for (int i = 0; i < n; i++) {
        float q = qround() ? std::round(x[i] / scale) : std::nearbyint(x[i] / scale);   // 0: half to even (np.round), 1: half away (lroundf)
        q = std::min(127.f, std::max(qround() >= 2 ? -127.f : -128.f, q));
        x[i] = q * scale;
    }
}

static void softmaxInPlace(float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) m = std::max(m, x[i]);
    double s = 0;
    for (int i = 0; i < n; i++) { x[i] = (float)std::exp((double)x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] = (float)(x[i] / s);
}

// Hadamard block sizes: b = 1 << (bit_length(n-1) // 2)
static int hadaBlock(int n) {
    int bl = 0;
    for (int v = n - 1; v > 0; v >>= 1) bl++;
    return 1 << (bl / 2);
}

// z viewed as [a][b]; out[k][l] = sum_ij z[i][j] * A[i][k] * B[j][l]
static void kronApply(const float* z, int n, const float* A, const float* B, int ba, int bb, float* out) {
    (void)n;
    static thread_local std::vector<float> t1;
    t1.assign((size_t)ba * bb, 0.f);
    for (int i = 0; i < ba; i++)
        for (int j = 0; j < bb; j++) {
            const float zij = z[i * bb + j];
            if (zij == 0.f) continue;
            for (int l = 0; l < bb; l++) t1[i * bb + l] += zij * B[j * bb + l];
        }
    for (int k = 0; k < ba; k++) {
        float* o = out + k * bb;
        for (int l = 0; l < bb; l++) o[l] = 0.f;
        for (int i = 0; i < ba; i++) {                          // same summation order over i, contiguous in l
            const float a = A[i * ba + k];
            const float* t = &t1[i * bb];
            for (int l = 0; l < bb; l++) o[l] += a * t[l];
        }
    }
}

static void sinkhorn4(const double in[16], int n, float out[16]) {
    double lk[16];
    for (int i = 0; i < n * n; i++) lk[i] = in[i];
    for (int it = 0; it < 20; it++) {
        for (int r = 0; r < n; r++) {                           // normalise rows (axis -1)
            double m = lk[r * n];
            for (int c = 1; c < n; c++) m = std::max(m, lk[r * n + c]);
            double s = 0;
            for (int c = 0; c < n; c++) s += std::exp(lk[r * n + c] - m);
            const double lse = m + std::log(s);
            for (int c = 0; c < n; c++) lk[r * n + c] -= lse;
        }
        for (int c = 0; c < n; c++) {                           // normalise columns (axis -2)
            double m = lk[c];
            for (int r = 1; r < n; r++) m = std::max(m, lk[r * n + c]);
            double s = 0;
            for (int r = 0; r < n; r++) s += std::exp(lk[r * n + c] - m);
            const double lse = m + std::log(s);
            for (int r = 0; r < n; r++) lk[r * n + c] -= lse;
        }
    }
    for (int i = 0; i < n * n; i++) out[i] = (float)std::exp(lk[i]);
}

// ------------------------------------------------------------------------------------------------
bool Config::isGlobal(int l) const {
    return std::find(globalLayers.begin(), globalLayers.end(), l) != globalLayers.end();
}
int Config::siteOf(int layer) const {
    for (size_t s = 0; s < engramLayers.size(); s++) if (engramLayers[s] == layer) return (int)s;
    return -1;
}

// Rotate each 128-group of x with the normalised Walsh-Hadamard transform (what the weights were
// rotated with): w . x = norm * (codebook[idx] . (H x)) because H is symmetric and orthogonal.
static int qmode() {
    static const int m = std::getenv("NEEDLE_QMODE") ? std::atoi(std::getenv("NEEDLE_QMODE")) : 1;
    return m;
}
static void rotateInput(const float* x, int n, float* xr) {
    std::memcpy(xr, x, (size_t)n * sizeof(float));
    const float s = 1.0f / std::sqrt(128.0f);
    for (int g = 0; g < n; g += 128) {
        float* v = xr + g;
        for (int len = 1; len < 128; len <<= 1)
            for (int i = 0; i < 128; i += len << 1)
                for (int j = i; j < i + len; j++) {
                    const float a = v[j], b = v[j + len];
                    v[j] = a + b;
                    v[j + len] = a - b;
                }
        for (int i = 0; i < 128; i++) v[i] *= s;
        if (qmode() >= 1) fakeQuantRow(v, 128);
    }
}

// dot of one packed 2-bit row with the rotated activation
static inline float qdot2(const uint8_t* row, const float* norms, int nGroups, const float* xr,
                          const float (*lut)[4]) {
    float total = 0.f;
    for (int g = 0; g < nGroups; g++) {
        const uint8_t* b = row + (size_t)g * 32;
        const float* x = xr + (size_t)g * 128;
#if defined(__SSE2__)
        __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
        for (int j = 0; j < 32; j += 2) {
            a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_load_ps(lut[b[j]]), _mm_loadu_ps(x + 4 * j)));
            a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_load_ps(lut[b[j + 1]]), _mm_loadu_ps(x + 4 * j + 4)));
        }
        a0 = _mm_add_ps(a0, a1);
        float t[4];
        _mm_storeu_ps(t, a0);
        total += norms[g] * ((t[0] + t[1]) + (t[2] + t[3]));
#else
        float s = 0.f;
        for (int j = 0; j < 32; j++)
            for (int k = 0; k < 4; k++) s += lut[b[j]][k] * x[4 * j + k];
        total += norms[g] * s;
#endif
    }
    return total;
}

#if defined(__SSE2__)
// Two rows at once: every activation load is shared, so the loop does 3 loads per 2 multiplies instead of 4.
static inline void qdot2x2(const uint8_t* rowA, const uint8_t* rowB, const float* normsA, const float* normsB,
                           int nGroups, const float* xr, const float (*lut)[4], float& outA, float& outB) {
    float totA = 0.f, totB = 0.f;
    for (int g = 0; g < nGroups; g++) {
        const uint8_t* ba = rowA + (size_t)g * 32;
        const uint8_t* bb = rowB + (size_t)g * 32;
        const float* x = xr + (size_t)g * 128;
        __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps(), b0 = _mm_setzero_ps(), b1 = _mm_setzero_ps();
        for (int j = 0; j < 32; j += 2) {
            const __m128 x0 = _mm_loadu_ps(x + 4 * j), x1 = _mm_loadu_ps(x + 4 * j + 4);
            a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_load_ps(lut[ba[j]]), x0));
            b0 = _mm_add_ps(b0, _mm_mul_ps(_mm_load_ps(lut[bb[j]]), x0));
            a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_load_ps(lut[ba[j + 1]]), x1));
            b1 = _mm_add_ps(b1, _mm_mul_ps(_mm_load_ps(lut[bb[j + 1]]), x1));
        }
        a0 = _mm_add_ps(a0, a1);
        b0 = _mm_add_ps(b0, b1);
        float t[4], u[4];
        _mm_storeu_ps(t, a0);
        _mm_storeu_ps(u, b0);
        totA += normsA[g] * ((t[0] + t[1]) + (t[2] + t[3]));
        totB += normsB[g] * ((u[0] + u[1]) + (u[2] + u[3]));
    }
    outA = totA;
    outB = totB;
}
#endif

// Expand one packed 128-wide table row exactly as Cact::dequantCQ does: codebook * norm, then the normalised WHT.
void Model::engramRow(const EngramSite& es, size_t row, float* dst) const {
    const uint8_t* p = &es.tPacked[row * es.tRowBytes];
    const float norm = es.tNorms[row];
    const float* cb = cbTable_[es.tBits - 2];
    if (es.tBits == 2) for (int k = 0; k < 128; k++) dst[k] = cb[(p[k >> 2] >> ((k & 3) * 2)) & 3] * norm;
    else for (int k = 0; k < 128; k++) dst[k] = cb[(p[k >> 1] >> ((k & 1) * 4)) & 15] * norm;
    for (int len = 1; len < 128; len <<= 1)
        for (int i = 0; i < 128; i += len << 1)
            for (int j = i; j < i + len; j++) {
                const float a = dst[j], b = dst[j + len];
                dst[j] = a + b;
                dst[j + len] = a - b;
            }
    const float sc = 1.0f / std::sqrt(128.0f);
    for (int i = 0; i < 128; i++) dst[i] *= sc;
}

Model::Model() { setThreads((int)std::max(1u, std::min(4u, std::thread::hardware_concurrency()))); }
Model::~Model() = default;

void Model::setThreads(int n) { pool_.reset(new Pool(n)); }

void Model::gemv(const float* W, int out, int in, const float* x, float* y) {
    if ((long long)out * in < 150000 || pool_->threads() == 1) {
        for (int r = 0; r < out; r++) y[r] = dotf(W + (size_t)r * in, x, in);
        return;
    }
    pool_->run([&](int tid, int nt) {
        const int chunk = (out + nt - 1) / nt;
        const int r0 = tid * chunk, r1 = std::min(out, r0 + chunk);
        for (int r = r0; r < r1; r++) y[r] = dotf(W + (size_t)r * in, x, in);
    });
}

void Model::matvec(const Mat& M, const float* x, const float* xr, float* y) {
    if (!M.quant) { gemv(M.f.data(), M.out, M.in, x, y); return; }
    if (gpu_) {
        for (const auto& kv : gpuMap_)
            if (kv.first == &M) { if (gpu_->matvec(kv.second, xr, y)) return; break; }
    }
    const int nGroups = M.in / 128;
    const size_t rowBytes = (size_t)M.in / 4;
    auto rows = [&](int r0, int r1) {
        int r = r0;
#if defined(__SSE2__)
        for (; r + 1 < r1; r += 2)
            qdot2x2(&M.packed[(size_t)r * rowBytes], &M.packed[(size_t)(r + 1) * rowBytes], &M.norms[(size_t)r * nGroups],
                    &M.norms[(size_t)(r + 1) * nGroups], nGroups, xr, lut2_, y[r], y[r + 1]);
#endif
        for (; r < r1; r++)
            y[r] = qdot2(&M.packed[(size_t)r * rowBytes], &M.norms[(size_t)r * nGroups], nGroups, xr, lut2_);
    };
    if ((long long)M.out * M.in < 150000 || pool_->threads() == 1) { rows(0, M.out); return; }
    pool_->run([&](int tid, int nt) {
        const int chunk = (M.out + nt - 1) / nt;
        rows(tid * chunk, std::min(M.out, (tid + 1) * chunk));
    });
}

// Batched form of matvec: Y[m*out + r] = row_r . XR[m].  One weight row is decoded once and reused
// across the whole batch.  Matrices that live on the GPU are left to the per-token path (there is no
// batched GPU kernel - see the driver note in NEEDLE.md).
void Model::matvecBatch(const Mat& M, const float* XR, int count, float* Y) {
    if (count <= 0) return;
    if (!M.quant || gpu_) {
        for (int m = 0; m < count; m++) matvec(M, XR + (size_t)m * M.in, XR + (size_t)m * M.in, Y + (size_t)m * M.out);
        return;
    }
    const int nGroups = M.in / 128;
    const size_t rowBytes = (size_t)M.in / 4;
    auto rows = [&](int r0, int r1) {
        for (int r = r0; r < r1; r++) {
            const uint8_t* row = &M.packed[(size_t)r * rowBytes];
            const float* norms = &M.norms[(size_t)r * nGroups];
            for (int m = 0; m < count; m++)
                Y[(size_t)m * M.out + r] = qdot2(row, norms, nGroups, XR + (size_t)m * M.in, lut2_);
        }
    };
    if ((long long)M.out * M.in < 150000 || pool_->threads() == 1) { rows(0, M.out); return; }
    pool_->run([&](int tid, int nt) {
        const int chunk = (M.out + nt - 1) / nt;
        rows(tid * chunk, std::min(M.out, (tid + 1) * chunk));
    });
}

// ------------------------------------------------------------------------------------------------
// Optional GPU offload of the 2-bit CQ matrix-vector products. The packed weights (and their
// per-group norms) are uploaded once, in exactly the form the CPU kernel consumes, so the GPU
// reproduces the same arithmetic; everything else in the per-token graph stays on the CPU.
// ------------------------------------------------------------------------------------------------
bool Model::enableGpu(NGpu* gpu) {
    if (!gpu || !gpu->ok()) return false;
    gpu->setCodebook(cbTable_[0]);                     // the same 2-bit codebook the CPU kernel uses
    gpuMap_.clear();
    auto add = [&](const Mat& m) -> bool {
        if (!m.quant) return true;                     // float matrices stay on the CPU (see NEEDLE.md)
        if (m.in % 128 != 0 || m.packed.size() != (size_t)m.out * (m.in / 4)) return false;
        const int h = gpu->uploadQuant(m.packed.data(), m.norms.data(), m.out, m.in);
        if (h < 0) return false;
        gpuMap_.emplace_back(&m, h);
        return true;
    };
    for (const Layer& l : layers_)
        if (!add(l.qProj) || !add(l.kProj) || !add(l.vProj) || !add(l.gateProj) || !add(l.outProj)) return false;
    for (const EngramSite& s : sites_)
        if (!add(s.keyProj) || !add(s.valueProj)) return false;
    gpu_ = gpu;
    return true;
}

bool Model::gpuSelfCheck(std::string& report) {
    if (!gpu_) { report = "GPU not enabled"; return false; }
    const int D = cfg_.D;
    std::vector<float> x(D), xr(D), ycpu, ygpu;
    for (int i = 0; i < D; i++) x[i] = (float)(std::sin((double)i * 0.1031) * 0.9 + 0.2 * ((i % 7) - 3));
    rotateInput(x.data(), D, xr.data());               // exactly what the model feeds the projections

    int checked = 0;
    double worstAbs = 0, worstRel = 0, scaleMax = 0;
    const void* worstMat = nullptr;
    auto check = [&](const Mat& m) -> bool {
        if (!m.quant) return true;                     // not offloaded
        if (m.in != D) return true;                    // only the projection-shaped matrices are checked
        ycpu.assign(m.out, 0.f);
        ygpu.assign(m.out, 0.f);
        NGpu* save = gpu_;
        gpu_ = nullptr;                                // force the CPU kernel
        matvec(m, x.data(), xr.data(), ycpu.data());
        gpu_ = save;
        matvec(m, x.data(), xr.data(), ygpu.data());   // now the GPU path
        double md = 0, mx = 0;
        for (int r = 0; r < m.out; r++) {
            md = std::max(md, (double)std::fabs(ycpu[r] - ygpu[r]));
            mx = std::max(mx, (double)std::fabs(ycpu[r]));
        }
        if (md > worstAbs) { worstAbs = md; worstMat = &m; }
        scaleMax = std::max(scaleMax, mx);
        worstRel = std::max(worstRel, mx > 0 ? md / mx : 0.0);
        checked++;
        return true;
    };
    for (const Layer& l : layers_)
        if (!check(l.qProj) || !check(l.kProj) || !check(l.vProj) || !check(l.gateProj) || !check(l.outProj)) return false;
    for (const EngramSite& s : sites_)
        if (!check(s.keyProj) || !check(s.valueProj)) return false;

    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "%d matrices checked: worst |CPU-GPU| = %.3e (at %p), largest |value| = %.3e, worst relative = %.3e",
                  checked, worstAbs, worstMat, scaleMax, worstRel);
    report = buf;
    return worstRel < 2e-4;
}

bool Model::loadMat(const Cact& c, size_t index, Mat& m) {
    CqMatrix cq;
    if (c.record(index).bits == 2 && c.cqMatrix(index, cq)) {
        m.out = cq.out; m.in = cq.in; m.quant = true;
        m.packed = std::move(cq.packed); m.norms = std::move(cq.norms);
        return true;
    }
    const TensorRecord& r = c.record(index);
    m.out = (int)r.shape[0]; m.in = (int)r.shape[1]; m.quant = false;
    m.f = c.floats(index);
    return m.f.size() == (size_t)m.out * m.in;
}

bool Model::load(const Cact& c, std::string& err) {
    for (int b = 2; b <= 4; b++)
        if (const float* cb = c.codebook(b)) for (int i = 0; i < (1 << b); i++) cbTable_[b - 2][i] = cb[i];
    if (const float* cb = c.codebook(2)) {
        for (int b = 0; b < 256; b++)
            for (int j = 0; j < 4; j++) lut2_[b][j] = cb[(b >> (2 * j)) & 3];
    }
    const CactHeader& h = c.header();
    cfg_.D = (int)h.dModel; cfg_.L = (int)h.numLayers; cfg_.H = (int)h.numHeads;
    cfg_.KVH = (int)h.numKvHeads; cfg_.QK = (int)h.qkHeadDim; cfg_.VH = (int)h.vHeadDim;
    cfg_.lanes = (int)h.mhcLanes; cfg_.taps = (int)h.qkvConvTaps;
    cfg_.vocab = (int)(h.outVocab ? h.outVocab : h.vocab);
    cfg_.hadaN = (int)h.hadaN;
    cfg_.engramSub = (int)h.engramSubDim; cfg_.engramTables = (int)h.numEngramTables;
    cfg_.engramSlots = (int)h.engramSlots; cfg_.engramTaps = (int)h.engramConvTaps;
    cfg_.engramDil = (int)h.engramConvDilation; cfg_.engramSeedHeads = (int)h.engramSeedHeads;
    cfg_.window = (int)h.slidingWindow; cfg_.rope = h.ropeTheta;
    cfg_.orders.clear();
    for (uint32_t i = 0; i < h.numEngramOrders; i++) cfg_.orders.push_back((int)h.engramOrders[i]);
    cfg_.engramHeads = cfg_.orders.empty() ? 0 : cfg_.engramTables / (int)cfg_.orders.size();
    cfg_.globalLayers.clear(); cfg_.engramLayers.clear();
    for (int l = 0; l < cfg_.L; l++) if (c.isGlobalLayer(l)) cfg_.globalLayers.push_back(l);
    for (uint32_t i = 0; i < h.numEngramSites; i++) cfg_.engramLayers.push_back((int)h.engramSites[i]);

    if (cfg_.taps < 1) { err = "archives without q/k/v conv taps are not supported"; return false; }
    if (cfg_.D != 768 && cfg_.D <= 0) { err = "bad d_model"; return false; }
    const size_t sites = cfg_.engramLayers.size();
    const size_t needed = 1 + (size_t)cfg_.L * 27 + 9 + 2 + sites * 4 + 1;
    if (c.numTensors() < needed) { err = "archive has too few tensors for this architecture"; return false; }

    size_t k = 0;
    auto take = [&]() { return c.floats(k++); };
    emb_ = take();
    if (emb_.size() != (size_t)h.vocab * cfg_.D) { err = "embedding shape mismatch"; return false; }

    layers_.assign(cfg_.L, Layer());
    for (int i = 0; i < cfg_.L; i++) {
        Layer& l = layers_[i];
        l.normIn = take();
        if (!loadMat(c, k++, l.qProj) || !loadMat(c, k++, l.kProj) || !loadMat(c, k++, l.vProj)) { err = "bad q/k/v matrix"; return false; }
        l.qTaps = take(); l.kTaps = take(); l.vTaps = take();
        l.qNorm = take(); l.kNorm = take();
        if (!loadMat(c, k++, l.gateProj) || !loadMat(c, k++, l.outProj)) { err = "bad gate/out matrix"; return false; }
        l.postNorm = take();
        std::vector<float> g = take(); l.attnGate = g.empty() ? 0.f : g[0];
        l.preHada = take(); l.d1 = take(); l.d2 = take(); l.b2 = take(); l.d3 = take(); l.d4 = take();
        l.w1a = take(); l.w1b = take(); l.w2a = take(); l.w2b = take(); l.w3a = take(); l.w3b = take();
        l.condV = take(); l.condU = take();
    }
    mhcAPre_ = take(); mhcAPost_ = take(); mhcARes_ = take();
    mhcBPre_ = take(); mhcBPost_ = take(); mhcBRes_ = take();
    phiPre_ = take(); phiPost_ = take(); phiRes_ = take();
    for (std::vector<int>* dst : {&hadaP1_, &hadaP2_}) {
        std::vector<float> p = take();
        dst->assign(p.size(), 0);
        for (size_t i = 0; i < p.size(); i++) (*dst)[i] = (int)p[i];
    }
    sites_.assign(sites, EngramSite());
    for (size_t s = 0; s < sites; s++) {
        {
            const size_t ti = k++;
            const TensorRecord& tr = c.record(ti);
            CqMatrix cq;
            if (tr.dtype == DT_CQ && tr.ndim == 2 && tr.shape[1] == 128 && (tr.bits == 2 || tr.bits == 4) && c.cqMatrix(ti, cq)) {
                sites_[s].tPacked = std::move(cq.packed);
                sites_[s].tNorms = std::move(cq.norms);
                sites_[s].tBits = (int)tr.bits;
                sites_[s].tRowBytes = 128 * (int)tr.bits / 8;
            } else {
                sites_[s].tables = c.floats(ti);
            }
        }
        if (!loadMat(c, k++, sites_[s].keyProj) || !loadMat(c, k++, sites_[s].valueProj)) { err = "bad engram matrix"; return false; }
        sites_[s].taps = take();
    }
    finalNorm_ = take();
    // ---- optional probe heads: heads.manifest (one code per head), then six tensors per head
    //      [probes, gain, query, row_bias, proj, bias]; the router head appends one more (calibration).
    if (k + 1 < c.numTensors()) {
        const std::vector<float> manifest = take();
        for (float codeF : manifest) {
            const int code = (int)(codeF + 0.5f);
            std::vector<float> probes = take(), gain = take(), query = take(), rowBias = take(), proj = take(), bias = take();
            if (code == 3) take();
            if (code == 2) {
                headProbes_ = std::move(probes); headGain_ = std::move(gain); headQuery_ = std::move(query);
                headRowBias_ = std::move(rowBias); headProj_ = std::move(proj);
                headBias_ = bias.empty() ? 0.f : bias[0];
            }
        }
        const int cells = cfg_.L + 1;
        if (!headProj_.empty()) {
            headK_ = (int)(headGain_.size() / cells);
            headQ_ = (int)(headQuery_.size() / cfg_.D);
            if (headProbes_.size() != (size_t)cells * headK_ * cfg_.D || headProj_.size() != (size_t)headQ_ * cfg_.D)
                headProj_.clear();                          // unexpected geometry: leave the head off
        }
    }

    ropeInv_.resize(cfg_.QK / 2);
    for (int i = 0; i < cfg_.QK / 2; i++)
        ropeInv_[i] = (float)(1.0 / std::pow((double)cfg_.rope, (double)(2 * i) / cfg_.QK));

    if (hadaP1_.size() != (size_t)cfg_.hadaN || hadaP2_.size() != (size_t)cfg_.hadaN) {
        err = "hadamard permutation size mismatch";
        return false;
    }
    return true;
}

State Model::newState() const {
    State st;
    st.layers.resize(cfg_.L);
    for (auto& lc : st.layers) {
        lc.prevQ.assign((size_t)2 * cfg_.H * cfg_.QK, 0.f);
        lc.prevK.assign((size_t)2 * cfg_.KVH * cfg_.QK, 0.f);
        lc.prevV.assign((size_t)2 * cfg_.KVH * cfg_.VH, 0.f);
    }
    st.engramV.resize(cfg_.engramLayers.size());
    if (!headProj_.empty()) {
        const size_t pk = (size_t)(cfg_.L + 1) * headK_;
        st.probeM.assign(pk, -1e30f);
        st.probeS.assign(pk, 0.f);
        st.probeR.assign(pk * cfg_.D, 0.f);
    }
    return st;
}

// ------------------------------------------------------------------------------------------------
void Model::step(State& st, int token, bool quant, float* logits) {
    const Config& c = cfg_;
    double lastT = profiling ? nowSeconds() : 0.0;
    auto lap = [&](double& acc) { if (!profiling) return; const double t = nowSeconds(); acc += t - lastT; lastT = t; };
    st.tokens.push_back(token);
    const int t = (int)st.tokens.size() - 1;
    const int D = c.D, n = c.lanes;
    const int qDim = c.H * c.QK, kDim = c.KVH * c.QK, vDim = c.KVH * c.VH, oDim = c.H * c.VH;

    // ---- engram hash indices for this position (previous tokens read as 0 before the start)
    uint32_t eidx[16];
    {
        int tbl = 0;
        const int stride = c.engramSeedHeads ? c.engramSeedHeads : c.engramHeads;
        for (size_t oi = 0; oi < c.orders.size(); oi++)
            for (int hd = 0; hd < c.engramHeads; hd++, tbl++) {
                uint32_t acc = 0x9E3779B9u * (uint32_t)(oi * stride + hd + 1);
                for (int j = 0; j < c.orders[oi]; j++) {
                    const uint32_t tok = (t - j >= 0) ? (uint32_t)st.tokens[t - j] : 0u;
                    acc = (acc ^ tok) * 0x01000193u;
                }
                acc ^= acc >> 15;
                eidx[tbl] = acc % (uint32_t)c.engramSlots;
            }
    }

    // ---- lanes
    std::vector<float> stream((size_t)n * D), nxs((size_t)n * D), newStream((size_t)n * D);
    {
        const float s = std::sqrt((float)D);
        for (int i = 0; i < D; i++) stream[i] = emb_[(size_t)token * D + i] * s;
        for (int j = 1; j < n; j++) std::memcpy(&stream[(size_t)j * D], &stream[0], D * sizeof(float));
    }
    const bool probing = !st.probeM.empty();
    if (probing) probeUpdate(st, 0, &stream[0]);

    std::vector<float> u(D), x(D), skip(D), xn(D), tmp(D), q(qDim), kk(kDim), vv(vDim), attn(oDim), gate(oDim),
        aOut(D), blk(D), y(D), kE(D), vE(D), vRaw(D), e(D), er(D), xnr(D), attnr(oDim);
    std::vector<float> cosv(c.QK / 2), sinv(c.QK / 2);
    for (int i = 0; i < c.QK / 2; i++) {
        const double ang = (double)t * ropeInv_[i];
        cosv[i] = (float)std::cos(ang);
        sinv[i] = (float)std::sin(ang);
    }
    const int nC = n * D;
    const int hb = hadaBlock(c.hadaN);
    std::vector<float> z(c.hadaN), z2(c.hadaN), cond(c.hadaN);
    std::vector<float> scores;

    for (int i = 0; i < c.L; i++) {
        const Layer& L = layers_[i];
        const int lane = i % n;

        // multi-lane hyper-connection inputs
        rmsUnit(stream.data(), nC, nxs.data());
        if (quant) fakeQuantRow(nxs.data(), nC);
        float hpre[4], hpost[4], res[16];
        for (int j = 0; j < n; j++) {
            const float pre = mhcAPre_[i] * dotf(&phiPre_[(size_t)(i * n + j) * nC], nxs.data(), nC)
                              + mhcBPre_[i * n + j] + (j == lane ? 4.f : -4.f);
            hpre[j] = sigmoidf(pre);
            const float post = mhcAPost_[i] * dotf(&phiPost_[(size_t)(i * n + j) * nC], nxs.data(), nC)
                               + mhcBPost_[i * n + j] + (j == lane ? 0.f : -4.f);
            hpost[j] = 2.f * sigmoidf(post);
        }
        for (int a = 0; a < n * n; a++) res[a] = dotf(&phiRes_[(size_t)(i * n * n + a) * nC], nxs.data(), nC);

        for (int d = 0; d < D; d++) {
            float s = 0.f;
            for (int j = 0; j < n; j++) s += hpre[j] * stream[(size_t)j * D + d];
            u[d] = s;
        }

        lap(profile.mhc);
        // ---- block
        x = u;
        const int site = c.siteOf(i);
        if (site >= 0) {
            const EngramSite& es = sites_[site];
            // gather: concatenation of one sub-vector per table (zero if the n-gram does not exist yet)
            int tb = 0;
            for (size_t oi = 0; oi < c.orders.size(); oi++)
                for (int hd = 0; hd < c.engramHeads; hd++, tb++) {
                    float* dst = &e[(size_t)tb * c.engramSub];
                    if (t >= c.orders[oi] - 1) {
                        const size_t row = (size_t)tb * c.engramSlots + eidx[tb];
                        if (es.tBits) engramRow(es, row, dst);
                        else std::memcpy(dst, &es.tables[row * c.engramSub], c.engramSub * sizeof(float));
                    }
                    else
                        std::memset(dst, 0, c.engramSub * sizeof(float));
                }
            if (quant && qmode() != 1) fakeQuantRow(e.data(), D);
            rotateInput(e.data(), D, er.data());
            matvec(es.keyProj, e.data(), er.data(), kE.data());
            matvec(es.valueProj, e.data(), er.data(), vRaw.data());
            std::vector<float>& hist = st.engramV[site];
            hist.insert(hist.end(), vRaw.begin(), vRaw.end());
            for (int d = 0; d < D; d++) {
                float s = es.taps[d] * vRaw[d];
                for (int j = 1; j < c.engramTaps; j++) {
                    const int tp = t - j * c.engramDil;
                    if (tp >= 0) s += es.taps[(size_t)j * D + d] * hist[(size_t)tp * D + d];
                }
                vE[d] = s;
            }
            rmsUnit(x.data(), D, tmp.data());
            rmsUnit(kE.data(), D, xn.data());
            const float alpha = sigmoidf(dotf(tmp.data(), xn.data(), D) / std::sqrt((float)D));
            for (int d = 0; d < D; d++) x[d] += alpha * vE[d];
        }
        lap(profile.engram);
        skip = x;

        zcrms(x.data(), D, L.normIn.data(), xn.data());
        if (quant && qmode() != 1) fakeQuantRow(xn.data(), D);
        std::vector<float> qraw(qDim), kraw(kDim), vraw(vDim);
        rotateInput(xn.data(), D, xnr.data());
        matvec(L.qProj, xn.data(), xnr.data(), qraw.data());
        matvec(L.kProj, xn.data(), xnr.data(), kraw.data());
        matvec(L.vProj, xn.data(), xnr.data(), vraw.data());

        State::LayerCache& lc = st.layers[i];
        auto tapMix = [&](const std::vector<float>& taps, const std::vector<float>& raw, std::vector<float>& prev,
                          int dim, float* outv) {
            for (int d = 0; d < dim; d++) {
                float s = taps[d] * raw[d];
                if (c.taps > 1) s += taps[(size_t)dim + d] * prev[d];            // t-1
                if (c.taps > 2) s += taps[(size_t)2 * dim + d] * prev[(size_t)dim + d];   // t-2
                outv[d] = s;
            }
            // shift history: [t-1, t-2] <- [raw, t-1]
            std::memmove(&prev[(size_t)dim], &prev[0], (size_t)dim * sizeof(float));
            std::memcpy(&prev[0], raw.data(), (size_t)dim * sizeof(float));
        };
        tapMix(L.qTaps, qraw, lc.prevQ, qDim, q.data());
        tapMix(L.kTaps, kraw, lc.prevK, kDim, kk.data());
        tapMix(L.vTaps, vraw, lc.prevV, vDim, vv.data());

        for (int hd = 0; hd < c.H; hd++) zcrms(&q[(size_t)hd * c.QK], c.QK, L.qNorm.data(), &q[(size_t)hd * c.QK]);
        for (int hd = 0; hd < c.KVH; hd++) zcrms(&kk[(size_t)hd * c.QK], c.QK, L.kNorm.data(), &kk[(size_t)hd * c.QK]);

        const int half = c.QK / 2;
        auto rope = [&](float* v) {
            for (int d = 0; d < half; d++) {
                const float x1 = v[d], x2 = v[d + half];
                v[d] = x1 * cosv[d] - x2 * sinv[d];
                v[d + half] = x2 * cosv[d] + x1 * sinv[d];
            }
        };
        for (int hd = 0; hd < c.H; hd++) rope(&q[(size_t)hd * c.QK]);
        for (int hd = 0; hd < c.KVH; hd++) rope(&kk[(size_t)hd * c.QK]);
        if (quant) {
            for (int hd = 0; hd < c.H; hd++) fakeQuantRow(&q[(size_t)hd * c.QK], c.QK);
            for (int hd = 0; hd < c.KVH; hd++) fakeQuantRow(&kk[(size_t)hd * c.QK], c.QK);
            for (int hd = 0; hd < c.KVH; hd++) fakeQuantRow(&vv[(size_t)hd * c.VH], c.VH);
        }
        lc.k.insert(lc.k.end(), kk.begin(), kk.end());
        lc.v.insert(lc.v.end(), vv.begin(), vv.end());

        lap(profile.qkv);
        const bool global = c.isGlobal(i);
        const int lo = (global || !c.window) ? 0 : std::max(0, t - c.window + 1);
        const int S = t - lo + 1;
        scores.resize(S);
        const int group = c.H / c.KVH;
        const float scale = 1.0f / std::sqrt((float)c.QK);
        for (int hd = 0; hd < c.H; hd++) {
            const int kh = hd / group;
            const float* qh = &q[(size_t)hd * c.QK];
            for (int s = 0; s < S; s++)
                scores[s] = dotf(qh, &lc.k[((size_t)(lo + s) * c.KVH + kh) * c.QK], c.QK) * scale;
            softmaxInPlace(scores.data(), S);
            float* oh = &attn[(size_t)hd * c.VH];
            for (int d = 0; d < c.VH; d++) oh[d] = 0.f;
            for (int s = 0; s < S; s++) {
                const float p = scores[s];
                const float* vs = &lc.v[((size_t)(lo + s) * c.KVH + kh) * c.VH];
                for (int d = 0; d < c.VH; d++) oh[d] += p * vs[d];
            }
        }
        lap(profile.attn);
        matvec(L.gateProj, xn.data(), xnr.data(), gate.data());
        for (int d = 0; d < oDim; d++) attn[d] *= sigmoidf(gate[d]);
        if (quant && qmode() != 1) fakeQuantRow(attn.data(), oDim);
        rotateInput(attn.data(), oDim, attnr.data());
        matvec(L.outProj, attn.data(), attnr.data(), aOut.data());
        zcrms(aOut.data(), D, L.postNorm.data(), tmp.data());
        const float ag = sigmoidf(L.attnGate);
        for (int d = 0; d < D; d++) x[d] = skip[d] + ag * tmp[d];
        skip = x;

        lap(profile.gateout);
        // ---- Hadamard MLP
        zcrms(x.data(), D, L.preHada.data(), xn.data());
        {
            float logit8[8];
            for (int r = 0; r < 8; r++) logit8[r] = 0.f;
            for (int d = 0; d < D; d++)
                for (int r = 0; r < 8; r++) logit8[r] += xn[d] * L.condV[(size_t)d * 8 + r];
            softmaxInPlace(logit8, 8);
            for (int m = 0; m < c.hadaN; m++) {
                float s = 0.f;
                for (int r = 0; r < 8; r++) s += logit8[r] * L.condU[(size_t)r * c.hadaN + m];
                cond[m] = 1.f + s;
            }
            std::fill(z.begin(), z.end(), 0.f);
            for (int d = 0; d < D; d++) z[d] = xn[d];
            for (int m = 0; m < c.hadaN; m++) z[m] *= L.d1[m];
            kronApply(z.data(), c.hadaN, L.w1a.data(), L.w1b.data(), hb, hb, z2.data());
            for (int m = 0; m < c.hadaN; m++) z[m] = z2[hadaP1_[m]];
            for (int m = 0; m < c.hadaN; m++) {
                const float v = L.d2[m] * cond[m] * z[m] + L.b2[m];
                z[m] = v * sigmoidf(v);                                     // silu
            }
            kronApply(z.data(), c.hadaN, L.w2a.data(), L.w2b.data(), hb, hb, z2.data());
            for (int m = 0; m < c.hadaN; m++) z[m] = z2[hadaP2_[m]] * L.d3[m];
            kronApply(z.data(), c.hadaN, L.w3a.data(), L.w3b.data(), hb, hb, z2.data());
            for (int d = 0; d < D; d++) blk[d] = skip[d] + L.d4[d] * z2[d];
        }

        lap(profile.hada);
        // ---- redistribute the block's delta over the lanes
        for (int d = 0; d < D; d++) y[d] = blk[d] - u[d];
        double lg[16];
        for (int a = 0; a < n * n; a++) lg[a] = (double)mhcARes_[i] * res[a] + mhcBRes_[(size_t)i * n * n + a];
        float hres[16];
        sinkhorn4(lg, n, hres);
        for (int a = 0; a < n; a++)
            for (int d = 0; d < D; d++) {
                float s = hpost[a] * y[d];
                for (int b = 0; b < n; b++) s += hres[a * n + b] * stream[(size_t)b * D + d];
                newStream[(size_t)a * D + d] = s;
            }
        stream.swap(newStream);
        if (probing) {
            for (int d = 0; d < D; d++) {
                float s = 0.f;
                for (int j = 0; j < n; j++) s += stream[(size_t)j * D + d];
                tmp[d] = s / n;
            }
            probeUpdate(st, i + 1, tmp.data());
        }
        lap(profile.mhc2);
    }
    profile.tokens++;

    if (!logits) return;
    // ---- final: lane mean, norm, tied head
    for (int d = 0; d < D; d++) {
        float s = 0.f;
        for (int j = 0; j < n; j++) s += stream[(size_t)j * D + d];
        tmp[d] = s / n;
    }
    zcrms(tmp.data(), D, finalNorm_.data(), xn.data());
    if (quant) fakeQuantRow(xn.data(), D);
    gemv(emb_.data(), c.vocab, D, xn.data(), logits);
    lap(profile.head);
}

// ------------------------------------------------------------------------------------------------
// Batched prefill. Same graph as step(), but the layer loop is outermost so that every 2-bit
// projection in a layer runs once as a GEMM over the whole batch instead of once per token.
//
// Why layer-outer order reproduces the per-token order exactly:
//   - attention: sequentially, token m only starts after tokens 0..m-1 have finished EVERY layer, so
//     when m reaches layer i the layer-i KV cache holds tokens 0..m.  Running the whole batch through
//     layer i and then capping token m's attention at position m gives the same set.
//   - the q/k/v conv taps read the raw projections of the previous tokens AT THE SAME LAYER, so inside
//     a batch they come from the batch, falling back to the carried-in history for the first taps.
//   - the engram value history is appended for the whole batch in order before any tap is read (the
//     taps only ever look backwards).
//   - the confidence probe pool is updated in strict token order from buffered cell vectors.
// ------------------------------------------------------------------------------------------------
void Model::prefill(State& st, const int* toks, int M, bool quant) {
    if (M <= 0) return;
    if (M == 1) { step(st, toks[0], quant, nullptr); return; }   // the carried-history edge cases
    double lastT = profiling ? nowSeconds() : 0.0;
    auto lap = [&](double& acc) { if (!profiling) return; const double t = nowSeconds(); acc += t - lastT; lastT = t; };
    const Config& c = cfg_;
    const int D = c.D, n = c.lanes;
    const int qDim = c.H * c.QK, kDim = c.KVH * c.QK, vDim = c.KVH * c.VH, oDim = c.H * c.VH;
    const int nC = n * D, hb = hadaBlock(c.hadaN);
    const int t0 = (int)st.tokens.size();
    const bool probing = !st.probeM.empty();
    for (int m = 0; m < M; m++) st.tokens.push_back(toks[m]);

    // ---- per-token engram hash indices (only ever looks backwards, so the batch can be pre-indexed)
    std::vector<uint32_t> eidxAll((size_t)M * 16, 0u);
    {
        int tbl = 0;
        const int stride = c.engramSeedHeads ? c.engramSeedHeads : c.engramHeads;
        for (size_t oi = 0; oi < c.orders.size(); oi++)
            for (int hd = 0; hd < c.engramHeads; hd++, tbl++)
                for (int m = 0; m < M; m++) {
                    const int t = t0 + m;
                    uint32_t acc = 0x9E3779B9u * (uint32_t)(oi * stride + hd + 1);
                    for (int j = 0; j < c.orders[oi]; j++) {
                        const uint32_t tk = (t - j >= 0) ? (uint32_t)st.tokens[t - j] : 0u;
                        acc = (acc ^ tk) * 0x01000193u;
                    }
                    acc ^= acc >> 15;
                    eidxAll[(size_t)m * 16 + tbl] = acc % (uint32_t)c.engramSlots;
                }
    }

    // ---- lane streams
    std::vector<float> stream((size_t)M * nC), nxs((size_t)M * nC), scr((size_t)M * nC);
    {
        const float s = std::sqrt((float)D);
        for (int m = 0; m < M; m++) {
            float* dst = &stream[(size_t)m * nC];
            for (int i = 0; i < D; i++) dst[i] = emb_[(size_t)toks[m] * D + i] * s;
            for (int j = 1; j < n; j++) std::memcpy(dst + (size_t)j * D, dst, D * sizeof(float));
        }
    }
    // Probe cell 0 is lane 0 of the stream (not the lane mean - see step()); cells 1..L are lane means.
    std::vector<float> cells;
    if (probing) {
        cells.assign((size_t)M * (c.L + 1) * D, 0.f);
        for (int m = 0; m < M; m++)
            std::memcpy(&cells[(size_t)m * (c.L + 1) * D], &stream[(size_t)m * nC], D * sizeof(float));
    }

    std::vector<float> ub((size_t)M * D), xb((size_t)M * D), skipb((size_t)M * D),
        xnb((size_t)M * D), xnrb((size_t)M * D);
    std::vector<float> qrawb((size_t)M * qDim), krawb((size_t)M * kDim), vrawb((size_t)M * vDim);
    std::vector<float> qb((size_t)M * qDim), kbb((size_t)M * kDim), vbb((size_t)M * vDim);
    std::vector<float> attnb((size_t)M * oDim), attnrb((size_t)M * oDim), gateb((size_t)M * oDim);
    std::vector<float> aoutb((size_t)M * D), blkb((size_t)M * D), yb((size_t)M * D);
    std::vector<float> eb((size_t)M * D), erb((size_t)M * D), kbE((size_t)M * D), vrE((size_t)M * D), vEb((size_t)M * D);
    std::vector<float> hpostb((size_t)M * n), resb((size_t)M * n * n);
    std::vector<float> cosv(c.QK / 2), sinv(c.QK / 2), scores;
    std::vector<float> z(c.hadaN), z2(c.hadaN), cond(c.hadaN);

    for (int i = 0; i < c.L; i++) {
        const Layer& L = layers_[i];
        const int lane = i % n;

        // ---- multi-lane hyper-connections (per token)
        for (int m = 0; m < M; m++) {
            const float* s = &stream[(size_t)m * nC];
            float* ns = &nxs[(size_t)m * nC];
            rmsUnit(s, nC, ns);
            if (quant) fakeQuantRow(ns, nC);
            float hpre[4];
            for (int j = 0; j < n; j++) {
                const float pre = mhcAPre_[i] * dotf(&phiPre_[(size_t)(i * n + j) * nC], ns, nC)
                                  + mhcBPre_[i * n + j] + (j == lane ? 4.f : -4.f);
                hpre[j] = sigmoidf(pre);
                const float post = mhcAPost_[i] * dotf(&phiPost_[(size_t)(i * n + j) * nC], ns, nC)
                                   + mhcBPost_[i * n + j] + (j == lane ? 0.f : -4.f);
                hpostb[(size_t)m * n + j] = 2.f * sigmoidf(post);
            }
            for (int a = 0; a < n * n; a++) resb[(size_t)m * n * n + a] = dotf(&phiRes_[(size_t)(i * n * n + a) * nC], ns, nC);
            float* u = &ub[(size_t)m * D];
            for (int d = 0; d < D; d++) {
                float acc = 0.f;
                for (int j = 0; j < n; j++) acc += hpre[j] * s[(size_t)j * D + d];
                u[d] = acc;
            }
        }
        lap(profile.mhc);
        for (int m = 0; m < M; m++) std::memcpy(&xb[(size_t)m * D], &ub[(size_t)m * D], D * sizeof(float));

        // ---- engram block
        const int site = c.siteOf(i);
        if (site >= 0) {
            const EngramSite& es = sites_[site];
            for (int m = 0; m < M; m++) {
                const int t = t0 + m;
                const uint32_t* ei = &eidxAll[(size_t)m * 16];
                int tb = 0;
                for (size_t oi = 0; oi < c.orders.size(); oi++)
                    for (int hd = 0; hd < c.engramHeads; hd++, tb++) {
                        float* dst = &eb[(size_t)m * D + (size_t)tb * c.engramSub];
                        if (t >= c.orders[oi] - 1) {
                            const size_t row = (size_t)tb * c.engramSlots + ei[tb];
                            if (es.tBits) engramRow(es, row, dst);
                            else std::memcpy(dst, &es.tables[row * c.engramSub], c.engramSub * sizeof(float));
                        } else {
                            std::memset(dst, 0, c.engramSub * sizeof(float));
                        }
                    }
                if (quant && qmode() != 1) fakeQuantRow(&eb[(size_t)m * D], D);
                rotateInput(&eb[(size_t)m * D], D, &erb[(size_t)m * D]);
            }
            matvecBatch(es.keyProj, erb.data(), M, kbE.data());
            matvecBatch(es.valueProj, erb.data(), M, vrE.data());
            std::vector<float>& hist = st.engramV[site];
            for (int m = 0; m < M; m++)
                hist.insert(hist.end(), vrE.begin() + (size_t)m * D, vrE.begin() + (size_t)(m + 1) * D);
            for (int m = 0; m < M; m++) {
                const int t = t0 + m;
                const float* vr = &vrE[(size_t)m * D];
                float* vE = &vEb[(size_t)m * D];
                for (int d = 0; d < D; d++) {
                    float acc = es.taps[d] * vr[d];
                    for (int j = 1; j < c.engramTaps; j++) {
                        const int tp = t - j * c.engramDil;
                        if (tp >= 0) acc += es.taps[(size_t)j * D + d] * hist[(size_t)tp * D + d];
                    }
                    vE[d] = acc;
                }
                rmsUnit(&xb[(size_t)m * D], D, &xnb[(size_t)m * D]);
                rmsUnit(&kbE[(size_t)m * D], D, &xnrb[(size_t)m * D]);
                const float alpha = sigmoidf(dotf(&xnb[(size_t)m * D], &xnrb[(size_t)m * D], D) / std::sqrt((float)D));
                for (int d = 0; d < D; d++) xb[(size_t)m * D + d] += alpha * vE[d];
            }
        }
        lap(profile.engram);
        for (int m = 0; m < M; m++) std::memcpy(&skipb[(size_t)m * D], &xb[(size_t)m * D], D * sizeof(float));

        // ---- q/k/v projections (batched) and the conv taps (per token)
        for (int m = 0; m < M; m++) {
            zcrms(&xb[(size_t)m * D], D, L.normIn.data(), &xnb[(size_t)m * D]);
            if (quant && qmode() != 1) fakeQuantRow(&xnb[(size_t)m * D], D);
            rotateInput(&xnb[(size_t)m * D], D, &xnrb[(size_t)m * D]);
        }
        matvecBatch(L.qProj, xnrb.data(), M, qrawb.data());
        matvecBatch(L.kProj, xnrb.data(), M, krawb.data());
        matvecBatch(L.vProj, xnrb.data(), M, vrawb.data());

        State::LayerCache& lc = st.layers[i];
        auto mix = [&](const std::vector<float>& taps, const float* base, std::vector<float>& prev,
                       int dim, float* outv) {
            for (int m = 0; m < M; m++) {
                const float* raw = base + (size_t)m * dim;
                const float* p1 = (m >= 1) ? base + (size_t)(m - 1) * dim : &prev[0];
                const float* p2 = (m >= 2) ? base + (size_t)(m - 2) * dim : (m == 1 ? &prev[0] : &prev[dim]);
                float* o = outv + (size_t)m * dim;
                for (int d = 0; d < dim; d++) {
                    float acc = taps[d] * raw[d];
                    if (c.taps > 1) acc += taps[(size_t)dim + d] * p1[d];
                    if (c.taps > 2) acc += taps[(size_t)2 * dim + d] * p2[d];
                    o[d] = acc;
                }
            }
            // shift history forward by M: [t-1, t-2] <- [last, second-last]
            std::memcpy(&prev[dim], base + (size_t)(M - 2) * dim, (size_t)dim * sizeof(float));
            std::memcpy(&prev[0], base + (size_t)(M - 1) * dim, (size_t)dim * sizeof(float));
        };
        mix(L.qTaps, qrawb.data(), lc.prevQ, qDim, qb.data());
        mix(L.kTaps, krawb.data(), lc.prevK, kDim, kbb.data());
        mix(L.vTaps, vrawb.data(), lc.prevV, vDim, vbb.data());

        for (int m = 0; m < M; m++) {
            const int t = t0 + m;
            float* q = &qb[(size_t)m * qDim];
            float* kk = &kbb[(size_t)m * kDim];
            float* vv = &vbb[(size_t)m * vDim];
            for (int hd = 0; hd < c.H; hd++) zcrms(q + (size_t)hd * c.QK, c.QK, L.qNorm.data(), q + (size_t)hd * c.QK);
            for (int hd = 0; hd < c.KVH; hd++) zcrms(kk + (size_t)hd * c.QK, c.QK, L.kNorm.data(), kk + (size_t)hd * c.QK);
            const int half = c.QK / 2;
            for (int d = 0; d < half; d++) {
                const double ang = (double)t * ropeInv_[d];
                cosv[d] = (float)std::cos(ang);
                sinv[d] = (float)std::sin(ang);
            }
            auto rope = [&](float* v) {
                for (int d = 0; d < half; d++) {
                    const float x1 = v[d], x2 = v[d + half];
                    v[d] = x1 * cosv[d] - x2 * sinv[d];
                    v[d + half] = x2 * cosv[d] + x1 * sinv[d];
                }
            };
            for (int hd = 0; hd < c.H; hd++) rope(q + (size_t)hd * c.QK);
            for (int hd = 0; hd < c.KVH; hd++) rope(kk + (size_t)hd * c.QK);
            if (quant) {
                for (int hd = 0; hd < c.H; hd++) fakeQuantRow(q + (size_t)hd * c.QK, c.QK);
                for (int hd = 0; hd < c.KVH; hd++) fakeQuantRow(kk + (size_t)hd * c.QK, c.QK);
                for (int hd = 0; hd < c.KVH; hd++) fakeQuantRow(vv + (size_t)hd * c.VH, c.VH);
            }
            lc.k.insert(lc.k.end(), kk, kk + kDim);
            lc.v.insert(lc.v.end(), vv, vv + vDim);
        }

        lap(profile.qkv);
        // ---- attention (per token, over the cache capped at this token's position)
        {
            const bool global = c.isGlobal(i);
            const int group = c.H / c.KVH;
            const float scale = 1.0f / std::sqrt((float)c.QK);
            for (int m = 0; m < M; m++) {
                const int t = t0 + m;
                const int lo = (global || !c.window) ? 0 : std::max(0, t - c.window + 1);
                const int S = t - lo + 1;
                scores.resize(S);
                const float* q = &qb[(size_t)m * qDim];
                float* attn = &attnb[(size_t)m * oDim];
                for (int hd = 0; hd < c.H; hd++) {
                    const int kh = hd / group;
                    const float* qh = q + (size_t)hd * c.QK;
                    for (int s = 0; s < S; s++)
                        scores[s] = dotf(qh, &lc.k[((size_t)(lo + s) * c.KVH + kh) * c.QK], c.QK) * scale;
                    softmaxInPlace(scores.data(), S);
                    float* oh = attn + (size_t)hd * c.VH;
                    for (int d = 0; d < c.VH; d++) oh[d] = 0.f;
                    for (int s = 0; s < S; s++) {
                        const float p = scores[s];
                        const float* vs = &lc.v[((size_t)(lo + s) * c.KVH + kh) * c.VH];
                        for (int d = 0; d < c.VH; d++) oh[d] += p * vs[d];
                    }
                }
            }
        }
        lap(profile.attn);

        matvecBatch(L.gateProj, xnrb.data(), M, gateb.data());
        for (int m = 0; m < M; m++) {
            float* attn = &attnb[(size_t)m * oDim];
            for (int d = 0; d < oDim; d++) attn[d] *= sigmoidf(gateb[(size_t)m * oDim + d]);
            if (quant && qmode() != 1) fakeQuantRow(attn, oDim);
            rotateInput(attn, oDim, &attnrb[(size_t)m * oDim]);
        }
        matvecBatch(L.outProj, attnrb.data(), M, aoutb.data());
        const float ag = sigmoidf(L.attnGate);
        for (int m = 0; m < M; m++) {
            zcrms(&aoutb[(size_t)m * D], D, L.postNorm.data(), &xnb[(size_t)m * D]);
            for (int d = 0; d < D; d++) xb[(size_t)m * D + d] = skipb[(size_t)m * D + d] + ag * xnb[(size_t)m * D + d];
            std::memcpy(&skipb[(size_t)m * D], &xb[(size_t)m * D], D * sizeof(float));
        }
        lap(profile.gateout);

        // ---- Hadamard MLP (per token)
        for (int m = 0; m < M; m++) {
            float* x = &xb[(size_t)m * D];
            float* xn = &xnb[(size_t)m * D];
            zcrms(x, D, L.preHada.data(), xn);
            float logit8[8];
            for (int r = 0; r < 8; r++) logit8[r] = 0.f;
            for (int d = 0; d < D; d++)
                for (int r = 0; r < 8; r++) logit8[r] += xn[d] * L.condV[(size_t)d * 8 + r];
            softmaxInPlace(logit8, 8);
            for (int q = 0; q < c.hadaN; q++) {
                float acc = 0.f;
                for (int r = 0; r < 8; r++) acc += logit8[r] * L.condU[(size_t)r * c.hadaN + q];
                cond[q] = 1.f + acc;
            }
            std::fill(z.begin(), z.end(), 0.f);
            for (int d = 0; d < D; d++) z[d] = xn[d];
            for (int q = 0; q < c.hadaN; q++) z[q] *= L.d1[q];
            kronApply(z.data(), c.hadaN, L.w1a.data(), L.w1b.data(), hb, hb, z2.data());
            for (int q = 0; q < c.hadaN; q++) z[q] = z2[hadaP1_[q]];
            for (int q = 0; q < c.hadaN; q++) {
                const float v = L.d2[q] * cond[q] * z[q] + L.b2[q];
                z[q] = v * sigmoidf(v);
            }
            kronApply(z.data(), c.hadaN, L.w2a.data(), L.w2b.data(), hb, hb, z2.data());
            for (int q = 0; q < c.hadaN; q++) z[q] = z2[hadaP2_[q]] * L.d3[q];
            kronApply(z.data(), c.hadaN, L.w3a.data(), L.w3b.data(), hb, hb, z2.data());
            float* blk = &blkb[(size_t)m * D];
            for (int d = 0; d < D; d++) blk[d] = skipb[(size_t)m * D + d] + L.d4[d] * z2[d];
        }
        lap(profile.hada);

        // ---- redistribute the block's delta over the lanes, then pool the cell for the probe head
        for (int m = 0; m < M; m++) {
            const float* u = &ub[(size_t)m * D];
            const float* blk = &blkb[(size_t)m * D];
            float* y = &yb[(size_t)m * D];
            for (int d = 0; d < D; d++) y[d] = blk[d] - u[d];
            double lg[16];
            for (int a = 0; a < n * n; a++)
                lg[a] = (double)mhcARes_[i] * resb[(size_t)m * n * n + a] + mhcBRes_[(size_t)i * n * n + a];
            float hres[16];
            sinkhorn4(lg, n, hres);
            float* s = &stream[(size_t)m * nC];
            float* ns = &scr[(size_t)m * nC];
            for (int a = 0; a < n; a++)
                for (int d = 0; d < D; d++) {
                    float acc = hpostb[(size_t)m * n + a] * y[d];
                    for (int b = 0; b < n; b++) acc += hres[a * n + b] * s[(size_t)b * D + d];
                    ns[(size_t)a * D + d] = acc;
                }
            std::memcpy(s, ns, (size_t)nC * sizeof(float));
            if (probing) {
                float* cell = &cells[((size_t)m * (c.L + 1) + i + 1) * D];
                for (int d = 0; d < D; d++) {
                    float acc = 0.f;
                    for (int j = 0; j < n; j++) acc += s[(size_t)j * D + d];
                    cell[d] = acc / n;
                }
            }
        }
        lap(profile.mhc2);
    }
    profile.tokens += M;

    // ---- the probe pool is a running online softmax: feed it in strict token order
    if (probing)
        for (int m = 0; m < M; m++) {
            probeUpdate(st, 0, &cells[((size_t)m * (c.L + 1)) * D]);
            for (int i = 0; i < c.L; i++) probeUpdate(st, i + 1, &cells[((size_t)m * (c.L + 1) + i + 1) * D]);
        }
}

// One token's cell vector x (D floats) for layer cell `cell` joins the softmax pooling of its K probes.
void Model::probeUpdate(State& st, int cell, const float* x) const {
    const int D = cfg_.D, K = headK_;
    const float inv = 1.0f / std::sqrt((float)D);
    for (int k = 0; k < K; k++) {
        const size_t idx = (size_t)cell * K + k;
        const float score = dotf(x, &headProbes_[idx * D], D) * inv;
        float& m = st.probeM[idx];
        float& s = st.probeS[idx];
        float* r = &st.probeR[idx * D];
        const float newM = std::max(m, score);
        const float scale = (float)std::exp((double)m - newM);          // 0 for the first token (m = -1e30)
        const float w = (float)std::exp((double)score - newM);
        for (int d = 0; d < D; d++) r[d] = r[d] * scale + w * x[d];
        s = s * scale + w;
        m = newM;
    }
}

double Model::confidenceLogit(const State& st) const {
    if (headProj_.empty() || st.probeM.empty()) return 0.0;
    const int D = cfg_.D, K = headK_, Q = headQ_, cells = cfg_.L + 1, pk = cells * K;
    const float inv = 1.0f / std::sqrt((float)D);
    // r[cell][k] = rms_unit(pooled cell vector) * gain
    std::vector<float> r((size_t)pk * D);
    for (int i = 0; i < pk; i++) {
        const float s = st.probeS[i] > 0.f ? st.probeS[i] : 1.f;
        double ss = 0;
        for (int d = 0; d < D; d++) { const float v = st.probeR[(size_t)i * D + d] / s; r[(size_t)i * D + d] = v; ss += (double)v * v; }
        const float scale = (float)(1.0 / std::sqrt(ss / D + 1e-6)) * headGain_[i];
        for (int d = 0; d < D; d++) r[(size_t)i * D + d] *= scale;
    }
    // u[q][cell*K+k] = r . query[q] / sqrt(D) + row_bias; softmax over all (cell,k) per query; pooled[q] = sum w r
    std::vector<float> pooled((size_t)Q * D, 0.f), u(pk);
    for (int q = 0; q < Q; q++) {
        float mx = -1e30f;
        for (int i = 0; i < pk; i++) {
            u[i] = dotf(&r[(size_t)i * D], &headQuery_[(size_t)q * D], D) * inv + headRowBias_[(size_t)q * pk + i];
            mx = std::max(mx, u[i]);
        }
        double den = 0;
        for (int i = 0; i < pk; i++) { u[i] = (float)std::exp((double)u[i] - mx); den += u[i]; }
        for (int i = 0; i < pk; i++) {
            const float w = (float)(u[i] / den);
            const float* ri = &r[(size_t)i * D];
            float* po = &pooled[(size_t)q * D];
            for (int d = 0; d < D; d++) po[d] += w * ri[d];
        }
    }
    return (double)dotf(headProj_.data(), pooled.data(), Q * D) + headBias_;
}

} // namespace needle
