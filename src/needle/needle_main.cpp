// Needle 3 mode of mmllm:  mmllm --needle needle3.cact --tools tools.json --prompt "..."
//                          mmllm --needle needle3.cact --tools tools.json --serve [--port N]
//
// The model runs on the CPU by default. --gpu offloads the 2-bit CQ matrix-vector products to the
// GPU (see NEEDLE.md): it is numerically equivalent but on the target GeForce 9400M it is slower.
#include "needle_main.h"
#include "cact.h"
#include "ngpu.h"
#include "nmodel.h"
#include "serve.h"
#include "session.h"
#include "tokenizer.h"
#include "../gl/context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace needle {

static const int kBos = 2;

static bool readFile(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---- golden-file test: compares tokenizer ids and full logit vectors with the NumPy reference
static int runGolden(Model& model, const Tokenizer& tok, const std::string& dir) {
    setenv("NEEDLE_QMODE", "0", 1);                 // the JAX oracle quantises whole rows before the rotation
    std::string bin, prompt;
    if (!readFile(dir + "/golden.bin", bin) || !readFile(dir + "/golden_prompt.txt", prompt)) {
        std::fprintf(stderr, "needle-test: cannot read golden.bin / golden_prompt.txt in %s\n", dir.c_str());
        return 1;
    }
    const uint8_t* p = (const uint8_t*)bin.data();
    uint32_t n; std::memcpy(&n, p, 4);
    std::vector<int32_t> ids(n);
    std::memcpy(ids.data(), p + 4, n * 4);
    size_t off = 4 + n * 4;
    uint32_t npos; std::memcpy(&npos, p + off, 4); off += 4;

    std::vector<int> mine = tok.encode(prompt);
    mine.insert(mine.begin(), kBos);
    bool same = mine.size() == ids.size();
    for (size_t i = 0; same && i < mine.size(); i++) same = mine[i] == ids[i];
    std::printf("tokenizer: %zu ids vs reference %u -> %s\n", mine.size(), n, same ? "IDENTICAL" : "MISMATCH");

    const int V = model.config().vocab;
    std::vector<float> logits(V), ref(V);
    int bad = same ? 0 : 1;
    for (int quant = 0; quant < 2; quant++) {
        State st = model.newState();
        std::vector<std::pair<uint32_t, size_t>> recs;              // position, byte offset of logits
        size_t scan = off;
        for (int qq = 0; qq < quant; qq++) scan += (size_t)npos * (8 + (size_t)V * 4);
        for (uint32_t r = 0; r < npos; r++) {
            uint32_t q, pos;
            std::memcpy(&q, p + scan, 4); std::memcpy(&pos, p + scan + 4, 4);
            recs.push_back({pos, scan + 8});
            scan += 8 + (size_t)V * 4;
        }
        double worst = 0, minCos = 1;
        int top1ok = 0;
        const double t0 = now();
        for (size_t t = 0; t < ids.size(); ) {
            const bool want = std::any_of(recs.begin(), recs.end(), [&](const std::pair<uint32_t, size_t>& r) { return r.first == t; });
            if (!want && model.batchPrefill) {
                // no reference logits are needed for this stretch: run it as one batch (covers prefill())
                size_t upto = t;
                while (upto < ids.size() && !std::any_of(recs.begin(), recs.end(),
                       [&](const std::pair<uint32_t, size_t>& r) { return r.first == upto; })) upto++;
                model.prefill(st, ids.data() + t, (int)(upto - t), quant != 0);
                t = upto;
                continue;
            }
            model.step(st, ids[t], quant != 0, want ? logits.data() : nullptr);
            if (!want) { t++; continue; }
            for (const auto& r : recs) if (r.first == t) std::memcpy(ref.data(), p + r.second, (size_t)V * 4);
            double dot = 0, na = 0, nb = 0, md = 0;
            int am = 0, bm = 0;
            for (int i = 0; i < V; i++) {
                dot += (double)logits[i] * ref[i]; na += (double)logits[i] * logits[i]; nb += (double)ref[i] * ref[i];
                md = std::max(md, (double)std::fabs(logits[i] - ref[i]));
                if (logits[i] > logits[am]) am = i;
                if (ref[i] > ref[bm]) bm = i;
            }
            worst = std::max(worst, md);
            minCos = std::min(minCos, dot / std::sqrt(na * nb));
            top1ok += (am == bm);
            t++;
        }
        std::printf("quant=%d: max|logit diff| %.5f, min cosine %.8f, top-1 agreement %d/%u positions  (%.1fs)\n",
                    quant, worst, minCos, top1ok, npos, now() - t0);
        // float mode must agree to ~1e-4; with the int8 simulation a single rounding flip moves a logit by a
        // whole quantisation step, so the criterion is cosine > 0.999 and the same top choice
        if (quant == 0 ? (worst > 0.01) : (minCos < 0.999)) bad = 1;
        if (top1ok != (int)npos) bad = 1;
    }
    std::printf("needle-test: %s\n", bad ? "FAILED" : "PASSED");
    return bad;
}

static void printUsage() {
    std::printf("Needle 3 mode:\n"
                "  mmllm --needle needle3.cact --tools tools.json --prompt \"dim the living room to 30\"\n"
                "  mmllm --needle needle3.cact --tools tools.json --serve [--port 8080]   (localhost HTTP)\n"
                "    --system <text>      session facts (date, locale, device)\n"
                "    --max <n>            max generated tokens per turn (default 220)\n"
                "    --no-quant           float activations instead of the engine's int8 simulation\n"
                "    --no-grammar         do not constrain the call to the tool schemas\n"
                "    --no-repair          return the model's calls as generated (no repair / withholding gates)\n"
                "    --strip-constraints  drop minimum/maximum/pattern/... from tool schemas before rendering\n"
                "    --threads <n>        worker threads (default: min(cores, 4))\n"
                "    --raw                also print the raw generated text\n"
                "    --gpu                run the 2-bit matrix-vector products on the GPU (needs a GL 3.3 context)\n"
                "    --gpu-check          verify the GPU projections against the CPU and exit\n"
                "    --batch-prefill      prefill the prompt with batched GEMMs instead of per-token GEMVs\n"
                "    --needle-test <dir>  check against golden.bin / golden_prompt.txt\n"
                "  HTTP: POST /complete {\"input\":\"...\"}  (turns continue one conversation), POST /reset\n");
}

int needleMain(int argc, char** argv) {
    std::string cactPath, toolsPath, prompt, system, testDir;
    int threads = 0, port = 8080;
    bool serveMode = false, raw = false, useGpu = false, gpuCheck = false, batchPrefill = false;
    Options opt;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--needle") cactPath = next();
        else if (a == "--tools") toolsPath = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--system") system = next();
        else if (a == "--max") opt.maxNew = std::atoi(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--port") port = std::atoi(next().c_str());
        else if (a == "--serve") serveMode = true;
        else if (a == "--no-quant") opt.quant = false;
        else if (a == "--no-grammar") opt.grammar = false;
        else if (a == "--no-repair") opt.repair = false;
        else if (a == "--strip-constraints") opt.strip = true;
        else if (a == "--no-normalize") opt.strip = false;       // accepted for compatibility
        else if (a == "--raw") raw = true;
        else if (a == "--gpu") useGpu = true;
        else if (a == "--gpu-check") { useGpu = true; gpuCheck = true; }
        else if (a == "--batch-prefill") batchPrefill = true;
        else if (a == "--needle-test") testDir = next();
        else if (a == "--help" || a == "-h") { printUsage(); return 0; }
    }
    if (cactPath.empty()) { printUsage(); return 1; }

    Cact cact;
    double t0 = now();
    if (!cact.load(cactPath)) { std::fprintf(stderr, "needle: %s\n", cact.error().c_str()); return 1; }
    Tokenizer tok;
    {
        bool ok = false;
        for (size_t i = cact.numTensors(); i-- > 0;)
            if (cact.record(i).dtype == DT_RAW) { ok = tok.load(cact.raw(i)); break; }
        if (!ok) { std::fprintf(stderr, "needle: archive has no usable tokenizer (%s)\n", tok.error().c_str()); return 1; }
    }
    Model model;
    if (threads > 0) model.setThreads(threads);
    std::string err;
    if (!model.load(cact, err)) { std::fprintf(stderr, "needle: %s\n", err.c_str()); return 1; }
    model.batchPrefill = batchPrefill;
    std::fprintf(stderr, "[needle] loaded %s in %.1fs (%u tensors, %d layers, width %d, vocab %d)\n", cactPath.c_str(),
                 now() - t0, cact.header().numTensors, model.config().L, model.config().D, model.config().vocab);

    cact.release();                                  // the model holds its own (packed) copies

    // ---- optional GPU offload of the 2-bit projections.
    gl::Context gpuCtx;
    NGpu gpu;
    if (useGpu) {
        gl::quiet = true;                            // stdout carries our JSON result: keep GL quiet
        if (!gpuCtx.init()) {
            std::fprintf(stderr, "needle: --gpu needs a headless OpenGL 3.3 context (egl/Mesa)\n");
            return 1;
        }
        std::string gerr;
        if (!gpu.init(gerr)) { std::fprintf(stderr, "needle: %s\n", gerr.c_str()); return 1; }
        if (!model.enableGpu(&gpu)) {
            std::fprintf(stderr, "needle: could not upload the 2-bit weights to the GPU\n");
            return 1;
        }
        std::fprintf(stderr, "[needle-gpu] %lld weight matrices uploaded (packed 2-bit) to %s\n",
                     gpu.uploads, glGetString(GL_RENDERER));
    }

    if (gpuCheck) {
        std::string rep;
        const bool ok = model.gpuSelfCheck(rep);
        std::printf("gpu-check: %s\n%s\n", ok ? "PASSED" : "FAILED", rep.c_str());
        return ok ? 0 : 1;
    }

    if (!testDir.empty()) return runGolden(model, tok, testDir);

    if (toolsPath.empty() || (!serveMode && prompt.empty())) { printUsage(); return 1; }
    std::string toolsText;
    if (!readFile(toolsPath, toolsText)) { std::fprintf(stderr, "needle: cannot read %s\n", toolsPath.c_str()); return 1; }

    model.profiling = std::getenv("NEEDLE_PROFILE") != nullptr;
    Session session(model, tok, opt);
    if (!session.init(toolsText, system, err)) { std::fprintf(stderr, "needle: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "[needle] tool prefix: %d tokens processed in %.2fs\n", session.prefixTokens(), session.prefixSeconds());

    if (serveMode) return serve(session, port);

    if ((int)session.prefixTokens() + opt.maxNew > (int)cact.header().maxSeqLen) { std::fprintf(stderr, "needle: prompt too long\n"); return 1; }
    const Result r = session.complete(prompt);
    std::printf("%s\n", r.toJson(session.prefixTokens()).c_str());
    if (raw) std::printf("--- raw generation ---\n%s\n", r.raw.c_str());
    if (model.profiling && model.profile.tokens) {
        const Model::Profile& p = model.profile;
        const double n = (double)p.tokens;
        std::fprintf(stderr, "[profile] ms/token over %ld steps: mhc-in %.2f  engram %.2f  qkv+taps+rope %.2f  attention %.2f  gate+out %.2f  hadamard-mlp %.2f  mhc-out %.2f  head %.2f\n",
                     p.tokens, p.mhc / n * 1000, p.engram / n * 1000, p.qkv / n * 1000, p.attn / n * 1000, p.gateout / n * 1000,
                     p.hada / n * 1000, p.mhc2 / n * 1000, p.head / n * 1000);
    }
    return r.success ? 0 : 2;
}

} // namespace needle
