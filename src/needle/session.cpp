#include "session.h"
#include "repair.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>

namespace needle {

static int histMode() {
    static const int m = std::getenv("NEEDLE_HIST") ? std::atoi(std::getenv("NEEDLE_HIST")) : 0;
    return m;
}
static const int kBos = 2, kEos = 1, kImEnd = 5, kThink = 6, kThinkEnd = 7, kToolCallStart = 10, kToolCallEnd = 11;

static double nowSec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---- tool schema handling. The shipped engine renders constraint keys such as minimum/maximum into
// the prompt (checked: the reasoning text only matches the engine's when they are left in), so the
// default is to render them verbatim. Stripping them is available as an explicit option.
static bool isDroppedKey(const std::string& k) {
    static const char* drop[] = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                                 "minLength", "maxLength", "pattern", "format", "minItems", "maxItems",
                                 "uniqueItems", nullptr};
    for (int i = 0; drop[i]; i++) if (k == drop[i]) return true;
    return false;
}
static void stripConstraints(json::Value& v) {
    if (v.type == json::Value::Object) {
        for (size_t i = 0; i < v.obj.size();) {
            if (isDroppedKey(v.obj[i].first)) v.obj.erase(v.obj.begin() + i);
            else { stripConstraints(v.obj[i].second); i++; }
        }
    } else if (v.type == json::Value::Array) {
        for (auto& e : v.arr) stripConstraints(e);
    }
}
static json::Value normaliseTools(const json::Value& tools, bool strip) {
    json::Value out;
    out.type = json::Value::Array;
    for (const json::Value& t : tools.arr) {
        const json::Value* fn = t.get("function");                       // OpenAI-style wrapper
        const json::Value& tool = fn ? *fn : t;
        json::Value kept;
        kept.type = json::Value::Object;
        for (const char* key : {"name", "description", "parameters"})
            if (const json::Value* v = tool.get(key)) kept.obj.emplace_back(key, *v);
        if (strip) if (json::Value* p = kept.get("parameters")) stripConstraints(*p);
        out.arr.push_back(std::move(kept));
    }
    return out;
}

Session::Session(Model& model, const Tokenizer& tok, const Options& opt)
    : model_(model), tok_(tok), opt_(opt) {}

bool Session::init(const std::string& toolsText, const std::string& system, std::string& err) {
    json::Value tools;
    try { tools = json::parse(toolsText); }
    catch (const std::exception& e) { err = std::string("tools: ") + e.what(); return false; }
    if (tools.type != json::Value::Array) { err = "tools must be a JSON array"; return false; }
    tools = normaliseTools(tools, opt_.strip);
    toolsNorm_ = tools;

    grammarOk_ = false;
    if (opt_.grammar) {
        std::string gerr;
        grammarOk_ = grammar_.build(tools, gerr);
        if (!grammarOk_) std::fprintf(stderr, "[needle] grammar unavailable (%s); decoding unconstrained\n", gerr.c_str());
    }

    std::string prefix;
    if (!system.empty()) prefix += "<|im_start|>system\n" + system + "<|im_end|>\n";
    prefix += "<|im_start|>user\n<tools>" + json::dump(tools) + "</tools>";
    std::vector<int> ids = tok_.encode(prefix);
    ids.insert(ids.begin(), kBos);

    prefix_ = model_.newState();
    const double t0 = nowSec();
    if (model_.batchPrefill && ids.size() > 1) model_.prefill(prefix_, ids.data(), (int)ids.size(), opt_.quant);
    else for (int id : ids) model_.step(prefix_, id, opt_.quant, nullptr);
    prefixSeconds_ = nowSec() - t0;
    prefixTokens_ = (int)ids.size();
    state_ = prefix_;
    turns_ = 0;
    logits_.assign(model_.config().vocab, 0.f);
    return true;
}

void Session::reset() {
    state_ = prefix_;
    turns_ = 0;
    conversation_.clear();
}

static std::string trim(std::string s) {
    while (!s.empty() && (s.front() == '\n' || s.front() == ' ')) s.erase(0, 1);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

Result Session::complete(const std::string& input) {
    Result r;
    conversation_ += (conversation_.empty() ? "" : "\n") + input;
    // ---- this turn's text
    std::string turn;
    bool resultTurn = false;
    if (turns_ == 0) {
        turn = "\n" + input + "<|im_end|>\n<|im_start|>assistant\n";        // continues the open user message
    } else {
        bool isResult = false;
        if (!input.empty() && (input[0] == '{' || input[0] == '[')) {
            try { json::parse(input); isResult = true; } catch (const std::exception&) {}
        }
        resultTurn = isResult;
        if (isResult) turn = "\n<|im_start|>tool\n<tool_result>" + input + "</tool_result><|im_end|>\n<|im_start|>assistant\n";
        else turn = "\n<|im_start|>user\n" + input + "<|im_end|>\n<|im_start|>assistant\n";
    }
    const std::vector<int> ids = tok_.encode(turn);
    const int room = (int)model_.config().vocab;   // (vocab only used for sizing below)
    (void)room;
    r.promptTokens = (int)ids.size();

    const double t0 = nowSec();
    if (model_.batchPrefill && ids.size() > 1) {
        model_.prefill(state_, ids.data(), (int)ids.size() - 1, opt_.quant);   // logits only needed on the last token
        model_.step(state_, ids.back(), opt_.quant, logits_.data());
    } else {
        for (size_t t = 0; t < ids.size(); t++)
            model_.step(state_, ids[t], opt_.quant, t + 1 == ids.size() ? logits_.data() : nullptr);
    }
    r.prefillSeconds = nowSec() - t0;
    const double headPrefill = model_.hasConfidenceHead() ? 1.0 / (1.0 + std::exp(-model_.confidenceLogit(state_))) : -1;
    double pMin = 1, pProd = 1, pSum = 0; int pN = 0;

    // ---- decode. Phases:  THINK (free, first token forced to <think>)  ->  after </think> force "\n<tool_call>"
    //      ->  CALL (constrained by the tools' grammar when available)  ->  force </tool_call> and <|im_end|>.
    //      The closing <|im_end|> goes into the state too, so the next turn continues the conversation.
    enum { THINK, FORCED, CALL } phase = THINK;
    std::vector<int> out, forced;
    const int V = model_.config().vocab;
    const bool useGrammar = grammarOk_;
    Grammar::State gs = grammar_.start();
    const double t1 = nowSec();
    std::string bytes;
    std::vector<int> order;
    for (int n = 0; n < opt_.maxNew; n++) {
        int nxt = -1, modelBest = -2;
        if (n == 0) nxt = kThink;
        else if (!forced.empty()) { nxt = forced.front(); forced.erase(forced.begin()); }
        else {
            // unconstrained best token
            int best = 0;
            for (int i = 1; i < V; i++) if (logits_[i] > logits_[best]) best = i;
            nxt = best;
            modelBest = best;
            if (phase == CALL && useGrammar) {
                if (gs.complete()) { nxt = kToolCallEnd; }             // the array is closed: only </tool_call> may follow
                else if (!(tok_.tokenBytes(best, bytes) && gs.accepts(bytes))) {
                    // walk down the ranked candidates until one keeps the output a valid prefix
                    order.resize(V);
                    for (int i = 0; i < V; i++) order[i] = i;
                    std::sort(order.begin(), order.end(), [&](int a, int b) { return logits_[a] > logits_[b]; });
                    nxt = -1;
                    for (int id : order)
                        if (tok_.tokenBytes(id, bytes) && gs.accepts(bytes)) { nxt = id; break; }
                    if (nxt < 0) { r.error = "grammar dead end"; break; }
                }
            }
        }

        if (phase == THINK && n > 0 && std::getenv("NEEDLE_TOP2_DEBUG")) {
            int t1 = 0, t2 = -1;
            for (int i = 1; i < V; i++) if (logits_[i] > logits_[t1]) t1 = i;
            for (int i = 0; i < V; i++) if (i != t1 && (t2 < 0 || logits_[i] > logits_[t2])) t2 = i;
            std::fprintf(stderr, "[top2] n=%d chose %d  top1 %d (%.3f) top2 %d (%.3f) margin %.3f\n", n, nxt, t1, logits_[t1], t2, logits_[t2], logits_[t1] - logits_[t2]);
        }
        if (phase == CALL && forced.empty() && n > 0 && nxt != kToolCallEnd && nxt == modelBest) {
            float mx = logits_[0];
            for (int i = 1; i < V; i++) mx = std::max(mx, logits_[i]);
            double den = 0;
            for (int i = 0; i < V; i++) den += std::exp((double)logits_[i] - mx);
            const double pr = std::exp((double)logits_[nxt] - mx) / den;
            pMin = std::min(pMin, pr); pProd *= pr; pSum += pr; pN++;
        }
        if (nxt == kEos || nxt == kImEnd) {
            if (nxt == kImEnd) model_.step(state_, nxt, opt_.quant, nullptr);
            break;
        }
        if (phase == CALL && useGrammar && nxt != kToolCallEnd && tok_.tokenBytes(nxt, bytes))
            for (unsigned char ch : bytes) gs.feed(ch);
        out.push_back(nxt);
        model_.step(state_, nxt, opt_.quant, logits_.data());

        if (phase == THINK && nxt == kThinkEnd) {                      // reasoning finished: force the call block
            std::vector<int> nl = tok_.encode("\n");
            forced.insert(forced.end(), nl.begin(), nl.end());
            forced.push_back(kToolCallStart);
            phase = FORCED;
        } else if (phase == FORCED && forced.empty()) {
            phase = CALL;
        }
        if (phase == CALL && nxt == kToolCallEnd) {                    // end of the call block: close the turn
            if (histMode() != 1) model_.step(state_, kImEnd, opt_.quant, nullptr);
            break;
        }
    }
    r.decodeSeconds = nowSec() - t1;
    {
        const double headEnd = model_.hasConfidenceHead() ? 1.0 / (1.0 + std::exp(-model_.confidenceLogit(state_))) : -1;
        if (headEnd >= 0) r.confidence = std::min(headEnd, pMin);
        if (std::getenv("NEEDLE_CONF_DEBUG"))
            std::fprintf(stderr, "[conf] headPrefill %.4f headEnd %.4f pmin %.4f pprod %.4f pmean %.4f n %d\n",
                         headPrefill, headEnd, pMin, pProd, pN ? pSum / pN : 0.0, pN);
    }
    r.generatedTokens = (int)out.size();
    r.raw = tok_.decode(out);
    turns_++;

    // ---- reasoning and calls from the generated text
    const std::string& gen = r.raw;
    size_t a = gen.find("<think>"), b = gen.find("</think>");
    if (a != std::string::npos && b != std::string::npos && b > a) r.reasoning = trim(gen.substr(a + 7, b - a - 7));
    size_t c0 = gen.find("<tool_call>"), c1 = gen.find("</tool_call>");
    if (c0 == std::string::npos || c1 == std::string::npos || c1 < c0) {
        r.error = "no <tool_call> block generated";
        return r;
    }
    try {
        r.calls = json::parse(gen.substr(c0 + 11, c1 - c0 - 11));
        r.success = r.calls.type == json::Value::Array;
        if (!r.success) r.error = "tool_call is not a JSON array";
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    r.suppressed.type = json::Value::Array;
    r.respond = r.success && resultTurn && r.calls.arr.empty();
    if (r.success && opt_.repair) {
        RepairOutcome ro = repairCalls(r.calls, toolsNorm_, conversation_, r.confidence);
        r.calls = std::move(ro.calls);
        r.suppressed = std::move(ro.suppressed);
        r.notes = std::move(ro.notes);
    }
    return r;
}

std::string Result::toJson(int prefixTokens) const {
    std::string js = respond ? "{\"type\":\"respond\",\"success\":" : "{\"type\":\"call\",\"success\":";
    js += success ? "true" : "false";
    js += ",\"error\":";
    if (error.empty()) js += "null"; else json::dumpString(js, error);
    js += ",\"function_calls\":";
    js += success ? json::dump(calls) : "[]";
    js += ",\"suppressed_calls\":";
    js += suppressed.type == json::Value::Array ? json::dump(suppressed) : "[]";
    js += ",\"reasoning\":";
    json::dumpString(js, reasoning);
    if (confidence >= 0) { char c[48]; std::snprintf(c, sizeof(c), ",\"confidence\":%.4f", confidence); js += c; }
    else js += ",\"confidence\":null";
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  ",\"prefix_tokens\":%d,\"prompt_tokens\":%d,\"generated_tokens\":%d,\"prefill_tps\":%.1f,\"decode_tps\":%.1f}",
                  prefixTokens, promptTokens, generatedTokens,
                  prefillSeconds > 0 ? promptTokens / prefillSeconds : 0.0,
                  decodeSeconds > 0 ? generatedTokens / decodeSeconds : 0.0);
    js += buf;
    return js;
}

} // namespace needle
