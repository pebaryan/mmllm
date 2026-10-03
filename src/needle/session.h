#pragma once
// A Needle conversation bound to one tool set.
//
// The prompt prefix (optional system turn + the tool schemas) is identical for every query, so it is
// processed once and its state (KV caches, conv-tap history, engram history) is snapshotted; each
// conversation starts from a copy. Because the tokenizer splits text at chat markers,
// encode(prefix) + encode(turn) == encode(prefix + turn), which init() relies on.
#include "grammar.h"
#include "json.h"
#include "nmodel.h"
#include "tokenizer.h"
#include <string>
#include <vector>

namespace needle {

struct Options {
    bool quant = true;          // simulate the engine's int8 activations / KV cache
    bool strip = false;         // drop minimum/maximum/... from tool schemas before rendering
    int maxNew = 220;           // generated tokens per turn
    bool grammar = true;        // constrain the call to the tools' schemas (--no-grammar to disable)
    bool repair = true;         // argument repair and withholding gates (--no-repair to disable)
};

struct Result {
    bool success = false;
    bool respond = false;       // tool-result turn that ended without a call: the assistant answers in text
    std::string error;
    std::string reasoning;
    json::Value calls;          // array of {name, arguments} that go out
    json::Value suppressed;     // calls withheld by a gate
    double confidence = -1;     // < 0 = not computed
    std::vector<std::string> notes;   // repairs applied / reasons a call was withheld
    std::string raw;            // raw generated text
    int promptTokens = 0, generatedTokens = 0;
    double prefillSeconds = 0, decodeSeconds = 0;
    std::string toJson(int prefixTokens) const;
};

class Session {
public:
    Session(Model& model, const Tokenizer& tok, const Options& opt);

    // Build and process the prefix. toolsJson is the user's tools file content.
    bool init(const std::string& toolsJson, const std::string& system, std::string& err);

    // One turn. The first call continues the open user message of the prefix; later calls open their
    // own user turn, or a tool-result turn when `input` is a JSON object/array.
    Result complete(const std::string& input);

    void reset();                       // forget the conversation, keep the tools
    int prefixTokens() const { return prefixTokens_; }
    double prefixSeconds() const { return prefixSeconds_; }

private:
    Model& model_;
    const Tokenizer& tok_;
    Options opt_;
    State prefix_, state_;
    Grammar grammar_;
    bool grammarOk_ = false;
    json::Value toolsNorm_;          // the tools as rendered into the prompt
    std::string conversation_;       // user text so far (what "stated in the request" is checked against)
    int turns_ = 0, prefixTokens_ = 0;
    double prefixSeconds_ = 0;
    std::vector<float> logits_;
};

} // namespace needle
