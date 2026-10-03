#pragma once
// Byte-level grammar for the tool-call array, compiled from the tools' JSON schemas.
//
//   calls := "[" [ call ("," call)* ] "]"
//   call  := {"name":"<declared tool>","arguments":<object for that tool's parameters>}
//
// Output is compact JSON (no whitespace). Objects only accept the schema's properties, each at most
// once and in any order, and may close only when every required property is present. Values follow
// the schema type: string (or enum), integer, number, boolean, array, object. State is a small pushdown
// automaton that is cheap to copy, so a candidate token can be tested by feeding its bytes to a copy.
#include "json.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace needle {

struct SNode {
    enum Kind { Object, String, Integer, Number, Boolean, Array } kind = String;
    std::vector<std::string> enumVals;                               // allowed literals (strings without quotes)
    std::vector<std::pair<std::string, std::shared_ptr<SNode>>> props;
    std::vector<std::string> required;
    std::shared_ptr<SNode> items;
};

class Grammar {
public:
    struct Tool { std::string name; std::shared_ptr<SNode> args; };

    // tools: array of {name, parameters}. Returns false (with err) if a schema cannot be compiled.
    bool build(const json::Value& tools, std::string& err);
    const std::vector<Tool>& tools() const { return tools_; }

    class State {
    public:
        // Feed one byte; false if it is not a valid continuation (the state is then unspecified).
        bool feed(uint8_t c);
        bool accepts(const std::string& bytes) const;                // on a copy
        bool complete() const;                                        // the closing "]" has been read

    private:
        friend class Grammar;
        enum FType { F_ROOT, F_LIT, F_NAME, F_VALUE, F_OBJ, F_KEY, F_ARR, F_STR, F_ENUMSTR, F_INT, F_NUM, F_BOOL };
        struct Frame {
            FType t = F_ROOT;
            const SNode* n = nullptr;
            int phase = 0;
            std::string s;
            std::vector<char> seen;
            size_t pos = 0;
        };
        bool pushCall();
        const Grammar* g_ = nullptr;
        std::vector<Frame> st_;
    };

    State start() const;

private:
    std::shared_ptr<SNode> compile(const json::Value& schema) const;
    std::vector<Tool> tools_;
};

} // namespace needle
