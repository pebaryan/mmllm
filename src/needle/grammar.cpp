#include "grammar.h"
#include <algorithm>
#include <cstring>

namespace needle {

// ------------------------------------------------------------------------------------------------
// schema compilation
// ------------------------------------------------------------------------------------------------
std::shared_ptr<SNode> Grammar::compile(const json::Value& s) const {
    auto n = std::make_shared<SNode>();
    std::string type = "string";
    if (const json::Value* t = s.get("type")) {
        if (t->type == json::Value::String) type = t->s;
        else if (t->type == json::Value::Array && !t->arr.empty() && t->arr[0].type == json::Value::String) type = t->arr[0].s;
    } else if (s.get("properties")) {
        type = "object";
    }
    if (type == "object") n->kind = SNode::Object;
    else if (type == "integer") n->kind = SNode::Integer;
    else if (type == "number") n->kind = SNode::Number;
    else if (type == "boolean") n->kind = SNode::Boolean;
    else if (type == "array") n->kind = SNode::Array;
    else n->kind = SNode::String;

    if (const json::Value* e = s.get("enum")) {
        if (e->type == json::Value::Array)
            for (const auto& v : e->arr) if (v.type == json::Value::String) n->enumVals.push_back(v.s);
    }
    if (n->kind == SNode::Object) {
        if (const json::Value* p = s.get("properties"))
            for (const auto& kv : p->obj) n->props.emplace_back(kv.first, compile(kv.second));
        if (const json::Value* r = s.get("required"))
            for (const auto& v : r->arr) if (v.type == json::Value::String) n->required.push_back(v.s);
    }
    if (n->kind == SNode::Array) {
        if (const json::Value* it = s.get("items")) n->items = compile(*it);
        else { n->items = std::make_shared<SNode>(); n->items->kind = SNode::String; }
    }
    return n;
}

bool Grammar::build(const json::Value& tools, std::string& err) {
    tools_.clear();
    if (tools.type != json::Value::Array) { err = "tools must be an array"; return false; }
    for (const json::Value& t : tools.arr) {
        const json::Value* fn = t.get("function");
        const json::Value& tool = fn ? *fn : t;
        const json::Value* name = tool.get("name");
        if (!name || name->type != json::Value::String) { err = "a tool has no name"; return false; }
        Tool out;
        out.name = name->s;
        if (const json::Value* params = tool.get("parameters")) out.args = compile(*params);
        else { out.args = std::make_shared<SNode>(); out.args->kind = SNode::Object; }
        if (out.args->kind != SNode::Object) { out.args = std::make_shared<SNode>(); out.args->kind = SNode::Object; }
        tools_.push_back(std::move(out));
    }
    return true;
}

Grammar::State Grammar::start() const {
    State s;
    s.g_ = this;
    State::Frame root;
    root.t = State::F_ROOT;
    s.st_.push_back(root);
    return s;
}

// ------------------------------------------------------------------------------------------------
// the pushdown automaton
// ------------------------------------------------------------------------------------------------
bool Grammar::State::complete() const { return st_.size() == 1 && st_[0].t == F_ROOT && st_[0].phase == 4; }

bool Grammar::State::accepts(const std::string& bytes) const {
    State copy = *this;
    for (unsigned char c : bytes) if (!copy.feed(c)) return false;
    return true;
}

// the opening '{' of a call was just read: expect "name":"<tool>" then ,"arguments":<args>}
bool Grammar::State::pushCall() {
    Frame name; name.t = F_NAME;
    st_.push_back(name);
    Frame lit; lit.t = F_LIT; lit.s = "\"name\":\"";
    st_.push_back(lit);
    return true;
}

static bool isDigit(uint8_t c) { return c >= '0' && c <= '9'; }

static bool prefixOfAny(const std::string& p, const std::vector<std::string>& names, const std::vector<char>* skip) {
    for (size_t i = 0; i < names.size(); i++) {
        if (skip && i < skip->size() && (*skip)[i]) continue;
        if (names[i].compare(0, p.size(), p) == 0) return true;
    }
    return false;
}

bool Grammar::State::feed(uint8_t c) {
    for (;;) {
        if (st_.empty()) return false;
        Frame& f = st_.back();
        switch (f.t) {
        case F_LIT:
            if (c != (uint8_t)f.s[f.pos]) return false;
            if (++f.pos == f.s.size()) st_.pop_back();
            return true;

        case F_NAME: {
            if (c == '"') {                                              // closing quote: must be a whole tool name
                int idx = -1;
                for (size_t i = 0; i < g_->tools_.size(); i++) if (g_->tools_[i].name == f.s) { idx = (int)i; break; }
                if (idx < 0) return false;
                st_.pop_back();
                Frame close; close.t = F_LIT; close.s = "}";
                st_.push_back(close);
                Frame args; args.t = F_VALUE; args.n = g_->tools_[idx].args.get();
                st_.push_back(args);
                Frame comma; comma.t = F_LIT; comma.s = ",\"arguments\":";
                st_.push_back(comma);
                return true;
            }
            std::vector<std::string> names;
            for (const Tool& t : g_->tools_) names.push_back(t.name);
            const std::string cand = f.s + (char)c;
            if (!prefixOfAny(cand, names, nullptr)) return false;
            f.s = cand;
            return true;
        }

        case F_ROOT:
            switch (f.phase) {
            case 0: if (c != '[') return false; f.phase = 1; return true;
            case 1:
                if (c == ']') { f.phase = 4; return true; }
                if (c == '{') { f.phase = 2; return pushCall(); }
                return false;
            case 2:
                if (c == ',') { f.phase = 3; return true; }
                if (c == ']') { f.phase = 4; return true; }
                return false;
            case 3: if (c != '{') return false; f.phase = 2; return pushCall();
            default: return false;
            }

        case F_VALUE: {
            const SNode* n = f.n;
            switch (n->kind) {
            case SNode::Object:
                if (c != '{') return false;
                f.t = F_OBJ; f.phase = 0; f.seen.assign(n->props.size(), 0);
                return true;
            case SNode::Array:
                if (c != '[') return false;
                f.t = F_ARR; f.phase = 0;
                return true;
            case SNode::String:
                if (c != '"') return false;
                f.t = n->enumVals.empty() ? F_STR : F_ENUMSTR; f.phase = 0; f.s.clear();
                return true;
            case SNode::Integer: f.t = F_INT; f.phase = 0; continue;                  // re-feed
            case SNode::Number: f.t = F_NUM; f.phase = 0; continue;
            case SNode::Boolean: f.t = F_BOOL; f.s.clear(); continue;
            }
            return false;
        }

        case F_OBJ: {
            const SNode* n = f.n;
            auto requiredDone = [&]() {
                for (const std::string& r : n->required) {
                    bool ok = false;
                    for (size_t i = 0; i < n->props.size(); i++) if (n->props[i].first == r && f.seen[i]) ok = true;
                    if (!ok && !std::any_of(n->props.begin(), n->props.end(), [&](const std::pair<std::string, std::shared_ptr<SNode>>& p) { return p.first == r; })) ok = true;  // required but undeclared: ignore
                    if (!ok) return false;
                }
                return true;
            };
            auto anyUnseen = [&]() {
                for (size_t i = 0; i < n->props.size(); i++) if (!f.seen[i]) return true;
                return false;
            };
            if (f.phase == 0 || f.phase == 3) {                           // expecting a key (or '}' right after '{')
                if (c == '}' && f.phase == 0 && requiredDone()) { st_.pop_back(); return true; }
                if (c == '"' && anyUnseen()) {
                    f.phase = 2;
                    Frame key; key.t = F_KEY; key.n = n;
                    st_.push_back(key);
                    return true;
                }
                return false;
            }
            // phase 2: after a member
            if (c == ',') { if (!anyUnseen()) return false; f.phase = 3; return true; }
            if (c == '}' && requiredDone()) { st_.pop_back(); return true; }
            return false;
        }

        case F_KEY: {
            const SNode* n = f.n;
            Frame& owner = st_[st_.size() - 2];                           // the OBJ frame below
            std::vector<std::string> names;
            for (const auto& p : n->props) names.push_back(p.first);
            if (c == '"') {
                int idx = -1;
                for (size_t i = 0; i < n->props.size(); i++) if (!owner.seen[i] && n->props[i].first == f.s) { idx = (int)i; break; }
                if (idx < 0) return false;
                owner.seen[idx] = 1;
                const SNode* valueNode = n->props[idx].second.get();
                st_.pop_back();
                Frame v; v.t = F_VALUE; v.n = valueNode;
                st_.push_back(v);
                Frame colon; colon.t = F_LIT; colon.s = ":";
                st_.push_back(colon);
                return true;
            }
            const std::string cand = f.s + (char)c;
            if (!prefixOfAny(cand, names, &owner.seen)) return false;
            f.s = cand;
            return true;
        }

        case F_ARR:
            if (f.phase == 0) {
                if (c == ']') { st_.pop_back(); return true; }
                f.phase = 1;
                Frame v; v.t = F_VALUE; v.n = f.n->items.get();
                st_.push_back(v);
                continue;
            }
            if (f.phase == 1) {
                if (c == ',') { f.phase = 2; return true; }
                if (c == ']') { st_.pop_back(); return true; }
                return false;
            }
            f.phase = 1;                                                  // phase 2: a value must follow the comma
            { Frame v; v.t = F_VALUE; v.n = f.n->items.get(); st_.push_back(v); }
            continue;

        case F_STR:
            if (f.phase == 0) {
                if (c == '"') { st_.pop_back(); return true; }
                if (c == '\\') { f.phase = 1; return true; }
                if (c < 0x20) return false;
                return true;
            }
            if (f.phase == 1) {
                if (std::strchr("\"\\/bfnrt", c) && c != 0) { f.phase = 0; return true; }
                if (c == 'u') { f.phase = 2; f.pos = 0; return true; }
                return false;
            }
            if (!(isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
            if (++f.pos == 4) f.phase = 0;
            return true;

        case F_ENUMSTR: {
            if (c == '"') {
                if (std::find(f.n->enumVals.begin(), f.n->enumVals.end(), f.s) == f.n->enumVals.end()) return false;
                st_.pop_back();
                return true;
            }
            const std::string cand = f.s + (char)c;
            if (!prefixOfAny(cand, f.n->enumVals, nullptr)) return false;
            f.s = cand;
            return true;
        }

        case F_INT:
        case F_NUM: {
            const bool num = f.t == F_NUM;
            // phases: 0 start, 1 after '-', 2 int digits, 3 leading zero, 4 after '.', 5 fraction digits,
            //         6 after e, 7 after e sign, 8 exponent digits
            switch (f.phase) {
            case 0: if (c == '-') { f.phase = 1; return true; }
                    if (c == '0') { f.phase = 3; return true; }
                    if (isDigit(c)) { f.phase = 2; return true; }
                    return false;
            case 1: if (c == '0') { f.phase = 3; return true; }
                    if (isDigit(c)) { f.phase = 2; return true; }
                    return false;
            case 2: if (isDigit(c)) return true; [[fallthrough]];
            case 3:
                if (f.phase == 3 && isDigit(c)) return false;
                if (num && c == '.') { f.phase = 4; return true; }
                if (num && (c == 'e' || c == 'E')) { f.phase = 6; return true; }
                st_.pop_back(); continue;                                   // number ended: the byte belongs to the parent
            case 4: if (isDigit(c)) { f.phase = 5; return true; } return false;
            case 5: if (isDigit(c)) return true;
                    if (c == 'e' || c == 'E') { f.phase = 6; return true; }
                    st_.pop_back(); continue;
            case 6: if (c == '+' || c == '-') { f.phase = 7; return true; }
                    if (isDigit(c)) { f.phase = 8; return true; }
                    return false;
            case 7: if (isDigit(c)) { f.phase = 8; return true; } return false;
            case 8: if (isDigit(c)) return true;
                    st_.pop_back(); continue;
            }
            return false;
        }

        case F_BOOL: {
            const std::string cand = f.s + (char)c;
            if (std::string("true").compare(0, cand.size(), cand) == 0 && cand.size() <= 4) {
                if (cand == "true") st_.pop_back(); else f.s = cand;
                return true;
            }
            if (std::string("false").compare(0, cand.size(), cand) == 0 && cand.size() <= 5) {
                if (cand == "false") st_.pop_back(); else f.s = cand;
                return true;
            }
            return false;
        }
        }
        return false;
    }
}

} // namespace needle
