#include "repair.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>

namespace needle {

namespace {

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::vector<std::string> words(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (unsigned char c : lower(s)) {
        if (std::isalnum(c) || c == '\'') cur += (char)c;
        else if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// All numbers stated in the text: digit runs (with an optional decimal part) and spelled-out numbers.
std::vector<double> numbersIn(const std::string& text) {
    std::vector<double> out;
    const std::string t = lower(text);
    for (size_t i = 0; i < t.size();) {
        if (std::isdigit((unsigned char)t[i])) {
            size_t j = i;
            while (j < t.size() && std::isdigit((unsigned char)t[j])) j++;
            if (j + 1 < t.size() && t[j] == '.' && std::isdigit((unsigned char)t[j + 1])) {
                j++;
                while (j < t.size() && std::isdigit((unsigned char)t[j])) j++;
            }
            out.push_back(std::atof(t.substr(i, j - i).c_str()));
            i = j;
        } else i++;
    }
    static const std::map<std::string, double> w = {
        {"zero", 0}, {"one", 1}, {"two", 2}, {"three", 3}, {"four", 4}, {"five", 5}, {"six", 6}, {"seven", 7},
        {"eight", 8}, {"nine", 9}, {"ten", 10}, {"eleven", 11}, {"twelve", 12}, {"thirteen", 13},
        {"fourteen", 14}, {"fifteen", 15}, {"sixteen", 16}, {"seventeen", 17}, {"eighteen", 18},
        {"nineteen", 19}, {"twenty", 20}, {"thirty", 30}, {"forty", 40}, {"fifty", 50}, {"sixty", 60},
        {"seventy", 70}, {"eighty", 80}, {"ninety", 90}, {"hundred", 100}, {"half", 0.5}, {"a", -1}};
    for (const std::string& word : words(text)) {
        auto it = w.find(word);
        if (it != w.end() && it->second >= 0) out.push_back(it->second);
    }
    return out;
}

bool numberStated(double v, const std::vector<double>& stated) {
    for (double s : stated) if (std::fabs(s - v) < 1e-9) return true;
    return false;
}

bool isNumeric(const json::Value& v) { return v.type == json::Value::Number; }

const json::Value* findTool(const json::Value& tools, const std::string& name) {
    for (const json::Value& t : tools.arr) {
        const json::Value* n = t.get("name");
        if (n && n->s == name) return &t;
    }
    return nullptr;
}

const json::Value* propertyOf(const json::Value* tool, const std::string& key) {
    if (!tool) return nullptr;
    const json::Value* params = tool->get("parameters");
    if (!params) return nullptr;
    const json::Value* props = params->get("properties");
    return props ? props->get(key) : nullptr;
}

bool isRequired(const json::Value* tool, const std::string& key) {
    if (!tool) return false;
    const json::Value* params = tool->get("parameters");
    if (!params) return false;
    const json::Value* req = params->get("required");
    if (!req) return false;
    for (const json::Value& r : req->arr) if (r.s == key) return true;
    return false;
}

bool hasDefault(const json::Value* prop) { return prop && prop->get("default"); }

std::string typeOf(const json::Value* prop) {
    if (!prop) return "";
    const json::Value* t = prop->get("type");
    return t && t->type == json::Value::String ? t->s : "";
}

// ---- polarity pairs: a tool and its opposite are named by an antonym verb pair, e.g. lock_door / unlock_door
const std::vector<std::pair<std::string, std::string>>& antonyms() {
    static const std::vector<std::pair<std::string, std::string>> a = {
        {"lock", "unlock"}, {"open", "close"}, {"enable", "disable"}, {"start", "stop"}, {"mute", "unmute"},
        {"arm", "disarm"}, {"show", "hide"}, {"increase", "decrease"}, {"raise", "lower"}, {"turn_on", "turn_off"},
        {"activate", "deactivate"}, {"connect", "disconnect"}, {"play", "pause"}};
    return a;
}

std::string replaceVerb(const std::string& name, const std::string& from, const std::string& to) {
    if (name.compare(0, from.size(), from) == 0 && (name.size() == from.size() || name[from.size()] == '_'))
        return to + name.substr(from.size());
    return "";
}

// Does the request use this verb? Single verbs match at the start of a word ("lock" does not match "unlock");
// two-word verbs such as turn_on match as a phrase ("turn on").
bool says(const std::string& conversation, const std::vector<std::string>& qwords, const std::string& verb) {
    if (verb.find('_') != std::string::npos) {
        std::string phrase = verb;
        std::replace(phrase.begin(), phrase.end(), '_', ' ');
        return lower(conversation).find(phrase) != std::string::npos;
    }
    for (const std::string& w : qwords) if (w.compare(0, verb.size(), verb) == 0) return true;
    return false;
}

std::string pairedTool(const std::string& name, const std::string& conversation,
                       const std::vector<std::string>& qwords, const json::Value& tools) {
    for (const auto& ap : antonyms()) {
        for (int dir = 0; dir < 2; dir++) {
            const std::string& mine = dir == 0 ? ap.first : ap.second;
            const std::string& other = dir == 0 ? ap.second : ap.first;
            const std::string sibling = replaceVerb(name, mine, other);
            if (sibling.empty() || !findTool(tools, sibling)) continue;
            // The request names the sibling's verb but not this tool's own verb -> switch to the sibling.
            if (says(conversation, qwords, other) && !says(conversation, qwords, mine)) return sibling;
        }
    }
    return "";
}

// ---- negation: "don't turn on the lights" excludes a call about the lights
bool negatesCall(const std::string& conversation, const json::Value& call) {
    static const std::set<std::string> neg = {"don't", "dont", "do", "never", "without", "avoid", "no"};
    const std::vector<std::string> w = words(conversation);
    std::set<std::string> topic;
    {   // words of the tool name (split on '_') and of the string arguments
        std::string name = call.get("name") ? call.get("name")->s : "";
        std::replace(name.begin(), name.end(), '_', ' ');
        for (const std::string& part : words(name)) topic.insert(part);
        if (const json::Value* a = call.get("arguments"))
            for (const auto& kv : a->obj) if (kv.second.type == json::Value::String)
                for (const std::string& part : words(kv.second.s)) topic.insert(part);
    }
    static const std::set<std::string> stop = {"set", "get", "the", "a", "an", "to", "of", "on", "off", "in", "it", "my", "me", "is"};
    for (size_t i = 0; i < w.size(); i++) {
        bool isNeg = neg.count(w[i]) > 0;
        if (w[i] == "do") isNeg = i + 1 < w.size() && w[i + 1] == "not";
        if (w[i] == "no") isNeg = i + 1 < w.size() && (w[i + 1] == "need" || w[i + 1] == "lights" || w[i + 1] == "light");
        if (!isNeg) continue;
        for (size_t j = i + 1; j < w.size() && j <= i + 5; j++)
            if (!stop.count(w[j]) && topic.count(w[j]) && w[j] != "not") return true;
    }
    return false;
}

std::string capitalise(std::string s) {
    if (!s.empty() && std::islower((unsigned char)s[0])) s[0] = (char)std::toupper((unsigned char)s[0]);
    return s;
}

} // namespace

RepairOutcome repairCalls(const json::Value& rawCalls, const json::Value& tools,
                          const std::string& conversation, double confidence) {
    RepairOutcome out;
    out.calls.type = json::Value::Array;
    out.suppressed.type = json::Value::Array;
    const std::vector<std::string> qwords = words(conversation);
    const std::vector<double> stated = numbersIn(conversation);

    std::vector<json::Value> kept;
    for (const json::Value& original : rawCalls.arr) {
        json::Value call = original;
        json::Value* nameV = call.get("name");
        json::Value* argsV = call.get("arguments");
        if (!nameV || !argsV || argsV->type != json::Value::Object) { kept.push_back(call); continue; }

        // 1. polarity: lock_door -> unlock_door when the request says "unlock"
        const std::string sibling = pairedTool(nameV->s, conversation, qwords, tools);
        if (!sibling.empty()) {
            out.notes.push_back("tool " + nameV->s + " -> " + sibling + " (request polarity)");
            nameV->s = sibling;
        }
        const json::Value* tool = findTool(tools, nameV->s);

        // 2. drop optional numbers / free strings the conversation never states
        for (size_t i = 0; i < argsV->obj.size();) {
            const std::string key = argsV->obj[i].first;
            const json::Value& val = argsV->obj[i].second;
            const json::Value* prop = propertyOf(tool, key);
            bool drop = false;
            if (!isRequired(tool, key)) {
                if (isNumeric(val)) drop = !numberStated(std::atof(val.s.c_str()), stated) && !hasDefault(prop);
                else if (val.type == json::Value::String && prop && !prop->get("enum") && typeOf(prop) == "string") {
                    drop = val.s.empty() || lower(conversation).find(lower(val.s)) == std::string::npos;
                }
            }
            if (drop) { out.notes.push_back("dropped ungrounded optional '" + key + "'"); argsV->obj.erase(argsV->obj.begin() + i); }
            else i++;
        }

        // 3. formatting
        for (auto& kv : argsV->obj) {
            if (kv.second.type != json::Value::String) continue;
            if (kv.first == "title" || kv.first == "label" || kv.first == "subject" || kv.first == "summary")
                kv.second.s = capitalise(kv.second.s);
            if (kv.first == "time" && kv.second.s.size() == 4 && std::isdigit((unsigned char)kv.second.s[0]) &&
                kv.second.s[1] == ':' && std::isdigit((unsigned char)kv.second.s[2]) && std::isdigit((unsigned char)kv.second.s[3]))
                kv.second.s = "0" + kv.second.s;
        }

        // 4. gates
        std::string why;
        std::vector<std::string> required;
        if (tool && tool->get("parameters") && tool->get("parameters")->get("required"))
            for (const json::Value& r : tool->get("parameters")->get("required")->arr) required.push_back(r.s);
        for (const std::string& req : required) {
            const json::Value* v = argsV->get(req);
            const json::Value* prop = propertyOf(tool, req);
            if (v && isNumeric(*v) && !hasDefault(prop) && !numberStated(std::atof(v->s.c_str()), stated))
                why = "required number '" + req + "' is never stated";
        }
        if (why.empty() && negatesCall(conversation, call)) why = "the request negates this call";
        if (why.empty()) {
            const json::Value* o = argsV->get("origin"); const json::Value* d = argsV->get("destination");
            if (!o) { o = argsV->get("from"); d = argsV->get("to"); }
            if (o && d && o->type == json::Value::String && d->type == json::Value::String && lower(o->s) == lower(d->s))
                why = "origin equals destination";
        }
        if (why.empty() && confidence >= 0 && confidence < 0.1) why = "confidence below the 0.1 floor";
        if (!why.empty()) {
            out.notes.push_back("withheld " + nameV->s + ": " + why);
            out.suppressed.arr.push_back(call);
        } else {
            kept.push_back(call);
        }
    }
    out.calls.arr = kept;
    return out;
}

} // namespace needle
