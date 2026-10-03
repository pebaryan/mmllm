#pragma once
// Post-processing of the model's raw tool calls: argument repair and the gates that withhold a call.
//
// The shipped engine applies a longer, undocumented set of rules; these are the ones its observable
// behaviour (see tools/needle_battery.py) shows, implemented as plainly as possible:
//
//   repairs  - a paired tool follows the request's polarity (lock_door -> unlock_door when the request
//              says "unlock"); optional numbers / free strings the conversation never states are dropped;
//              title/label arguments start with a capital; H:MM times are zero-padded.
//   gates    - a call is withheld (moved to suppressed_calls) when a required number is never stated, when
//              the request negates the call ("don't turn on the lights"), when an origin equals its
//              destination, or when the confidence is below the 0.1 floor.
#include "json.h"
#include <string>
#include <vector>

namespace needle {

struct RepairOutcome {
    json::Value calls;                  // calls that go out
    json::Value suppressed;             // calls withheld by a gate
    std::vector<std::string> notes;     // what was repaired / why a call was withheld
};

// tools: the normalised tools array (name, description, parameters). conversation: all user text so far.
// confidence < 0 means "unknown" (the confidence floor is then skipped).
RepairOutcome repairCalls(const json::Value& rawCalls, const json::Value& tools,
                          const std::string& conversation, double confidence);

} // namespace needle
