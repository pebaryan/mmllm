#pragma once
#include "session.h"

namespace needle {

// Minimal blocking HTTP/1.1 server bound to 127.0.0.1 only (one connection at a time):
//   POST /complete  {"input": "..."}   -> the same JSON object as the CLI
//   POST /reset                        -> forgets the conversation, keeps the tools
//   GET  /                             -> a one-line usage message
// Returns a process exit code (only returns on a fatal socket error).
int serve(Session& session, int port);

} // namespace needle
