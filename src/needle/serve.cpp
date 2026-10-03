#include "serve.h"
#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace needle {

static bool readRequest(int fd, std::string& method, std::string& path, std::string& body) {
    std::string buf;
    char tmp[4096];
    size_t headerEnd = std::string::npos;
    while (headerEnd == std::string::npos) {
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, (size_t)n);
        headerEnd = buf.find("\r\n\r\n");
        if (buf.size() > (1u << 20)) return false;
    }
    const size_t eol = buf.find("\r\n");
    const std::string line = buf.substr(0, eol);
    const size_t sp1 = line.find(' '), sp2 = line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    method = line.substr(0, sp1);
    path = line.substr(sp1 + 1, sp2 - sp1 - 1);

    size_t contentLength = 0;
    std::string headers = buf.substr(0, headerEnd);
    for (char& c : headers) c = (char)std::tolower((unsigned char)c);
    const size_t cl = headers.find("content-length:");
    if (cl != std::string::npos) contentLength = (size_t)std::strtoul(headers.c_str() + cl + 15, nullptr, 10);
    if (contentLength > (1u << 20)) return false;

    body = buf.substr(headerEnd + 4);
    while (body.size() < contentLength) {
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        body.append(tmp, (size_t)n);
    }
    body.resize(contentLength);
    return true;
}

static void respond(int fd, int code, const std::string& payload) {
    const char* status = code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" : "Error";
    char head[160];
    std::snprintf(head, sizeof(head),
                  "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                  code, status, payload.size());
    std::string msg = head + payload;
    size_t sent = 0;
    while (sent < msg.size()) {
        const ssize_t n = ::send(fd, msg.data() + sent, msg.size() - sent, 0);
        if (n <= 0) break;
        sent += (size_t)n;
    }
}

int serve(Session& session, int port) {
    const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) { std::perror("socket"); return 1; }
    int one = 1;
    ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);        // localhost only
    addr.sin_port = htons((uint16_t)port);
    if (::bind(ls, (sockaddr*)&addr, sizeof(addr)) < 0) { std::perror("bind"); return 1; }
    if (::listen(ls, 4) < 0) { std::perror("listen"); return 1; }
    std::fprintf(stderr, "needle serving on http://127.0.0.1:%d  (POST /complete {\"input\":\"...\"}, POST /reset)\n", port);

    for (;;) {
        const int fd = ::accept(ls, nullptr, nullptr);
        if (fd < 0) continue;
        std::string method, path, body;
        if (!readRequest(fd, method, path, body)) { respond(fd, 400, "{\"error\":\"bad request\"}"); ::close(fd); continue; }

        if (method == "POST" && path == "/complete") {
            std::string input;
            bool ok = false;
            try {
                const json::Value v = json::parse(body);
                if (const json::Value* in = v.get("input")) { input = in->s; ok = in->type == json::Value::String; }
            } catch (const std::exception&) {}
            if (!ok) respond(fd, 400, "{\"error\":\"expected {\\\"input\\\": \\\"...\\\"}\"}");
            else respond(fd, 200, session.complete(input).toJson(session.prefixTokens()));
        } else if (method == "POST" && path == "/reset") {
            session.reset();
            respond(fd, 200, "{\"ok\":true}");
        } else if (method == "GET" && path == "/") {
            respond(fd, 200, "{\"usage\":\"POST /complete {\\\"input\\\":\\\"...\\\"} or POST /reset\"}");
        } else {
            respond(fd, 404, "{\"error\":\"not found\"}");
        }
        ::close(fd);
    }
}

} // namespace needle
