#!/usr/bin/env python3
"""Compare two battery result files (official vs mine): reasoning, calls, suppressed calls, confidence."""
import json, sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
la = sys.argv[3] if len(sys.argv) > 3 else "official"
lb = sys.argv[4] if len(sys.argv) > 4 else "mine"


def calls(r):
    return json.dumps(r.get("function_calls"), separators=(",", ":"), sort_keys=True)


tot = rs = cs = cf = 0
confdiffs = []
for name in a:
    print("=== %s" % name)
    for ra, rb in zip(a[name], b[name]):
        A, B = ra["response"], rb["response"]
        tot += 1
        same_r = A.get("reasoning") == B.get("reasoning")
        same_c = calls(A) == calls(B)
        rs += same_r; cs += same_c
        tag = ("R" if same_r else "r") + ("C" if same_c else "c")
        if A.get("confidence") is not None and B.get("confidence") is not None:
            confdiffs.append(abs(A["confidence"] - B["confidence"]))
        if same_r and same_c:
            print("  [%s] %s" % (tag, ra["query"]))
            continue
        print("  [%s] %s" % (tag, ra["query"]))
        for lab, X in ((la, A), (lb, B)):
            print("      %-9s calls=%s  supp=%s  conf=%s" % (lab, calls(X)[:150],
                  json.dumps(X.get("suppressed_calls"), separators=(",", ":"))[:80], X.get("confidence")))
            print("      %-9s why  =%s" % ("", X.get("reasoning")))
print()
print("reasoning identical: %d/%d   calls identical: %d/%d" % (rs, tot, cs, tot))
if confdiffs:
    print("confidence: mean |diff| %.4f over %d responses" % (sum(confdiffs) / len(confdiffs), len(confdiffs)))
