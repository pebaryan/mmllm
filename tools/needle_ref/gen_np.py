"""Greedy tool-call generation with the NumPy incremental model on the dequantised .cact weights,
compared with the answers the official engine gave on the Mac mini (same tools, same queries)."""
import json, re, sys, time
import numpy as np
sys.path.insert(0, ".")
from needle.model.export import parse_tokenizer_blob, RefTokenizer
import needle_np as N, needle_inc as I

TOOLS = json.loads('[{"name":"set_lights","description":"Turn a room lights on or off and set the brightness","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer","minimum":0,"maximum":100}},"required":["room"]}},{"name":"set_thermostat","description":"Set the thermostat temperature in Celsius","parameters":{"type":"object","properties":{"temperature":{"type":"integer"}},"required":["temperature"]}}]')

# What the official engine (x86-64 runner on the Mac mini) returned: (query, calls, reasoning)
OFFICIAL = [
    ("dim the living room to 30",
     [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 30}}],
     "room 'living room' from query; brightness 30 from '30'"),
    ("make it 22 degrees in here",
     [{"name": "set_thermostat", "arguments": {"temperature": 22}}],
     "'22 degrees' -> temperature 22. No unit specified, default to Celsius."),
    ("who won the world cup in 1998?", [],
     "No tool available for sports data or world coverage."),
    ("set the temperature to 19",
     [{"name": "set_thermostat", "arguments": {"temperature": 19}}], None),
    ("turn off the kitchen lights",
     [{"name": "set_lights", "arguments": {"room": "kitchen", "on": False}}], None),
]


def render_prompt(query, tools, system=""):
    tools_json = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    prefix = "<|im_start|>system\n" + system + "<|im_end|>\n" if system else ""
    return (prefix + "<|im_start|>user\n<tools>" + tools_json + "</tools>\n" + query
            + "<|im_end|>\n<|im_start|>assistant\n")


def generate(W, cfg, tok, prompt_text, max_new=220, quant=False, verbose=False):
    ids = [2] + tok.encode(prompt_text)
    cos, sin = N.rope_tables(cfg.qk, len(ids) + max_new + 1, cfg.rope_theta)
    st = I.State(cfg)
    logits = None
    for t, i in enumerate(ids):
        logits = I.step(W, cfg, st, i, cos[t], sin[t], quant)
    first = int(np.argmax(logits))
    out, pos = [], len(ids)
    for _ in range(max_new):
        nxt = int(np.argmax(logits))
        if nxt in (1, 5):            # </s> or <|im_end|>
            break
        out.append(nxt)
        logits = I.step(W, cfg, st, nxt, cos[pos], sin[pos], quant)
        pos += 1
    return ids, out, first


def parse(text):
    think = re.search(r"<think>\s*(.*?)\s*</think>", text, re.S)
    call = re.search(r"<tool_call>(.*?)</tool_call>", text, re.S)
    calls = None
    if call:
        try: calls = json.loads(call.group(1))
        except Exception: calls = "UNPARSEABLE: " + call.group(1)
    return (think.group(1) if think else None), calls


if __name__ == "__main__":
    quant = "--quant" in sys.argv
    d = np.load("../weights/needle3_deq.npz")
    W = {k: d[k] for k in d.files}
    tok = RefTokenizer(parse_tokenizer_blob(open("../weights/needle3_tok.bin", "rb").read()))
    cfg = N.Cfg()
    print("quant (activation fake-quant) =", quant)
    ok_calls = ok_reason = 0
    for query, want_calls, want_reason in OFFICIAL:
        t0 = time.time()
        ids, out, first = generate(W, cfg, tok, render_prompt(query, TOOLS), quant=quant)
        text = tok.decode(out)
        reasoning, calls = parse(text)
        same_calls = calls == want_calls
        same_reason = (want_reason is None) or (reasoning == want_reason)
        ok_calls += same_calls; ok_reason += (want_reason is not None and same_reason)
        print("\nquery: %r  [%d prompt tokens, %d generated, %.1fs, first token id %d (%s)]" % (
            query, len(ids), len(out), time.time() - t0, first, tok.pieces[first]))
        print("  mine    : %s | %s" % (json.dumps(calls), reasoning))
        print("  official: %s | %s" % (json.dumps(want_calls), want_reason))
        print("  calls %s, reasoning %s" % ("MATCH" if same_calls else "DIFFER",
              "n/a" if want_reason is None else ("MATCH" if same_reason else "DIFFER")))
    print("\nSUMMARY: calls match %d/%d; reasoning text identical %d/%d" % (
        ok_calls, len(OFFICIAL), ok_reason, sum(1 for _, _, r in OFFICIAL if r)))
