#!/usr/bin/env python3
"""Battery of tool sets and queries. Collect the answers of either the official Needle runner or
mmllm --needle (both run in server mode) into a JSON file so they can be compared.

  python3 battery.py official out_official.json
  python3 battery.py mine     out_mine.json
"""
import json, os, subprocess, sys, time, urllib.error, urllib.request

HOME = "/tmp/needle-eval"
MM = os.path.expanduser("~/mmllm")

TOOLSETS = {
    "home": {
        "tools": [
            {"name": "set_lights", "description": "Turn a room's lights on or off and set the brightness",
             "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "on": {"type": "boolean"},
                            "brightness": {"type": "integer", "minimum": 0, "maximum": 100}}, "required": ["room"]}},
            {"name": "set_thermostat", "description": "Set the thermostat temperature in Celsius and the mode",
             "parameters": {"type": "object", "properties": {"temperature": {"type": "integer"},
                            "mode": {"type": "string", "enum": ["heat", "cool", "auto"]}}, "required": ["temperature"]}},
            {"name": "lock_door", "description": "Lock a door",
             "parameters": {"type": "object", "properties": {"door": {"type": "string"}}, "required": ["door"]}},
            {"name": "unlock_door", "description": "Unlock a door",
             "parameters": {"type": "object", "properties": {"door": {"type": "string"}}, "required": ["door"]}},
        ],
        "queries": [
            "turn on the kitchen lights", "turn off the bedroom lights", "dim the hallway to 20 percent",
            "make it 21 degrees", "switch the heating to cool mode at 18", "lock the front door",
            "unlock the garage door", "set the living room lights to 60 and make it 23 degrees",
            "it is too cold in here", "don't turn on the lights", "turn off all the lights in the house",
            "put the thermostat on auto", "what time is it", "brighten the porch",
        ],
    },
    "assistant": {
        "tools": [
            {"name": "send_message", "description": "Send a text message to a contact",
             "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}}, "required": ["to", "text"]}},
            {"name": "create_event", "description": "Create a calendar event",
             "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "date": {"type": "string"},
                            "time": {"type": "string"}, "duration_minutes": {"type": "integer"}}, "required": ["title", "date"]}},
            {"name": "set_alarm", "description": "Set an alarm",
             "parameters": {"type": "object", "properties": {"time": {"type": "string"}, "label": {"type": "string"}}, "required": ["time"]}},
            {"name": "set_timer", "description": "Start a countdown timer",
             "parameters": {"type": "object", "properties": {"minutes": {"type": "integer"}}, "required": ["minutes"]}},
        ],
        "queries": [
            "text Anna that I will be late", "send a message to Bob saying see you at 5",
            "set an alarm for 6:30", "wake me up at 7 tomorrow", "start a 10 minute timer",
            "set a timer for 45 minutes", "add a dentist appointment on 2026-03-04 at 14:00 for 30 minutes",
            "create a meeting called planning on Friday", "message mom happy birthday",
            "set an alarm for 6am labeled gym and text Dave I am up", "cancel my timer", "tell me a joke",
        ],
    },
    "media": {
        "tools": [
            {"name": "play_music", "description": "Play a song or an artist",
             "parameters": {"type": "object", "properties": {"artist": {"type": "string"}, "song": {"type": "string"}}, "required": []}},
            {"name": "set_volume", "description": "Set the playback volume",
             "parameters": {"type": "object", "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100}}, "required": ["level"]}},
            {"name": "next_track", "description": "Skip to the next track", "parameters": {"type": "object", "properties": {}}},
            {"name": "pause", "description": "Pause playback", "parameters": {"type": "object", "properties": {}}},
        ],
        "queries": [
            "play some Beatles", "play Yesterday by The Beatles", "turn the volume up to 80", "mute it",
            "skip this song", "pause the music", "volume 30", "play jazz and set the volume to 40",
            "stop", "make it louder", "what is playing",
        ],
    },
    "maps": {
        "tools": [
            {"name": "get_directions", "description": "Get directions between two places",
             "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                            "mode": {"type": "string", "enum": ["driving", "walking", "transit", "cycling"]}}, "required": ["origin", "destination"]}},
            {"name": "search_places", "description": "Search for places near the user",
             "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "open_now": {"type": "boolean"}}, "required": ["query"]}},
        ],
        "queries": [
            "directions from Paris to Lyon", "how do I walk from the station to the museum",
            "find a pizza place near me", "find coffee shops that are open now", "take me to the airport by bus",
            "cycling route from Berlin to Potsdam", "nearest pharmacy", "what is the capital of France",
            "directions to the hospital", "drive from Madrid to Barcelona",
        ],
    },
}


def start(kind, port, toolfile, log):
    if kind == "official":
        cmd = ["./needle", "--model", "needle3.cact", "--tools", toolfile, "--serve", "--port", str(port)]
        cwd = HOME
    else:
        cmd = [os.path.join(MM, "build/mmllm"), "--needle", os.path.join(MM, "models/needle3.cact"),
               "--tools", toolfile, "--serve", "--port", str(port)]
        cwd = MM
    return subprocess.Popen(cmd, cwd=cwd, stdout=open(log, "w"), stderr=subprocess.STDOUT)


def post(port, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    kind, out = sys.argv[1], sys.argv[2]
    results = {}
    port = 18200 if kind == "official" else 18300
    for name, ts in TOOLSETS.items():
        toolfile = "/tmp/battery_tools_%s.json" % name
        json.dump(ts["tools"], open(toolfile, "w"))
        proc = start(kind, port, toolfile, "/tmp/battery_%s_%s.log" % (kind, name))
        for _ in range(120):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=2); break
            except urllib.error.HTTPError:
                break                      # any HTTP answer means the server is up
            except Exception:
                time.sleep(1)
        rows = []
        for q in ts["queries"]:
            try:
                post(port, "/reset")
                r = post(port, "/complete", {"input": q})
            except Exception as e:
                r = {"error": repr(e)}
            rows.append({"query": q, "response": r})
            print("%-9s %-62s -> %s" % (name, q, json.dumps(r.get("function_calls"), separators=(",", ":"))[:90]), flush=True)
        results[name] = rows
        proc.terminate(); proc.wait()
        port += 1
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
