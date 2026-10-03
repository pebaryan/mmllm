#!/bin/bash
# Compare the official Needle runner with mmllm --needle on a set of queries (same tools.json).
cd ~/mmllm
T=/tmp/needle-eval/tools.json
same=0; callsame=0; total=0

oneline='import sys,json; j=json.load(sys.stdin); print(j["reasoning"] + " ## " + json.dumps(j["function_calls"], separators=(",", ":")))'

QUERIES=(
  "dim the living room to 30"
  "make it 22 degrees in here"
  "who won the world cup in 1998?"
  "set the temperature to 19"
  "turn off the kitchen lights"
  "set the lights in the bedroom to 80 percent"
  "what is the weather like"
  "turn on the porch light"
  "warm it up to 25"
  "play some jazz"
)

for q in "${QUERIES[@]}"; do
  O=$(cd /tmp/needle-eval && timeout 120 ./needle --model needle3.cact --tools tools.json --prompt "$q" 2>/dev/null | python3 -c "$oneline")
  M=$(timeout 120 ./build/mmllm --needle models/needle3.cact --tools $T --prompt "$q" 2>/dev/null | python3 -c "$oneline")
  RO="${O%% ##*}"; RM="${M%% ##*}"; CO="${O##*## }"; CM="${M##*## }"
  total=$((total+1))
  tr="reasoning!"; tc="calls!"
  [ "$RO" = "$RM" ] && { same=$((same+1)); tr="reasoning="; }
  [ "$CO" = "$CM" ] && { callsame=$((callsame+1)); tc="calls="; }
  echo "[$tr $tc] $q"
  if [ "$O" != "$M" ]; then
    echo "     official: $O"
    echo "     mmllm   : $M"
  fi
done
echo
echo "reasoning text identical: $same/$total   function_calls identical: $callsame/$total"
echo
for th in 1 2; do
  printf "threads=%s  " $th
  timeout 120 ./build/mmllm --needle models/needle3.cact --tools $T --threads $th --prompt "dim the living room to 30" 2>/dev/null \
    | python3 -c 'import sys,json; j=json.load(sys.stdin); print("prefill %.1f tok/s, decode %.1f tok/s" % (j["prefill_tps"], j["decode_tps"]))'
done
