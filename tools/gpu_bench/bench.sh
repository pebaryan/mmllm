#!/bin/bash
# Repeatable benchmark: waits for an idle CPU, runs each case 3 times, prints best tok/s and the load seen.
cd ~/mmllm
idle_wait() { for i in $(seq 1 60); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l=$l 'BEGIN{exit !(l<0.25)}' && return; sleep 5; done; }
best() { # $1 = env, $2 = model, $3 = tokens
  local b=0
  for r in 1 2 3; do
    idle_wait
    t=$(env $1 timeout 280 ./build/mmllm --model models/$2 --prompt "42,128,256" --tokens $3 --temperature 0 2>&1 | grep -o "([0-9.]* tok/s)" | tr -d "(a-z/ )")
    b=$(awk -v a=$b -v c=${t:-0} 'BEGIN{print (c>a)?c:a}')
  done; echo $b
}
echo "load now: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "3M    best: $(best X=1 tinystories-3m.mlm 30)"
echo "33M   best: $(best X=1 tinystories-33m.mlm 20)"
echo "GPT-2 CPU-LM best: $(best X=1 gpt2.mlm 20)"
[ "${SKIP_GPULM:-0}" = 1 ] && echo "GPT-2 GPU-LM best: skipped (256 MB card)" || echo "GPT-2 GPU-LM best: $(best 'MMLLM_GPU_BUDGET_MB=800 MMLLM_LM=gpu' gpt2.mlm 20)"
nd=0; for r in 1 2 3; do idle_wait; d=$(./build/mmllm --needle models/needle3.cact --tools ~/tools_home.json --prompt "turn on the kitchen lights" 2>&1 | grep -o 'decode_tps":[0-9.]*' | cut -d: -f2); nd=$(awk -v a=$nd -v c=${d:-0} 'BEGIN{print (c>a)?c:a}'); done
echo "Needle CPU decode best: $nd"
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
