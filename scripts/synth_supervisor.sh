#!/bin/zsh
# Restart the (resumable) F5 synthesis until every segment is done.
# MPS/Metal kills the process at random every few hundred segments;
# each restart skips already-synthesized files. Guards against a
# crash-loop: 5 deaths within a minute each => give up.
WORK="$1"
TOTAL="$2"
cd "$(dirname "$0")/.."
quick_deaths=0
# if a synthesis for this work dir is already running, babysit it:
# wait for it to exit before deciding whether a restart is needed
while pgrep -f "synthesize_f5.py $WORK" > /dev/null; do sleep 30; done
while [ "$(ls "$WORK/tts_f5" 2>/dev/null | wc -l | tr -d ' ')" -lt "$TOTAL" ]; do
  t0=$(date +%s)
  venv-f5/bin/python scripts/synthesize_f5.py "$WORK" >> "$WORK/synth.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  echo "[supervisor] synth exited rc=$rc after ${dt}s ($(ls "$WORK/tts_f5" | wc -l | tr -d ' ')/$TOTAL)" >> "$WORK/synth.log"
  if [ "$rc" -eq 0 ]; then break; fi
  if [ "$dt" -lt 60 ]; then
    quick_deaths=$((quick_deaths + 1))
    if [ "$quick_deaths" -ge 5 ]; then
      echo "[supervisor] crash-looping, giving up" >> "$WORK/synth.log"
      exit 1
    fi
  else
    quick_deaths=0
  fi
  sleep 10
done
echo "[supervisor] DONE: $(ls "$WORK/tts_f5" | wc -l | tr -d ' ')/$TOTAL" >> "$WORK/synth.log"
