#!/bin/zsh
# Keep the (resumable) F5 synthesis running until it says it is done.
# MPS/Metal kills the process at random every few hundred segments; each
# restart skips clips that are present AND fresh per tts_f5/manifest.json
# (synthesize_f5.py exits 0 once nothing is left to do, non-zero on a
# crash or on a consistency failure). Guards against a crash-loop: 5
# deaths within a minute each => give up — a persistent non-zero exit
# means a REAL error (e.g. stale accents), look at the end of synth.log.
WORK="$1"
cd "$(dirname "$0")/.."
quick_deaths=0
# if a synthesis for this work dir is already running, babysit it:
# wait for it to exit before deciding whether a restart is needed
while pgrep -f "synthesize_f5.py $WORK" > /dev/null; do sleep 30; done
while true; do
  t0=$(date +%s)
  venv-f5/bin/python scripts/synthesize_f5.py "$WORK" >> "$WORK/synth.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  echo "[supervisor] synth exited rc=$rc after ${dt}s" >> "$WORK/synth.log"
  if [ "$rc" -eq 0 ]; then break; fi
  if [ "$dt" -lt 60 ]; then
    quick_deaths=$((quick_deaths + 1))
    if [ "$quick_deaths" -ge 5 ]; then
      echo "[supervisor] crash-looping (rc=$rc), giving up — see above" >> "$WORK/synth.log"
      exit 1
    fi
  else
    quick_deaths=0
  fi
  sleep 10
done
echo "[supervisor] DONE" >> "$WORK/synth.log"
