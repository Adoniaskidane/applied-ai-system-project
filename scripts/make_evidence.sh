#!/usr/bin/env bash
# Regenerate every file in evidence/ from a clean run. Offline: no API key needed.
#   bash scripts/make_evidence.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p evidence

run() {  # run <outfile> <command...>: record the command and its full output
  local out="evidence/$1"; shift
  # Replace the absolute repo path with "." so no local paths end up in the repo.
  { echo "\$ $*"; echo; "$@" 2>&1; } | sed "s|$PWD|.|g" > "$out"
  echo "wrote $out"
}

run pytest_output.txt       python -m pytest -v
run eval_offline.txt        python scripts/eval_rag.py
run guardrail_demo.txt      python scripts/guardrail_demo.py
run cli_output.txt          python src/main.py

# Real LLM requests from manual sessions (logs/ is git-ignored, so keep a copy here).
if [ -f logs/tunejudge.jsonl ]; then
  cp logs/tunejudge.jsonl evidence/ai_requests_log.jsonl
  echo "wrote evidence/ai_requests_log.jsonl"
fi
