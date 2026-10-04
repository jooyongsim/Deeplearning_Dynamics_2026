#!/usr/bin/env bash
# Copy this project's Claude Code session transcripts (*.jsonl) into ./claude/
# as <machine>_<session-id>.jsonl. Only newer/changed files are copied.
#
# Usage: tools/sync_claude_logs.sh [machine-prefix] [project-dir]
#   machine-prefix  default: dgx01
#   project-dir     dir Claude Code was launched in (default: parent of this repo)
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
prefix="${1:-dgx01}"
project="$(cd "${2:-$repo/..}" && pwd)"

# Claude Code stores transcripts under ~/.claude/projects/<path with / and _ -> ->
src="$HOME/.claude/projects/$(echo "$project" | sed 's|[/_.]|-|g')"
dst="$repo/claude"

[ -d "$src" ] || { echo "No transcripts at $src" >&2; exit 1; }
mkdir -p "$dst"

for f in "$src"/*.jsonl; do
  [ -e "$f" ] || continue
  out="$dst/${prefix}_$(basename "$f")"
  if [ ! -e "$out" ] || [ "$f" -nt "$out" ]; then
    cp -p "$f" "$out"
    echo "synced  $(basename "$out")"
  else
    echo "current $(basename "$out")"
  fi
done
