#!/usr/bin/env bash
set -euo pipefail

repo="/home/al/opt/pyquante22"
ts="$(date +%Y%m%d_%H%M%S)"
out_dir="./dev_scripts/pyspy"
mkdir -p "$out_dir"
raw="${out_dir}/pyspy.${ts}.raw"
folded="${out_dir}/pyspy.${ts}.repo.folded"
svg="${out_dir}/pyspy.${ts}.repo.svg"
speedscope="${out_dir}/pyspy.${ts}.repo.speedscope.json"

py-spy record --format raw --full-filenames -o "$raw" -- \
  python -m pyquante22.basis.integral_engines || true

if [[ ! -s "$raw" ]]; then
  echo "No raw output at $raw"
  exit 1
fi

REPO_PATH="$repo" RAW_PATH="$raw" FOLDED_PATH="$folded" SPEEDSCOPE_PATH="$speedscope" \
python - <<'PY'
import json
import os
repo = os.environ["REPO_PATH"]
inp = os.environ["RAW_PATH"]
folded_out = os.environ["FOLDED_PATH"]
speed_out = os.environ["SPEEDSCOPE_PATH"]

def parse_raw(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            stack, _, count = line.rpartition(" ")
            try:
                count_val = int(count)
            except ValueError:
                continue
            frames = [fr for fr in stack.split(";") if repo in fr]
            if frames:
                yield frames, count_val

frames_index = {}
frames_list = []
samples = []
weights = []

with open(folded_out, "w") as g:
    for frames, count in parse_raw(inp):
        g.write(";".join(frames) + " " + str(count) + "\n")
        idxs = []
        for fr in frames:
            idx = frames_index.get(fr)
            if idx is None:
                idx = len(frames_list)
                frames_index[fr] = idx
                frames_list.append({"name": fr})
            idxs.append(idx)
        samples.append(idxs)
        weights.append(count)

profile = {
    "$schema": "https://www.speedscope.app/file-format-schema.json",
    "shared": {"frames": frames_list},
    "profiles": [
        {
            "type": "sampled",
            "name": "py-spy repo-only",
            "unit": "samples",
            "startValue": 0,
            "endValue": sum(weights),
            "samples": samples,
            "weights": weights,
        }
    ],
    "activeProfileIndex": 0,
}

with open(speed_out, "w") as f:
    json.dump(profile, f)
PY

if command -v flamegraph.pl >/dev/null 2>&1; then
  flamegraph.pl "$folded" > "$svg"
  echo "Wrote $svg"
else
  echo "flamegraph.pl not found; skipping SVG."
fi
echo "Wrote $speedscope"
