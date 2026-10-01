#!/usr/bin/env bash
# Downloads the input videos of the demo scene into data/input_video and verifies them.
set -euo pipefail
cd "$(dirname "$0")/data/input_video"
URL=https://r2.nazarenus.dev/gaussians_on_fire/eccv_demo/input_video
for f in 0011_0.mkv 0011_1.mkv 0011_2.mkv; do
    curl -fL --retry 3 -o "$f" "$URL/$f"
done
sha256sum -c <<'SUMS'
0a477cce90874a2c4ad833dc91928cb85fc44de2a1d216b0f67f4954832d9232  0011_0.mkv
8b56fb0cb4ae90be3feef263e397db54522a657502433feeb36ee385a4abe531  0011_1.mkv
c9143dbec8d97f1ab220d38b6de2041f1de4edb2cbf8d5cbc1455e5c84116c3f  0011_2.mkv
SUMS
