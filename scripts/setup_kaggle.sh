#!/usr/bin/env bash
set -euo pipefail

# Setup Kaggle credentials non-interactively.
# Usage options:
#   1) From environment variables:
#        export KAGGLE_USERNAME=your_user
#        export KAGGLE_KEY=your_key
#        scripts/setup_kaggle.sh
#   2) From a file path containing JSON credentials:
#        scripts/setup_kaggle.sh /path/to/kaggle.json

KDIR="$HOME/.kaggle"
KJSON="$KDIR/kaggle.json"

mkdir -p "$KDIR"

if [[ $# -ge 1 ]]; then
  SRC="$1"
  if [[ ! -f "$SRC" ]]; then
    echo "Provided path does not exist: $SRC" >&2
    exit 1
  fi
  cp "$SRC" "$KJSON"
  chmod 600 "$KJSON"
  echo "Kaggle credentials installed at $KJSON"
  exit 0
fi

if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  echo -n '{"username":"' > "$KJSON"
  echo -n "$KAGGLE_USERNAME" >> "$KJSON"
  echo -n '","key":"' >> "$KJSON"
  echo -n "$KAGGLE_KEY" >> "$KJSON"
  echo '"}' >> "$KJSON"
  chmod 600 "$KJSON"
  echo "Kaggle credentials created from environment at $KJSON"
  exit 0
fi

echo "No credentials provided. Set KAGGLE_USERNAME and KAGGLE_KEY or pass a kaggle.json path." >&2
exit 1

