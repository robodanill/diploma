#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-data/plantclef2015/raw}"
BASE_URL="https://lab.plantnet.org/LifeCLEF/PlantCLEF2015"

mkdir -p "$ROOT_DIR"

wget -c --tries=20 --timeout=120 --read-timeout=120 \
  "$BASE_URL/TrainingPackage/PlantCLEF2015TrainingData.tar.gz" \
  -O "$ROOT_DIR/PlantCLEF2015TrainingData.tar.gz"

# Optional evaluation packages. They are large, so keep them explicit.
# wget -c --tries=20 --timeout=120 --read-timeout=120 \
#   "$BASE_URL/TestPackage/PlantCLEF2015TestDataWithAnnotations.tar.gz" \
#   -O "$ROOT_DIR/PlantCLEF2015TestDataWithAnnotations.tar.gz"
#
# wget -c --tries=20 --timeout=120 --read-timeout=120 \
#   "$BASE_URL/TestPackage/PlantCLEF2015DevToolkit.tar.gz" \
#   -O "$ROOT_DIR/PlantCLEF2015DevToolkit.tar.gz"
