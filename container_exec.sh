#!/usr/bin/env bash
set -eux
docker run --rm -v "$PWD":"$PWD" -w "$PWD" rocm/primus:v26.2 "$@"
