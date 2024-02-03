#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

full_path=$SCRIPT_DIR/../data/duplicate_candidates/full/
block_path=$SCRIPT_DIR/../data/duplicate_candidates/blocked/


python src/measure_performance.py --full_path $full_path --block_path $block_path
