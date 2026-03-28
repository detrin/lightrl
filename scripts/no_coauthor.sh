#!/usr/bin/env bash
if grep -qi "co-authored-by" "$1"; then
    echo "ERROR: Co-Authored-By not allowed in commit messages"
    exit 1
fi
