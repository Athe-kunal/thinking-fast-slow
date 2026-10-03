#!/bin/bash
mkdir -p /logs/verifier
if [ "$(python3 /app/sum.py 2>/dev/null | tr -d "[:space:]")" = "5050" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
