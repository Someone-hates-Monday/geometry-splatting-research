#!/usr/bin/env python3
"""Dump OptimizationParams defaults for rebuild provenance."""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from arguments import OptimizationParams

ap = ArgumentParser()
opt = OptimizationParams(ap)
ns = ap.parse_args([])
# Extract fields from group defaults
out = {k: getattr(ns, k) for k in vars(ns)}
Path("output_compare/ggs_trust_rebuild_20261116/scan24/logs").mkdir(parents=True, exist_ok=True)
Path("output_compare/ggs_trust_rebuild_20261116/scan24/logs/optimizationparams_defaults.json").write_text(
    json.dumps(out, indent=2, default=str), encoding="utf-8"
)
print("wrote optimizationparams_defaults.json")
