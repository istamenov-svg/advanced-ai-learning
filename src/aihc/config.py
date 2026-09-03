"""Paths and settings. One place, so nothing else hardcodes a path or a model name."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
CACHE_DIR = DATA_DIR / ".cache"
REPORTS_DIR = ROOT / "reports"
STATE_FILE = ROOT / "state.json"
RUN_LOG = ROOT / "runs.jsonl"

MODEL = os.getenv("AIHC_MODEL", "claude-sonnet-4-5")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
STUB = os.getenv("AIHC_STUB", "0") == "1" or not API_KEY

# Bump this whenever a prompt template changes. It is part of the cache key,
# so a prompt change invalidates cached answers the same way a data change does.
PROMPT_VERSION = "2026-09-03.1"

# Agent guardrails
MAX_TOOL_CALLS_PER_RUN = 12
MAX_OUTPUT_TOKENS = 1500
