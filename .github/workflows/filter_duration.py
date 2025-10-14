import argparse
import json
from loguru import logger
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

COLLECTED_TESTS_FILE = "model_analysis_ooo.txt"
DURATIONS_FILE = ".test_durations" 

def load_json_file(path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return the parsed object.
    Raises exception on error.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """
    Atomically write JSON `data` to `path` using a temp file + os.replace.
    """
    ddir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(ddir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=ddir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception as exc:
                logger.debug(f"Failed to remove temp file {tmp}: {exc}")

def load_existing_durations(path: str) -> Dict[str, Any]:
    """
    Load existging durations from `path` or return empty dict if missing or corrupted.
    """
    if not os.path.exists(path):
        logger.debug(f"Durations file {path} does not exist, returning empty dict")
        return {}
    try:
        return load_json_file(path)
    except Exception as exc:
        logger.warning(f"Failed to parse existing durations file {path}: {exc} — treating as empty")
        return {}

def filter_test_durations(collected_tests_file, durations_file):
    """
    Filters a test duration dictionary, keeping only tests present in a list of collected tests.
    """
    
    existing_test_durations = load_existing_durations(durations_file)

    # 1. Read the list of collected tests into a set for fast lookup
    collected_tests = set()
    try:
        with open(collected_tests_file, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                collected_tests.add(stripped_line)

    except FileNotFoundError:
        print(f"Error: Collected tests file not found: {collected_tests_file}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error reading collected tests file: {e}", file=sys.stderr)
        return

    # 3. Filter the durations
    filtered_durations = {}
    for test_name, duration in existing_test_durations.items():
        if test_name not in collected_tests:
            filtered_durations[test_name] = duration


    atomic_write_json(durations_file, filtered_durations)

if __name__ == "__main__":
    filter_test_durations(COLLECTED_TESTS_FILE, DURATIONS_FILE)