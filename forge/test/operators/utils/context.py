import random

from contextlib import contextmanager
from typing import Dict, Optional
from loguru import logger


__all__ = [
    "test_sweep_context",
    "get_test_sweep_context",
    "test_sweep_context_provider",
]


class TestSweepContext:

    def __init__(self):
        self.rngs: Dict[str, random.Random] = {}

    # @classmethod
    def get_rng(self, name: str):
        if name not in self.rngs:
            self.rngs[name] = random.Random(31)
        return self.rngs[name]


test_sweep_context: Optional[TestSweepContext] = None


def get_test_sweep_context():
    global test_sweep_context
    if test_sweep_context is None:
        raise ValueError("No context")
    return test_sweep_context


@contextmanager
def test_sweep_context_provider():
    logger.trace("Entering test sweep context")
    global test_sweep_context
    test_sweep_context = TestSweepContext()
    yield
    logger.trace("Exiting test sweep context")
    test_sweep_context = None
