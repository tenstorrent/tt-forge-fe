from .detector import XLA_MODE

if XLA_MODE:
    from .xla import XlaFrontend as SweepsFrontend
else:
    from .forge import ForgeFrontend as SweepsFrontend

__ALL__ = [
    "XLA_MODE",
    "SweepsFrontend",
]
