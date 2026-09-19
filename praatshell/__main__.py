"""Entry point: an interactive shell, or a single command and out."""

import os
import sys

from .shell import run

if __name__ == "__main__":
    sys.exit(run(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), sys.argv[1:]))
