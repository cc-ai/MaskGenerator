import os
from pathlib import Path
import argparse
import torch
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from source.utils import load_opts

root = Path(__file__).parent.parent.resolve()


opts = load_opts(default=root / "shared/defaults.yml")
