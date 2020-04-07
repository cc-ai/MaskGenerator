from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from utils import load_opts

root = Path(__file__).parent.parent.resolve()


opts = load_opts(
    path=root / "shared/featureDA.yml", default=root / "shared/defaults.yml"
)
