from pathlib import Path

module_path = Path(__file__).resolve().parent.parent
DATADIR = module_path / "data"

INPUT_DATADIR = DATADIR / "diagrams"
OUTPUT_DATADIR = DATADIR / "ransac_output"

max_dimension = 1000
