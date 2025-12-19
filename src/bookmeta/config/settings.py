from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESOURCES_ROOT = PROJECT_ROOT / "resources"
CACHE_ROOT = PROJECT_ROOT / ".cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

PIPELINE_CACHE_DIR = CACHE_ROOT / "pipeline"
PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

NORMALIZE_CACHE_DIR = CACHE_ROOT / "normalize_pipeline"
NORMALIZE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_PATH = RESOURCES_ROOT / "bookmeta.db"
