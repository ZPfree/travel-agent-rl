"""Root conftest: add src/ to sys.path so 'import tools' etc. work."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
