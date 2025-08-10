# app/argmining/implementations/__init__.py
import sys
import os
from pathlib import Path

# Add external repositories to Python path
project_root = Path(__file__).parent.parent.parent.parent
external_api = project_root / "external" / "argument-mining-api"
external_db = project_root / "external" / "argument-mining-db"

if external_api.exists():
    sys.path.insert(0, str(external_api))
if external_db.exists():
    sys.path.insert(0, str(external_db))

# Import specific modules you need
try:
    from app.api.services import ArgumentMiningService  # from argument-mining-api
    from db.models import ADU, Argument  # from argument-mining-db
except ImportError as e:
    print(f"Warning: Could not import from external repositories: {e}")
