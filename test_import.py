#!/usr/bin/env python3
import sys
from pathlib import Path

# Add external submodules to Python path FIRST
external_api = Path(__file__).parent / "external" / "api"
external_db = Path(__file__).parent / "external" / "db"

print(f"External API path: {external_api}")
print(f"External API exists: {external_api.exists()}")

if external_api.exists():
    sys.path.insert(0, str(external_api))
    print(f"✓ Added argument-mining-api to Python path: {external_api}")
else:
    print(f"⚠️  argument-mining-api not found at: {external_api}")

# Try to import the module
try:
    import app.argmining.interfaces.adu_and_stance_classifier
    print("✓ Successfully imported adu_and_stance_classifier")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

# Check what's in sys.path
print("\nPython path:")
for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
    print(f"  {i}: {path}")
