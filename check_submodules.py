#!/usr/bin/env python3
"""
Submodule Status Checker

This script checks the status of external submodules and provides
helpful information for troubleshooting import issues.
"""

import sys
from pathlib import Path
import subprocess
import os

def check_submodule_status():
    """Check the status of all submodules."""
    project_root = Path(__file__).parent
    
    print("=" * 60)
    print("SUBMODULE STATUS CHECK")
    print("=" * 60)
    
    # Check external directory
    external_dir = project_root / "external"
    if not external_dir.exists():
        print("❌ External directory not found!")
        print(f"   Expected: {external_dir}")
        return False
    
    print(f"✓ External directory found: {external_dir}")
    
    # Check each submodule
    submodules = {
        'argument-mining-api': external_dir / "argument-mining-api",
        'argument-mining-db': external_dir / "argument-mining-db"
    }
    
    all_good = True
    
    for name, path in submodules.items():
        print(f"\n--- {name.upper()} ---")
        
        if not path.exists():
            print(f"❌ {name} not found at: {path}")
            all_good = False
            continue
        
        print(f"✓ Directory exists: {path}")
        
        # Check if it's a git repository
        git_dir = path / ".git"
        if git_dir.exists():
            print(f"✓ Git repository detected")
            
            # Get git status
            try:
                result = subprocess.run(
                    ['git', 'status', '--porcelain'],
                    cwd=path,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    if result.stdout.strip():
                        print(f"⚠️  Repository has uncommitted changes")
                    else:
                        print(f"✓ Repository is clean")
                else:
                    print(f"⚠️  Could not check git status")
            except Exception as e:
                print(f"⚠️  Error checking git status: {e}")
        else:
            print(f"❌ Not a git repository")
            all_good = False
        
        # Check for key files
        key_files = {
            'argument-mining-api': ['app/argmining/implementations', 'app/argmining/interfaces'],
            'argument-mining-db': ['db/queries.py', 'db/models.py']
        }
        
        if name in key_files:
            for file_path in key_files[name]:
                full_path = path / file_path
                if full_path.exists():
                    print(f"✓ Key file/directory found: {file_path}")
                else:
                    print(f"❌ Key file/directory missing: {file_path}")
                    all_good = False
    
    return all_good

def check_python_imports():
    """Test if the submodules can be imported in Python."""
    print("\n" + "=" * 60)
    print("PYTHON IMPORT TEST")
    print("=" * 60)
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Add external submodules to path
    external_api = project_root / "external" / "argument-mining-api"
    external_db = project_root / "external" / "argument-mining-db"
    
    if external_api.exists():
        sys.path.insert(0, str(external_api))
        print(f"✓ Added argument-mining-api to Python path")
    else:
        print(f"❌ argument-mining-api not found")
    
    if external_db.exists():
        sys.path.insert(0, str(external_db))
        print(f"✓ Added argument-mining-db to Python path")
    else:
        print(f"❌ argument-mining-db not found")
    
    # Test imports
    import_tests = [
        ('app.argmining.interfaces.adu_and_stance_classifier', 'AduAndStanceClassifier'),
        ('app.argmining.interfaces.claim_premise_linker', 'ClaimPremiseLinker'),
        ('app.argmining.implementations.openai_llm_classifier', 'OpenAILLMClassifier'),
        ('app.argmining.implementations.tinyllama_llm_classifier', 'TinyLLamaLLMClassifier'),
        ('db.queries', 'get_benchmark_data'),
    ]
    
    all_imports_good = True
    
    for module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                print(f"✓ {module_name}.{class_name}")
            else:
                print(f"⚠️  {module_name} imported but {class_name} not found")
                all_imports_good = False
        except ImportError as e:
            print(f"❌ Failed to import {module_name}.{class_name}: {e}")
            all_imports_good = False
        except Exception as e:
            print(f"❌ Error importing {module_name}.{class_name}: {e}")
            all_imports_good = False
    
    return all_imports_good

def provide_recommendations():
    """Provide recommendations based on the current state."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    external_dir = project_root / "external"
    
    if not external_dir.exists():
        print("1. Create external directory:")
        print(f"   mkdir {external_dir}")
        print()
        print("2. Initialize submodules:")
        print("   git submodule add https://github.com/Horizontal-Labs/argument-mining-api.git external/argument-mining-api")
        print("   git submodule add https://github.com/Horizontal-Labs/argument-mining-db.git external/argument-mining-db")
        return
    
    submodules = {
        'argument-mining-api': external_dir / "argument-mining-api",
        'argument-mining-db': external_dir / "argument-mining-db"
    }
    
    for name, path in submodules.items():
        if not path.exists():
            print(f"1. Initialize {name} submodule:")
            print(f"   git submodule add https://github.com/Horizontal-Labs/{name}.git external/{name}")
        else:
            git_dir = path / ".git"
            if not git_dir.exists():
                print(f"2. {name} exists but is not a git repository. Consider reinitializing:")
                print(f"   rm -rf external/{name}")
                print(f"   git submodule add https://github.com/Horizontal-Labs/{name}.git external/{name}")
    
    print("\n3. Update all submodules to latest versions:")
    print("   git submodule update --remote")
    print()
    print("4. Initialize submodules if not already done:")
    print("   git submodule update --init --recursive")

def main():
    """Main function to run all checks."""
    print("Checking submodule status...")
    
    submodules_ok = check_submodule_status()
    imports_ok = check_python_imports()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if submodules_ok and imports_ok:
        print("✅ All checks passed! Your submodules are properly configured.")
    else:
        print("❌ Some issues found. See recommendations below.")
        provide_recommendations()

if __name__ == "__main__":
    main() 