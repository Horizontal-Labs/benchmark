#!/usr/bin/env python3
"""
Python 3.9 Compatibility Test Script

This script tests if the requirements are compatible with Python 3.9
and provides guidance on any compatibility issues.
"""

import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if we're running Python 3.9."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 9:
        print("✓ Running Python 3.9")
        return True
    else:
        print(f"⚠️  Not running Python 3.9 (current: {version.major}.{version.minor})")
        return False

def check_requirements_compatibility():
    """Check if requirements can be installed with Python 3.9."""
    print("\n" + "="*60)
    print("CHECKING REQUIREMENTS COMPATIBILITY")
    print("="*60)
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("Testing requirements installation...")
    
    try:
        # Try to install requirements in a dry-run mode
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--dry-run", "-r", "requirements.txt"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ All requirements are compatible with Python 3.9")
            return True
        else:
            print("❌ Some requirements are not compatible:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Timeout checking requirements")
        return False
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        return False

def check_installed_packages():
    """Check currently installed packages and their versions."""
    print("\n" + "="*60)
    print("CURRENTLY INSTALLED PACKAGES")
    print("="*60)
    
    try:
        installed_packages = [d for d in pkg_resources.working_set]
        installed_packages.sort(key=lambda x: x.project_name.lower())
        
        print(f"Found {len(installed_packages)} installed packages:")
        
        # Check key packages
        key_packages = [
            'pandas', 'numpy', 'scikit-learn', 'torch', 'transformers',
            'fastapi', 'pydantic', 'openai', 'sqlalchemy'
        ]
        
        for package_name in key_packages:
            try:
                version = pkg_resources.get_distribution(package_name).version
                print(f"  {package_name}: {version}")
            except pkg_resources.DistributionNotFound:
                print(f"  {package_name}: Not installed")
                
    except Exception as e:
        print(f"❌ Error checking installed packages: {e}")

def check_import_compatibility():
    """Test importing key packages to check compatibility."""
    print("\n" + "="*60)
    print("IMPORT COMPATIBILITY TEST")
    print("="*60)
    
    test_imports = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'sklearn'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('fastapi', 'fastapi'),
        ('pydantic', 'pydantic'),
        ('openai', 'openai'),
        ('sqlalchemy', 'sqlalchemy'),
        ('dotenv', 'dotenv'),
    ]
    
    failed_imports = []
    
    for package_name, import_name in test_imports:
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError as e:
            print(f"❌ {package_name}: {e}")
            failed_imports.append(package_name)
        except Exception as e:
            print(f"⚠️  {package_name}: {e}")
            failed_imports.append(package_name)
    
    return len(failed_imports) == 0

def provide_compatibility_guidance():
    """Provide guidance on Python 3.9 compatibility issues."""
    print("\n" + "="*60)
    print("PYTHON 3.9 COMPATIBILITY GUIDANCE")
    print("="*60)
    
    print("""
Common Python 3.9 compatibility issues and solutions:

1. PANDAS < 2.0.0:
   - Use pandas 1.5.x for Python 3.9 compatibility
   - Some newer pandas features may not be available

2. NUMPY < 2.0.0:
   - Use numpy 1.21.x - 1.24.x for Python 3.9
   - Avoid numpy 2.x which requires Python 3.10+

3. PYDANTIC < 2.0.0:
   - Use pydantic 1.x for Python 3.9
   - Pydantic 2.x requires Python 3.10+
   - Update code to use pydantic v1 syntax

4. TORCH < 2.0.0:
   - Use PyTorch 1.13.x for Python 3.9
   - Some newer PyTorch features may not be available

5. TRANSFORMERS < 4.30.0:
   - Use transformers 4.21.x - 4.29.x for Python 3.9
   - Newer versions may require Python 3.10+

6. FASTAPI < 0.100.0:
   - Use FastAPI 0.95.x for Python 3.9
   - Newer versions may have compatibility issues

7. OPENAI < 1.0.0:
   - Use openai 0.28.x for Python 3.9
   - Newer versions may require Python 3.10+

If you encounter specific import errors, check:
- Package version compatibility
- Missing dependencies
- Python version requirements
""")

def main():
    """Run all compatibility tests."""
    print("PYTHON 3.9 COMPATIBILITY TEST")
    print("="*60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check requirements
    requirements_ok = check_requirements_compatibility()
    
    # Check installed packages
    check_installed_packages()
    
    # Check imports
    imports_ok = check_import_compatibility()
    
    # Provide guidance
    provide_compatibility_guidance()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"Python 3.9: {'✓' if python_ok else '❌'}")
    print(f"Requirements compatible: {'✓' if requirements_ok else '❌'}")
    print(f"Imports working: {'✓' if imports_ok else '❌'}")
    
    if python_ok and requirements_ok and imports_ok:
        print("\n🎉 All compatibility tests passed!")
        print("Your environment is ready for Python 3.9 development.")
    else:
        print("\n⚠️  Some compatibility issues found.")
        print("Please review the guidance above and fix any issues.")
    
    return python_ok and requirements_ok and imports_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 