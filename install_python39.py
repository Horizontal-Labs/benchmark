#!/usr/bin/env python3
"""
Python 3.9 Installation Script for Argument Mining Benchmark

This script sets up the environment for Python 3.9 users.
"""

import sys
import subprocess
import os
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
        print("This script is designed for Python 3.9. You may encounter compatibility issues.")
        return False

def install_requirements():
    """Install Python 3.9 compatible requirements."""
    print("\n" + "="*60)
    print("INSTALLING PYTHON 3.9 COMPATIBLE REQUIREMENTS")
    print("="*60)
    
    # Check if requirements_python39.txt exists
    requirements_file = Path("requirements_python39.txt")
    if not requirements_file.exists():
        print("❌ requirements_python39.txt not found")
        print("Falling back to main requirements.txt...")
        requirements_file = Path("requirements.txt")
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
    
    print(f"Installing from {requirements_file}...")
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        # Install requirements
        print("Installing requirements...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Requirements installed successfully")
            return True
        else:
            print("❌ Failed to install requirements:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def setup_environment():
    """Set up environment variables."""
    print("\n" + "="*60)
    print("SETTING UP ENVIRONMENT")
    print("="*60)
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        env_content = """# OpenAI API Key (note the underscore: OPEN_AI_KEY)
OPEN_AI_KEY=your-openai-api-key-here

# Hugging Face Token (optional)
HF_TOKEN=your-huggingface-token-here
"""
        env_file.write_text(env_content)
        print("✓ Created .env file")
        print("⚠️  Please edit .env file and add your actual API keys")
    else:
        print("✓ .env file already exists")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print("✓ Results directory ready")

def test_installation():
    """Test the installation."""
    print("\n" + "="*60)
    print("TESTING INSTALLATION")
    print("="*60)
    
    test_imports = [
        'pandas', 'numpy', 'sklearn', 'torch', 'transformers',
        'fastapi', 'pydantic', 'openai', 'sqlalchemy', 'dotenv'
    ]
    
    failed_imports = []
    
    for package in test_imports:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
        except Exception as e:
            print(f"⚠️  {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  {len(failed_imports)} packages failed to import:")
        for package in failed_imports:
            print(f"  - {package}")
        return False
    else:
        print("\n✓ All packages imported successfully")
        return True

def run_compatibility_test():
    """Run the compatibility test."""
    print("\n" + "="*60)
    print("RUNNING COMPATIBILITY TEST")
    print("="*60)
    
    test_script = Path("test_python39_compatibility.py")
    if test_script.exists():
        try:
            result = subprocess.run([
                sys.executable, str(test_script)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Compatibility test passed")
                return True
            else:
                print("❌ Compatibility test failed:")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"❌ Error running compatibility test: {e}")
            return False
    else:
        print("⚠️  Compatibility test script not found")
        return True

def main():
    """Main installation function."""
    print("PYTHON 3.9 INSTALLATION SCRIPT")
    print("="*60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Install requirements
    requirements_ok = install_requirements()
    
    # Setup environment
    setup_environment()
    
    # Test installation
    test_ok = test_installation()
    
    # Run compatibility test
    compatibility_ok = run_compatibility_test()
    
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    
    print(f"Python 3.9: {'✓' if python_ok else '❌'}")
    print(f"Requirements installed: {'✓' if requirements_ok else '❌'}")
    print(f"Import test: {'✓' if test_ok else '❌'}")
    print(f"Compatibility test: {'✓' if compatibility_ok else '❌'}")
    
    if python_ok and requirements_ok and test_ok and compatibility_ok:
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your API keys")
        print("2. Run: python test_benchmark_integration.py")
        print("3. Run: python app/benchmark.py")
        return True
    else:
        print("\n⚠️  Installation completed with issues.")
        print("Please review the errors above and fix any problems.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 