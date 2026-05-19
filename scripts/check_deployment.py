#!/usr/bin/env python3
"""Pre-deployment verification script for BirdNET Validator on HF Spaces."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_python_version() -> bool:
    """Verify Python 3.11+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    print(f"[FAIL] Python {version.major}.{version.minor} (require 3.11+)")
    return False


def check_files() -> bool:
    """Verify all required files exist."""
    required = [
        "app.py",
        "src/ui/app_factory.py",
        "src/config/runtime_config.py",
        "src/services/validation_service.py",
        "src/services/audio_fetch_service.py",
        "requirements.txt",
    ]
    
    all_exist = True
    for file in required:
        path = Path(file)
        if path.exists():
            print(f"[OK] {file}")
        else:
            print(f"[FAIL] {file} NOT FOUND")
            all_exist = False
    
    return all_exist


def check_imports() -> bool:
    """Verify all imports work."""
    try:
        print("Checking imports...")
        from src.config.runtime_config import RuntimeConfig
        from src.ui.app_factory import create_app
        from src.auth.auth_service import AuthService
        print("[OK] All imports work")
        return True
    except Exception as exc:
        print(f"[FAIL] Import failed: {exc}")
        return False


def check_config() -> bool:
    """Verify runtime config loads."""
    try:
        print("Checking runtime config...")
        from src.config.runtime_config import RuntimeConfig
        cfg = RuntimeConfig.from_env()
        print("[OK] Config loaded:")
        print(f"  - Page size: {cfg.page_size}")
        print(f"  - Demo bootstrap: {cfg.enable_demo_bootstrap}")
        print(f"  - Validation dir: {cfg.validation_base_dir}")
        return True
    except Exception as exc:
        print(f"[FAIL] Config error: {exc}")
        return False


def check_dependencies() -> bool:
    """Verify critical dependencies."""
    dependencies = ["gradio", "huggingface_hub", "pandas", "numpy"]
    all_ok = True
    
    for pkg in dependencies:
        try:
            __import__(pkg)
            print(f"[OK] {pkg}")
        except ImportError:
            print(f"[FAIL] {pkg} NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_app_creation() -> bool:
    """Attempt to create the app without starting server."""
    try:
        print("Checking app creation...")
        import os
        os.environ["BIRDNET_ENABLE_DEMO_BOOTSTRAP"] = "true"
        
        from src.ui.app_factory import create_app
        app = create_app()
        print(f"[OK] App created (type: {type(app).__name__})")
        return True
    except Exception as exc:
        print(f"[FAIL] App creation failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Run all checks."""
    print("=" * 60)
    print("BirdNET Validator - Pre-Deployment Check")
    print("=" * 60)
    print()
    
    checks = [
        ("Python version", check_python_version),
        ("Required files", check_files),
        ("Dependencies", check_dependencies),
        ("Runtime config", check_config),
        ("Code imports", check_imports),
        ("App creation", check_app_creation),
    ]
    
    results = []
    for name, check_fn in checks:
        print(f"\n{name}:")
        try:
            result = check_fn()
            results.append(result)
        except Exception as exc:
            print(f"[FAIL] {name} check failed: {exc}")
            results.append(False)
    
    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")
    print("=" * 60)
    
    if all(results):
        print("\n[OK] Ready for deployment!")
        return 0
    else:
        print("\n[FAIL] Fix issues above before deploying")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
