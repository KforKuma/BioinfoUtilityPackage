def ensure_package(pkg_name: str):
    """确保包已安装，否则自动安装"""
    import importlib, subprocess, sys
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        print(f"⚙️  Installing missing package: {pkg_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])


