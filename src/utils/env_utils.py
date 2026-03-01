from __future__ import annotations
import importlib
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
import inspect


def _run_with_timeout(
        cmd: list[str],
        timeout: int = 300
) -> tuple[bool, str]:
    """Run command with timeout and capture output."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        
        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.kill()
                return False, f"Command timed out after {timeout} seconds"
            time.sleep(0.1)
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.strip() or f"Command failed with return code {process.returncode}"
            return False, error_msg
        
        return True, stdout.strip()
    except Exception as e:
        return False, str(e)


def ensure_package(
    install_name: str,
    import_name: str | None = None,
    version: str | None = None,
    timeout: int = 300,
) -> bool:
    import_name = import_name or install_name.replace("-", "_")

    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        pass

    if version:
        if version[0].isdigit():
            version = f"=={version}"
        pkg_spec = f"{install_name}{version}"
    else:
        pkg_spec = install_name

    print(f"⚙️  Installing package: {pkg_spec}")

    success, output = _run_with_timeout(
        [sys.executable, "-m", "pip", "install", pkg_spec],
        timeout=timeout,
    )

    if not success:
        print(f"❌ Failed to install {pkg_spec}: {output}")
        return False

    try:
        importlib.import_module(import_name)
        print(f"✅ Successfully installed {pkg_spec}")
        return True
    except ImportError:
        print(f"❌ Package installed but cannot be imported: {import_name}")
        return False



def count_element_list_occurrence(list_of_lists: Iterable[Iterable]) -> dict:
    '''
    跨列表元素统计，常用于多个列表之间的基因出现的频率统计
    :param list_of_lists: 多个列表的列表
    :return: 返回计数字典
    '''
    from collections import defaultdict
    counter = defaultdict(int)
    for unique_list in list_of_lists:
        for item in set(unique_list):  # 用 set() 保证列表内唯一
            counter[item] += 1
    return dict(counter)


def sanitize_filename(filename: str) -> str:
    import re
    return re.sub(r'[<>:"/\\|?*]', '_', filename)




def call_with_compatible_args(func, **kwargs):
    sig = inspect.signature(func)
    accepted_params = sig.parameters
    
    # 只保留 func 明确声明的参数
    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k in accepted_params
    }
    
    return func(**filtered_kwargs)
