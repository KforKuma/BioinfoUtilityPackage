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
    """在超时时间内运行命令并捕获输出。

    Args:
        cmd: 命令参数列表。
        timeout: 超时时间，单位秒。

    Returns:
        ``(success, output)``。失败时 ``output`` 为错误信息。

    Example:
        >>> ok, output = _run_with_timeout(["python", "--version"], timeout=10)
        >>> ok
        True
    """
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
    """确保当前 Python 环境可导入指定包，缺失时尝试 pip 安装。

    该函数主要用于 notebook/交互式环境的轻量兜底。HPC 批量任务中建议优先使用
    已固定的环境，而不是在运行时安装依赖。

    Args:
        install_name: pip 安装名。
        import_name: import 使用的模块名；为 ``None`` 时用 ``install_name`` 推断。
        version: 可选版本号或版本约束，例如 ``"1.0.0"`` 或 ``">=1.0"``。
        timeout: pip 安装超时时间，单位秒。

    Returns:
        若最终可以导入则返回 ``True``，否则返回 ``False``。

    Example:
        >>> ensure_package("scikit-posthocs", import_name="scikit_posthocs")
        # 缺失时会尝试安装；安装失败则返回 False。
    """
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

    print(f"[ensure_package] Installing package: {pkg_spec}")

    success, output = _run_with_timeout(
        [sys.executable, "-m", "pip", "install", pkg_spec],
        timeout=timeout,
    )

    if not success:
        print(f"[ensure_package] Warning! Failed to install {pkg_spec}: {output}")
        return False

    try:
        importlib.import_module(import_name)
        print(f"[ensure_package] Successfully installed {pkg_spec}")
        return True
    except ImportError:
        print(f"[ensure_package] Warning! Package installed but cannot be imported: {import_name}")
        return False



def count_element_list_occurrence(list_of_lists: Iterable[Iterable]) -> dict:
    """统计元素出现在多少个子列表中。

    同一子列表内的重复元素只计一次，适合统计多个基因列表之间的出现频率。

    Args:
        list_of_lists: 多个可迭代对象组成的集合。

    Returns:
        ``{element: occurrence_count}`` 字典。

    Example:
        >>> count_element_list_occurrence([["A", "A", "B"], ["B", "C"]])
        {'A': 1, 'B': 2, 'C': 1}
    """
    from collections import defaultdict
    counter = defaultdict(int)
    for unique_list in list_of_lists:
        for item in set(unique_list):  # 用 set() 保证列表内唯一
            counter[item] += 1
    return dict(counter)


def sanitize_filename(filename: str) -> str:
    """替换文件名中不适合 Windows 路径的字符。

    Args:
        filename: 原始文件名。

    Returns:
        清理后的文件名。

    Example:
        >>> sanitize_filename("A/B:C?.txt")
        'A_B_C_.txt'
    """
    import re
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def call_with_compatible_args(func, *args, **kwargs):
    """仅用目标函数支持的关键字参数调用函数。

    Args:
        func: 目标函数。
        *args: 位置参数。
        **kwargs: 候选关键字参数。

    Returns:
        ``func(*args, **filtered_kwargs)`` 的返回值。

    Example:
        >>> def func(a, b=1):
        ...     return a + b
        >>> call_with_compatible_args(func, 2, b=3, unused=True)
        5
    """
    sig = inspect.signature(func)
    accepted_params = sig.parameters
    
    # 只保留 func 明确声明的关键字参数
    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k in accepted_params
    }
    
    # 传递 *args 和过滤后的 **kwargs 给目标函数
    return func(*args, **filtered_kwargs)
