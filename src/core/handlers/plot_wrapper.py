import logging
import os
from functools import wraps

import matplotlib
import matplotlib.pyplot as plt

from src.core.plot.utils import matplotlib_savefig
from src.utils.env_utils import call_with_compatible_args
from src.utils.hier_logger import logged

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@logged
def split_filename_with_ext(filename: str, allowed_exts: tuple[str, ...] = (".jpg", ".png", ".pdf")) -> tuple[str, str]:
    """智能拆分文件名与扩展名。

    仅当扩展名位于允许列表中时才进行拆分；否则返回原始文件名和空扩展名。

    Args:
        filename: 输入文件名。
        allowed_exts: 允许识别的扩展名元组。

    Returns:
        一个二元组 `(root, ext)`。若未识别到合法扩展名，则 `ext` 为空字符串。

    Example:
        root, ext = split_filename_with_ext("umap_plot.pdf")
    """
    if not isinstance(filename, str):
        raise TypeError("Argument `filename` must be a string.")

    filename = filename.strip()
    for ext in allowed_exts:
        if filename.lower().endswith(ext):
            return filename[:-len(ext)], ext
    return filename, ""


class ScanpyPlotWrapper(object):
    """对 `scanpy.pl` 系列绘图函数进行统一包装与保存。

    该包装器会在调用绘图函数后尝试提取 `Figure` 对象，并根据传入的
    `save_addr` 与 `filename` 自动保存图像。它适用于返回 `Figure`、
    `Axes`、Scanpy plot 对象或字典结果的多种场景。

    Args:
        func: 被包装的绘图函数，通常来自 `scanpy.pl`。

    Example:
        umap_plot = ScanpyPlotWrapper(sc.pl.umap)
        umap_plot(save_addr=save_addr, filename="umap_plot.pdf", adata=adata, color="Celltype")
    """

    def __init__(self, func):
        wraps(func)(self)
        self.func = func
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, save_addr: str, filename: str, *args, **kwargs):
        """执行绘图并保存结果。

        Args:
            save_addr: 输出目录。
            filename: 输出文件名，可包含扩展名。
            *args: 传递给绘图函数的位置参数。
            **kwargs: 传递给绘图函数的关键字参数。

        Returns:
            原始绘图函数的返回值。
        """
        if not isinstance(save_addr, str) or save_addr.strip() == "":
            raise ValueError("Argument `save_addr` must be a non-empty string.")
        if not isinstance(filename, str) or filename.strip() == "":
            raise ValueError("Argument `filename` must be a non-empty string.")

        os.makedirs(save_addr, exist_ok=True)
        self.logger.info(f"[{self.func.__name__}] Start plotting with output file: '{filename}'.")

        # 强制要求绘图函数返回对象，并在保存前关闭即时显示。
        kwargs.setdefault("return_fig", True)
        kwargs.setdefault("show", False)

        with plt.rc_context():
            result = call_with_compatible_args(self.func, *args, **kwargs)
            fig = self._extract_figure(result)

            abs_file_path = os.path.join(save_addr, filename)
            matplotlib_savefig(fig, abs_file_path, close_after=True)
            self.logger.info(f"[{self.func.__name__}] Plot saved to: '{abs_file_path}'.")

        plt.close("all")
        return result

    def _extract_figure(self, result):
        """从不同类型的绘图返回值中提取 `Figure` 对象。

        Args:
            result: 绘图函数返回值。

        Returns:
            提取到的 `matplotlib.figure.Figure` 对象。
        """
        fig = None

        if result is None:
            self.logger.warning(
                f"[{self.func.__name__}] Warning! Plot function returned `None`; fallback to `plt.gcf()`."
            )
            fig = plt.gcf()
        elif hasattr(result, "make_figure"):
            # 部分 Scanpy 绘图对象在首次返回时尚未真正构造 figure。
            if not hasattr(result, "fig") or result.fig is None:
                self.logger.info(
                    f"[{self.func.__name__}] Figure is not initialized yet; call `make_figure()` as fallback."
                )
                result.make_figure()
            fig = result.fig
        elif isinstance(result, plt.Figure):
            fig = result
        elif hasattr(result, "get_figure"):
            fig = result.get_figure()
        elif isinstance(result, dict):
            for value in result.values():
                target = value[0] if isinstance(value, (list, tuple)) and len(value) > 0 else value
                if hasattr(target, "get_figure"):
                    fig = target.get_figure()
                    break

        if fig is None:
            self.logger.warning(
                f"[{self.func.__name__}] Warning! Could not extract a figure from the return value; "
                "fallback to `plt.gcf()`."
            )
            fig = plt.gcf()

        if fig is None:
            raise RuntimeError("Failed to extract a valid matplotlib figure for saving.")
        return fig
