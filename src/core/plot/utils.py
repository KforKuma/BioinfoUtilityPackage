import logging
import os
from typing import Optional, Sequence, Tuple

import matplotlib
import numpy as np

from src.utils.hier_logger import logged

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

logger = logging.getLogger(__name__)


def jitter_color(base_rgb: Sequence[float], scale: float = 0.1,
                 rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """对基础颜色做轻微随机扰动。

    该函数常用于在同一大类 cell subtype/subpopulation 下，为其子群生成
    相近但可区分的颜色，避免图例中颜色完全重复。

    Args:
        base_rgb: 长度为 3 的 RGB 元组或列表，取值范围通常为 0 到 1。
        scale: 每个通道允许扰动的幅度。
        rng: 可选的随机数生成器；若不提供则使用默认生成器。

    Returns:
        扰动后的 RGB 三元组。

    Example:
        base_color = (0.25, 0.50, 0.75)
        # 在同一主类群下，为不同子群生成相近颜色
        child_color = jitter_color(base_color, scale=0.08)
    """
    if len(base_rgb) != 3:
        raise ValueError("Argument `base_rgb` must contain exactly 3 numeric values.")
    if scale < 0:
        raise ValueError("Argument `scale` must be greater than or equal to 0.")

    rng = rng or np.random.default_rng()
    r, g, b = [float(x) for x in base_rgb]
    r = min(max(r + rng.uniform(-scale, scale), 0.0), 1.0)
    g = min(max(g + rng.uniform(-scale, scale), 0.0), 1.0)
    b = min(max(b + rng.uniform(-scale, scale), 0.0), 1.0)
    return (r, g, b)


@logged
def matplotlib_savefig(fig, abs_file_path: str, dpi: int = 150, close_after: bool = True) -> None:
    """安全保存 Matplotlib 图像。

    该函数会自动创建目录、识别扩展名，并在保存前清理图中的 NaN/Inf，
    以减少 PDF/PNG 导出时的异常。

    Args:
        fig: 需要保存的 Matplotlib figure 对象。
        abs_file_path: 输出路径。若未指定扩展名，则默认同时保存 `.png` 和 `.pdf`。
        dpi: 位图导出分辨率。
        close_after: 保存后是否关闭 figure。

    Example:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1])
        # 未指定扩展名时，会同时保存 PNG 和 PDF
        matplotlib_savefig(fig, os.path.join(save_addr, "demo_plot"))
    """
    if fig is None:
        raise ValueError("Argument `fig` must not be `None`.")
    if not isinstance(abs_file_path, str) or abs_file_path.strip() == "":
        raise ValueError("Argument `abs_file_path` must be a non-empty string.")

    abs_file_path = abs_file_path.strip()
    dirname = os.path.dirname(abs_file_path) or os.getcwd()
    os.makedirs(dirname, exist_ok=True)

    valid_exts = {".png", ".pdf", ".svg", ".eps", ".jpg", ".jpeg", ".tif", ".tiff"}
    filename = os.path.basename(abs_file_path)
    stem, ext = os.path.splitext(filename)
    ext = ext.lower()
    base = os.path.join(dirname, stem if ext else filename)

    if ext and ext not in valid_exts:
        raise ValueError(
            f"Unsupported file extension for `abs_file_path`: '{ext}'. "
            "Supported extensions are .png, .pdf, .svg, .eps, .jpg, .jpeg, .tif, and .tiff."
        )

    # 保存前统一清理图中可能导致矢量导出报错的异常数值。
    for coll in fig.findobj(matplotlib.collections.Collection):
        offsets = coll.get_offsets()
        if getattr(offsets, "size", 0) > 0:
            coll.set_offsets(np.nan_to_num(offsets, nan=0.0, posinf=0.0, neginf=0.0))

        if hasattr(coll, "get_facecolors"):
            facecolors = coll.get_facecolors()
            if getattr(facecolors, "size", 0) > 0:
                coll.set_facecolors(np.nan_to_num(facecolors, nan=0.0, posinf=0.0, neginf=0.0))

        if hasattr(coll, "get_edgecolors"):
            edgecolors = coll.get_edgecolors()
            if getattr(edgecolors, "size", 0) > 0:
                coll.set_edgecolors(np.nan_to_num(edgecolors, nan=0.0, posinf=0.0, neginf=0.0))

    if ext == "":
        fig.savefig(base + ".png", bbox_inches="tight", dpi=dpi)
        fig.savefig(base + ".pdf", bbox_inches="tight", format="pdf", dpi=dpi)
        logger.info(f"[matplotlib_savefig] Figure was saved to: '{base}.png' and '{base}.pdf'.")
    else:
        fig.savefig(base + ext, bbox_inches="tight", dpi=dpi)
        logger.info(f"[matplotlib_savefig] Figure was saved to: '{base + ext}'.")

    if close_after:
        import matplotlib.pyplot as plt
        plt.close(fig)
