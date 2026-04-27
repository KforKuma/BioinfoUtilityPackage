"""TRUST4 相关辅助函数。

Notes:
    1. 当前目录处于半废弃状态，因此本模块以保守维护为主。
    2. 仅做输入检查、输出规范统一和低风险兜底，不主动引入激进重构。
"""

import logging
import os

import pandas as pd

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


@logged
def AIRR_combine(input_file_list, output_dir):
    """合并同一样本的多个 AIRR TSV 文件。

    Args:
        input_file_list: 字典，键为样本名，值为该样本对应的 AIRR 文件路径列表。
        output_dir: 合并结果输出目录。

    Returns:
        `None`。

    Example:
        AIRR_combine(
            input_file_list={
                "SampleA": ["A_chain1.tsv", "A_chain2.tsv"],
                "SampleB": ["B_chain1.tsv"],
            },
            output_dir="./AIRR_combined",
        )
    """
    if not isinstance(input_file_list, dict) or not input_file_list:
        raise ValueError("Argument `input_file_list` must be a non-empty dictionary.")
    if not isinstance(output_dir, str) or output_dir.strip() == "":
        raise ValueError("Argument `output_dir` must be a non-empty string.")

    output_dir = output_dir.strip()
    os.makedirs(output_dir, exist_ok=True)

    for sample, files in input_file_list.items():
        if not files:
            print(f"[AIRR_combine] Warning! Sample '{sample}' had no input files and will be skipped.")
            continue

        print(f"[AIRR_combine] Processing sample: '{sample}'.")
        output_file = os.path.join(output_dir, f"{sample}_combined_AIRR.tsv")
        dfs = []
        reference_columns = None

        for file_path in files:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Input AIRR file was not found: '{file_path}'.")
            print(f"[AIRR_combine] Reading file: '{file_path}'.")
            df = pd.read_csv(file_path, sep="\t")
            if reference_columns is None:
                reference_columns = df.columns.tolist()
            elif df.columns.tolist() != reference_columns:
                print(
                    f"[AIRR_combine] Warning! Column order in file '{file_path}' did not match the first file. "
                    "The file will still be concatenated by column names."
                )
            dfs.append(df)

        if not dfs:
            print(f"[AIRR_combine] Warning! Sample '{sample}' produced no readable dataframes and will be skipped.")
            continue

        merged_df = pd.concat(dfs, ignore_index=True, sort=False)
        merged_df.to_csv(output_file, sep="\t", index=False)
        print(f"[AIRR_combine] Combined result was saved to: '{output_file}'.")
