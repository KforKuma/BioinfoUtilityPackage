import os
import pandas as pd


def AIRR_combine(input_file_list, output_dir):
    """
    Combine AIRR files from the same sample together.

    Parameters
    ----------
    input_file_list : dict
        A dictionary where the keys are the sample names and the values are lists of AIRR file paths.
    output_dir : str
        The directory where the combined AIRR files will be saved.

    Returns
    -------
    None

    Notes
    -----
    This function assumes that the AIRR files are in the same format and can be combined directly.
    If the files are in different formats, additional processing may be required before combining them.

    Examples
    --------
    AIRR_combine(input_dir="/data/HeLab/bio/IBD_plus/GSE116222/bam/TRUST4_output",
                 output_dir="/data/HeLab/bio/IBD_plus/GSE116222/bam/AIRR")
    """
    # Check that the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Combine each sample's files
    for sample, files in input_file_list.items():
        print(f"[AIRR_combine] Processing sample {sample}...")
        output_file = os.path.join(output_dir, f"{sample}_combined_AIRR.tsv")

        dfs = []
        for file in files:
            print(f"Reading file: {file}")
            df = pd.read_csv(file, sep="\t")
            dfs.append(df)

        merged_df = pd.concat(dfs, ignore_index=True)
        print("[AIRR_combine] All files combined!")

        merged_df.to_csv(output_file, sep="\t", index=False)
        print(f"Combined result saved to {output_file}")

