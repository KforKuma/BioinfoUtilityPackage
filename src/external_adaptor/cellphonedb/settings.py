import matplotlib.pyplot as plt

from itertools import cycle




CPDB_version = "v5.0.1"
ktplotspy_version = "v0.3.2"

DEFAULT_SEP = ">@<"
DEFAULT_SPEC_PAT = "/|:|\\?|\\*|\\+|\\(|\\)|\\/|\\[|\\]"
DEFAULT_CELLSIGN_ALPHA = 0.5
DEFAULT_COLUMNS = ["interaction_group", "celltype_group"]

INTERACTION_COLUMNS = ['interacting_pair', 'partner_a', 'partner_b', 'gene_a', 'gene_b', 'directionality', 'classification']

DEFAULT_V5_COL_START = 13
DEFAULT_V5_COL_NAMES = ["id_cp_interaction","interacting_pair","partner_a","partner_b",
                        "gene_a","gene_b","secreted","receptor_a","receptor_b","annotation_strategy",
                        "is_integrin","directionality","classification"]

DEFAULT_COL_START = 11
DEFAULT_CLASS_COL = 12
DEFAULT_CPDB_SEP = "|"
DEFAULT_PAL = plt.cm.tab20.colors
DEFAULT_PAL_CYCLER = cycle(plt.cm.tab20.colors)