**Bioinfo-Utility-Package**

This is an integrated utility package primarily designed for AnnData / Scanpy workflows. It aims to assist with the following tasks:
1.	Performing routine statistical analyses, especially those that are somewhat tedious to implement from scratch; 
2.	Managing cell identity annotation and refinement — a process that can become particularly time-consuming when data quality is suboptimal; 
3.	Streamlining plotting workflows and figure export; 
4.	Facilitating seamless data exchange between internal environments and external tools (e.g., CellPhoneDB). 
________________________________________
**Key Features**

Handlers (src/core/handlers)
•	ScanpyPlotWrapper
A wrapper utility that automatically saves all generated plots. 
•	ObsEditor
During iterative clustering and re-annotation, cell identities often need to be repeatedly adjusted or transferred between objects.
This tool helps keep your code clean, structured, and more readable. 
•	Geneset
Designed to improve the usability and maintainability of gene signatures.
Hardcoding gene sets directly in scripts is neither elegant nor maintainable, yet ideas often need to be recorded quickly.
This utility enables standardized reading, updating, and storage of gene sets, avoiding scattered variables such as my_marker_dict. 
________________________________________
**Cell Abundance Analysis Tools (src/stats)**

•	Multiple simulation strategies for generating cell abundance data, along with corresponding evaluation functions 
•	Various differential abundance analysis methods, including an integrated meta-method (RCE) 
•	Associated visualization functions 

The phase-1 differential-abundance interface now separates each canonical result into two linked tables:

• `CanonicalContrastPublicView` contains the tested contrast, effect estimate and direction, the paradigm-agnostic `primary_decision`, and availability/validity state. Ordinary filtering is identical for frequentist and Bayesian methods.
• `CanonicalEvidenceLayer` retains method-native p-values, posterior/inclusion probabilities, uncertainty intervals, discovery metrics, and `native_decision`. These quantities are never numerically converted into one another or collapsed into a common confidence score.

Every finite effect records its estimand, scale, direction, and direction basis. In particular, a compositional coefficient relative to a reference cell type is labelled as a relative log effect and must not be interpreted as an absolute abundance difference. Propeller, DCATS, sccomp, and Pertpy/scCODA adapters are available under `src/stats/adapters`; all native outputs and diagnostics are retained by the runner.

Phase 2 provides one configuration-driven entry point for both simulation and real-data work:

```powershell
python -m src.stats.pipeline project/Step08_Abundance/configs/phase2_simulation.yml
python -m src.stats.pipeline project/Step08_Abundance/configs/phase2_real_data.yml
```

Simulation results are evaluated only through `src.stats.evaluation.evaluate_contrasts`, using an explicitly selected truth source and a common complete method universe. Real-data mode never emits truth-dependent Power, FPR, FDP, or FDR. Every run uses a distinct immutable `run_id`, writes a file-hash manifest, and refuses to overwrite an existing run directory.
________________________________________
**AnnData Toolbox (src/core/adata)**

Provides convenient, high-level interfaces for standard analysis workflows, including:
•	Differential expression (DEG) analysis 
•	Dimensionality reduction and clustering 
•	Integrated plotting utilities 
In addition, several commonly used functions in existing workflows (e.g., score_genes, downsampling) have been optimized and extended with additional functionality.
________________________________________
**External Adapters (src/external_adaptor)**

Provides interfaces for connecting AnnData objects to external analysis tools.
These workflows are adapted from existing implementations with practical modifications.
Currently supported tools include:
•	pySCENIC 
•	CellPhoneDB v5 
•	CellRank 
________________________________________
**Recent Updates**

This update focuses on:
•	The development of differential abundance testing methods 
•	A simulation framework based on cell tag units, rather than neighborhood or hypersphere-based structures 
For typical scenarios with reference cell types, we recommend trying the CLR-LMM approach.
Historical experiment summaries should be re-evaluated with the corrected, separately reported FPR and FDP definitions before comparing methods.
Users are encouraged to benchmark and compare different methods on their own (simulated) datasets.
________________________________________
**Installation**

This project is currently under active development.
For now, you can simply import it directly into your environment.
(Yes, it’s that simple—for now 🙂)
________________________________________

