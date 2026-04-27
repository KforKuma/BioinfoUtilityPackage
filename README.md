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
However, in our experiments, the RCE method achieved the best balance between statistical power and false positive rate (FPR).
Users are encouraged to benchmark and compare different methods on their own (simulated) datasets.
________________________________________
**Installation**

This project is currently under active development.
For now, you can simply import it directly into your environment.
(Yes, it’s that simple—for now 🙂)
________________________________________

