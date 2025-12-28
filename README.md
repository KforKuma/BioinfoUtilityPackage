**Bioinfo-Utility-Package**

An integrated package mainly devoted for anndata/scanpy workflow, helps you:

(1) perform statistics calculation that you usually need, but could be a little tricky;

(2) manipulate cellular identifications, this could takes lots of time especially when the quality of data is not satisfied enough;

(3) deal with your plot drawing and saving;

(4) plug your data in and out of some external tools, like CPDB.


**Features**

- *ScanpyPlotWrapper* class: a wrapper helps u save every plot


- *ObsEditor* class:

  sometimes u need to adjust cell identities repetitively, copy idents from one object to the other.

  This helps ur code get readable.

  
- *Geneset* class: for better reusing gene signatures.

  Writing down gene signatures in the code is an eyesore, but sometimes u just pop up an idea.

  This helps u read, update, and save a good format of gene signatures. No more `my_marker_dict` everywhere.

- Anndata toolbox

  Easiest ways for invoking some most standardised protocol, like finding DEG, or dim-reduction plus clustering.

  Some better adjust for existing code, like `score_genes` or downsample. And more.

- Adaptors for your AnnData to external function, currently including pyscenic, cellphonedb v5, and xgboost.


 **Recent Updates**
 
 Major update in developing differential abundance testing method and data simulation tools with unit of cell tag (rather than neighbors/hypersphere structure).

 CLR-LMM is recomended, but also welcome to test different methods on your (simulative) data.

**Installation Instructions**

Currently under construction. So just go import lol.

<pre>```
   (\_ _/)
   ( •_• )  Gimme a star
  / >🍪   / and I'll give u a cookie
```
</pre>
