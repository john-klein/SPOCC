# SPOCC
Code for SPOCC, a Scalable POssibilistic Combination of Classifiers

An ensemble method that relies on a possibility theory and confusion matrices  [arXiv](https://arxiv.org/pdf/1908.06475.pdf).

The repository contains several python files.

It can be readily downloaded and executed in a pyhton console (**python 3**) provided that the imported python module versions on your machine match the following ones. The default parameter values allows to obtain a quick comparison between SPOOC and reference methods (classifier selection, weighted vote aggregation, stacking, or centralized learning) on simple synthetic datasets. To achieve a prescribed confidence level in the returned accuracies the parameter `iter_max` must be set to `np.inf` but the execution will be significantly longer.

**Warning**: please use the following (or more recent) module versions: `numpy` 1.14.0, `matplotlib` 0.98.0, `sklearn` 0.18.1


Licence
=======
This software is distributed under the [CeCILL Free Software Licence Agreement](http://www.cecill.info/licences/Licence_CeCILL_V2-en.html)
