# SPOCC
Code for SPOCC, a Scalable POssibilistic Combination of Classifiers

An ensemble method that relies on a possibility theory and confusion matrices [arXiv](https://arxiv.org/pdf/1908.06475.pdf).

The repository contains four python files.

It can be readily downloaded and executed in a pyhton console (**python 3**) provided that the imported python module versions on your machine match the following ones. 

To have a quick look on the performances of SPOOC run the `example()` function in `spocc_main.py` The default parameter values allows to obtain a quick comparison between SPOOC and reference methods (classifier selection, weighted vote aggregation, stacking, exponential weights, naive Bayes and Bayes aggregation or centralized learning) on simple synthetic datasets. To achieve a prescribed confidence level in the returned accuracies the parameter `iter_max` must be set to `np.inf` but the execution will be significantly longer.

**Warning**: the code is compatible with these module versions: `numpy` 1.17.2, `matplotlib` 3.1.1, `sklearn` 0.22.1, `scipy` 1.4.1


Licence
=======
This software is distributed under the [CeCILL Free Software Licence Agreement](http://www.cecill.info/licences/Licence_CeCILL_V2-en.html)
