#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:12:34 2019

@author: johnklein
"""

import numpy as np
from numpy import matlib
import matplotlib
import copy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster

def smax(x,r=1):
    return np.exp(r*x)/np.sum(np.exp(r*x))

def getAlphas(preds,y,r=10):
    (Ns,n) = preds.shape
    correct_preds = np.equal(preds , matlib.repmat(y,Ns,1))
    err_clf = 1-np.sum(correct_preds,axis=1)/n
    alpha = np.zeros(err_clf.shape)
    if (np.max(err_clf)> np.min(err_clf)):
        alpha = smax(err_clf,r)
        norm_ct = np.where(err_clf == np.max(err_clf))[0].size
        alpha *= norm_ct
    alpha -= np.min(alpha)
    return alpha
    
    
    
def depend_level(preds,n_class,rho=1):
    epsi = 1e-6
    (Ns,n) = preds.shape
    (clf_mle,clf_counts) = clf_prob(preds,n_class,laplace_smooth=False)
    clf_mle[np.where(clf_mle==0)] = epsi
    (clf_pair_mle,clf_pair_counts) = clf_pair_prob(preds,n_class,laplace_smooth=False)
    clf_pair_mle[np.where(clf_pair_mle==0)] = epsi
    D = np.zeros((Ns,Ns))
    log_ratio = np.zeros((Ns,Ns))
    for i in range(Ns):
        for j in range(Ns):
            num = np.dot(clf_counts[i],np.log(clf_mle[i])) + np.dot(clf_counts[j],np.log(clf_mle[j]))
            den = np.dot(clf_pair_counts[i,j].flatten(),np.log(clf_pair_mle[i,j]).flatten())
            log_ratio[i,j] = num - den
            #log_ratio[i,j] = np.sum(np.log(clf_mle[i,preds[i,:]])) + np.sum(np.log(clf_mle[j,preds[j,:]])) - np.sum(np.log(clf_pair_mle[i,j,preds[i,:],preds[j,:]]))
            D[i,j] = 1 - np.exp( rho / n * log_ratio[i,j])
    return D, log_ratio

from scipy.cluster.hierarchy import dendrogram, linkage

def HAC_aggregation_fit(preds,agg,n_class,y,rho=1,display=False):
    """
    This function aggregates sequentially classifier predictions. The sequence
    is defined in terms of dependecy level between between base classifiers. 
    This level is used as part of hierarchical agglomerative clustering to 
    obtain the combination sequence. Pairwise combinations are performed using
    a (pseudo) tnorm between possibility distributions spanned by the two 
    operands following the principle depicted as part of adaSPOCC.
    """
    if (agg.name not in ['adaspocc']):
        raise ValueError('Invalid type of aggregator. Must be a spocc instance.')
    (D,log_ratio) = depend_level(preds,n_class,rho=1)
    agg.params["log_ratio"]=log_ratio
    (Ns,n) = preds.shape
    Z = linkage(1-D[np.triu_indices(Ns,1)],method='ward')
    agg.params["dendro"]=Z
    if display:
        plt.figure(figsize=(10, 5))
        dendrogram(Z)
  

def HAC_aggregation_predict(preds,agg,n_class,rho=1):
    """
    This function aggregates sequentially classifier predictions. The sequence
    is defined in terms of dependecy level between between base classifiers. 
    This level is used as part of hierarchical agglomerative clustering to 
    obtain the combination sequence. Pairwise combinations are performed using
    a (pseudo) tnorm between possibility distributions spanned by the two 
    operands following the principle depicted as part of SPOCC.
    """
    if (agg.name not in ['adaspocc']):
        raise ValueError('Invalid type of aggregator. Must be a spocc instance.')

    (Ns,n) = preds.shape
    (n_combi,_) = agg.params["dendro"].shape
    poss = np.zeros((n_combi+Ns,n,n_class))
    for i in range(Ns):
        for j in range(n):
            poss[i,j] = agg.params["cond_possib"][i,preds[i,j],:]
    for i in range(n_combi):
        for j in range(n):
            operands = agg.params["dendro"][i,:2].astype(int)
            if (isinstance(agg.params["hyper"],np.ndarray)):
                poss[i+Ns,j,:] = tnorm_op(poss[operands,j],agg.params["hyper"][i],agg.params["tnorm"])
            else:
                poss[i+Ns,j,:] = tnorm_op(poss[operands,j],agg.params["hyper"],agg.params["tnorm"])
    loc_preds = np.argmax(poss[i+Ns],axis=1)
    return loc_preds
    
        
def rGridSearch(agg,clf_preds,y, display=False):
    r_range = np.logspace(-1,2,201)
    r_success = np.zeros((r_range.size,))
    (Ns,n) = clf_preds.shape
    for kk in range(r_range.size):
        alpha = getAlphas(clf_preds,y,r=r_range[kk])
        for i in range(Ns):
            agg.params["cond_possib"][i] = (1-alpha[i])*agg.params["cond_possib_saved"][i] + alpha[i]
        preds_agg = agg.predict(clf_preds) 
        r_success[kk] = np.sum(preds_agg==y)/n
    agg.params["r"]=np.median(r_range[np.where(r_success == np.max(r_success))])
    #need to reset alphas and updated conditional possibilities
    agg.params["alpha"] = getAlphas(clf_preds,y,r=agg.params["r"])
    for i in range(Ns):
        agg.params["cond_possib"][i] = (1-agg.params["alpha"][i])*agg.params["cond_possib_saved"][i] + agg.params["alpha"][i]
    if (display == True):        
        plt.figure()
        plt.plot(r_range,r_success)
        plt.show()   
        
        
        
def rhoGridSearch(agg,clf_preds,y, display=False):
    rho_success = np.zeros((agg.params["rho_range"].size,))
    (Ns,n) = clf_preds.shape
    possib_save = copy.deepcopy(agg.params["cond_possib"])
    for kk in range(agg.params["rho_range"].size):
        alpha = 1 - np.power(agg.params["clf_scores"]/np.max(agg.params["clf_scores"]),agg.params["rho_range"][kk])
        alpha = alpha[:,None,None]
        agg.params["cond_possib"] = np.multiply(possib_save,1-alpha) + alpha
        preds_agg = agg.predict(clf_preds) 
        rho_success[kk] = np.sum(preds_agg==y)/n
    agg.params["rho"]=np.median(agg.params["rho_range"][np.where(rho_success == np.max(rho_success))])
    alpha = 1 - np.power(agg.params["clf_scores"]/np.max(agg.params["clf_scores"]),agg.params["rho"])
    alpha = alpha[:,None,None]
    #need to reset alphas and updated conditional possibilities
    agg.params["cond_possib"] = np.multiply(possib_save,1-alpha) + alpha
    if (display == True):        
        plt.figure()
        plt.plot(agg.params["rho_range"],rho_success)
        plt.show()
         
        
def heuris_adaspoccGridSearch(agg,clf_preds,y):
    count = 0
    (n_combi,_) = agg.params["dendro"].shape
    (Ns,n) = clf_preds.shape
    success = np.zeros((agg.params["hyper_range"].size,)) 
    evolution = True
    best = 0
    roots = [[],]
    for i in range(Ns):
        if (i==0):
            roots[i]=[i]
        else:
            roots.append([i])
    for i in range(n_combi):
        roots.append(roots[int(agg.params["dendro"][i,0])]+roots[int(agg.params["dendro"][i,1])])
    roots = roots[Ns:]    
    while (count<n_combi) and (evolution):
        n_clus = count + 2
        agg.params["clus"] = fcluster(agg.params["dendro"], n_clus, criterion='maxclust')
        agg.params["hyper"] = np.ones((n_combi,)) #re-init
        total = []
        ind_list = [] # list of lists containing the indices of hypers that must be updated together
        for i in range(n_clus): 
            members = list(np.where(agg.params["clus"]==(i+1))[0])
            inds=[]
            if len(members)>0:
                for j in range(n_combi):
                    is_sublist =  all(elem in members  for elem in roots[j])
                    if is_sublist:
                        inds.append(j)
                if len(inds)>0:
                    ind_list.append(inds)
                    total += inds
        remaining = list(set(list(range(n_combi))) - set(total)) 
        if len(remaining)>0:
            for i in remaining:
                ind_list.append([i])
        end=agg.params["hyper_range"].size
        for i in ind_list: #from the cluster with highest intra-dependency level to the smallest  
            range_loc = agg.params["hyper_range"][:end]
            success[end:]=0
            for k in range(range_loc.size):
                agg.params["hyper"][i]=range_loc[k]
                success[k] = np.sum(agg.predict(clf_preds)==y)/y.size
            if (range_loc.size>0):
                values = range_loc[np.where(success == np.max(success))]
                if (np.min(values)== np.min(range_loc)):
                    agg.params["hyper"][i]=np.min(values)
                elif (np.max(values)== np.max(range_loc)):
                    agg.params["hyper"][i]=np.max(values)
                else:
                    agg.params["hyper"][i]=values[int(len(values)/2)] #median
                best_loc = np.max(success)
                end = np.where(range_loc==np.min(agg.params["hyper"][i]))[0][0]           
        if best_loc>best:
            best = best_loc
            best_hyper = copy.deepcopy(agg.params["hyper"])
        else:
            evolution = False
        count +=1
    agg.params["hyper"] = best_hyper

        
def tnorm_op(possib,theta,tnorm):
    """
    This function compute the aggregation of possibility distribution using a
    specified tnorm.
    Parameters
    ----------    
     - possib : a 2D numpy array of floats
         It contains several possibility distributions that one wishes to combine.
    
    - theta : float
        A hyperparameter of the tnorm.
        
    - tnorm : string
        the name of the chose tnorm.
    Returns
    -------    
    - poss_agg : a 1D numpy array of floats
        It contains the aggregated possibility distribution
   
    """
    (Ns,n_class) = possib.shape
    poss_agg = np.ones((n_class,))
    if (tnorm == 'Aczel-Alsina'):
        
        if (theta == 0):  
            poss_agg = np.multiply(np.min(possib,axis=0),np.max(possib,axis=0)==1)
        elif (theta < np.inf) and (theta>0):
            zeros = np.where(np.min(possib,axis=0)==0)
            poss_agg[zeros]=0
            not_zeros = np.setdiff1d(np.arange(n_class),zeros)
            poss_agg[not_zeros] = np.exp(-np.power(np.sum(np.power(np.abs(np.log(possib[:,not_zeros])),theta),axis=0),1.0/theta))
            
            if (np.sum(np.isnan(poss_agg))):
                print('tnorm_op', poss_agg,possib )
        elif (theta == np.inf):
            poss_agg = np.min(possib,axis=0)
        else:
            raise ValueError('Invalid value for tnorm parameter theta which must be non negative.')
    elif (tnorm == 'convex'):
        poss_agg = (1-theta)*np.prod(possib,axis=0) + theta * np.min(possib,axis=0)
    else:
        raise ValueError('Unknown tnorm operation.')
    return poss_agg

def clf_prob(preds,n_class,laplace_smooth=True):
    """
    This function computes maximum likelihood estimates of the probabilities 
    that each classifier predicts a given class label.
    """
    (Ns,n) = preds.shape
    clf_mle = np.zeros((Ns,n_class))
    for i in range(Ns):
        for k in range(n_class):
            clf_mle[i,k] = np.sum((preds[i,:]==k).astype(int))
    counts = copy.deepcopy(clf_mle)
    if (laplace_smooth):
        clf_mle += 1
        clf_mle /= (n+n_class)
    else:
        clf_mle /= n
    return clf_mle, counts

def clf_pair_prob(preds,n_class,laplace_smooth=True):
    """
    This function computes maximum likelihood estimates of the joint 
    probabilities that each pair of classifier predicts a given pair of class 
    labels.
    """
    (Ns,n) = preds.shape
    clf_pair_mle = np.zeros((Ns,Ns,n_class,n_class))
    for i in range(Ns):
        for j in range(Ns):
            if (j<i):
                clf_pair_mle[i,j,:,:] = clf_pair_mle[j,i,:,:].T
            else:
                for k in range(n_class):
                    for o in range(n_class):
                        clf_pair_mle[i,j,k,o] = np.dot((preds[i,:]==k).astype(int),(preds[j,:]==o).astype(int))
    counts = copy.deepcopy(clf_pair_mle)
    if (laplace_smooth):
        clf_pair_mle += 1
        clf_pair_mle /= (n+n_class**2)
    else:
        clf_pair_mle /= n
    return clf_pair_mle, counts

def normalize(x):
    """
    This function normalizes an input array so that its entries sum to one.
    """
    return x/np.sum(x)

def prob2possib(prob):
    """Turn a probability distribution into a possibility distribution.
    Parameters
    ----------
    prob : 1D numpy array
        An array containing the probabilities
    Returns
    -------
    possib_out : 1D numpy array
        An array containing the possibilities
    """
    sorted_prob = np.sort(prob)[::-1]
    cum_prob = np.cumsum(sorted_prob[::-1])[::-1]
    sorted_pos = np.argsort(prob)[::-1]
    possib = np.ones(prob.shape)
    for i in range(len(prob)-1):
        if (sorted_prob[i+1]==sorted_prob[i]):
            possib[i+1] = possib[i]
        else:
            possib[i+1] = cum_prob[i+1]
    possib_out = np.ones(prob.shape)
    possib_out[sorted_pos] =  possib     
    return possib_out



def combinedPredPossib(preds,cond_possib,n_class,tnorm='Aczel-Alsina',theta=None, poss_values=False):
    """
    Computes the combined predictions in the possibilistic framework 
    based on those returned by classifiers and the specified tnorm.
    
    Parameters
    ----------    
     - preds : a 2D numpy array of int
         It containing the predictions returned by each 
         classifier individually. The 1st dimension is the number of classifier.
         The second dimension is the number of examples from which the predictions 
         are obtained.
    
    - cond_possib : 3D numpy array of float
        It is a collection of conditional possibibility distributions of predicted class given true 
        class for each classifier. The 1st dimension is the classifier index.
        The second dimension is the predicted class index. The third dimension
        is the true class index
        
    - tnorm : string
        The name of the chosen tnorm (Aczel-Alsina or franck)
        
    - theta : float or 1D numpy array of float
        The parameter of the tnorm .
        The parameters can be different for each true class.
    Returns
    -------
    - pred_out : 1D numpy array
        This array contains the prediction of the classifier ensemble for each
        example.
    """    
    if (preds.ndim==1):
        Ns_loc = preds.size
        n_loc = 1
        my_preds = np.reshape(preds,(Ns_loc,1))
    if (preds.ndim==2):
        (Ns_loc,n_loc) = preds.shape
        my_preds = preds
    if (np.isscalar(theta) == False):
        raise ValueError('theta must be a scalar.')
    (Ns_loc,n_class1,n_class2) = cond_possib.shape
    pred_out = np.zeros((n_loc,),dtype=int)
    poss_agg = np.zeros((n_loc,n_class))
    if (n_class1 != n_class2):
        raise ValueError('The second and third dimensions must agree (number of classes).')     
    for i in range(n_loc):#loop on examples
        possib_loc = np.asarray([])
        for j in range(Ns_loc):    
            possib_loc = np.append(possib_loc,cond_possib[j,my_preds[j,i],:])
        possib_loc = np.reshape(possib_loc,(Ns_loc,n_class1))
        poss_agg[i] = tnorm_op(possib_loc,theta,tnorm)
        pred_out[i] = np.argmax(poss_agg[i])
    if poss_values:
        return poss_agg
    else :
        return pred_out