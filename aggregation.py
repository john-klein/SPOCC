#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:28:42 2019

@author: johnklein
"""
import numpy as np
from numbers import Number
from sklearn.linear_model import LogisticRegression
from spocc_API import HAC_aggregation_fit, HAC_aggregation_predict, heuris_adaspoccGridSearch, prob2possib, combinedPredPossib, smax, rhoGridSearch
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def weighted_vote(preds,acc,n_class):
    """
    A simple classifier combination based on a weighted vote. Each classifier 
    vote is weighted according to its estimated accuraccy.
    """
    (Ns,n) = preds.shape
    votes = np.zeros((n,n_class))
    for i in range(Ns):
        votes[range(n), preds[i,:]] += acc[i]
    pred_out = np.argmax(votes,axis=1)
    return pred_out

def rGridSearch_expoWeights(agg,clf_preds,y, display=False):
    if (agg.params["expo"]):             
        r_success = np.zeros((agg.params["r_range"].size,))
        (Ns,n) = clf_preds.shape
        for kk in range(agg.params["r_range"].size):
            agg.params["weights"] = smax(agg.params["clf_scores"],agg.params["r_range"][kk])
            preds_agg = agg.predict(clf_preds) 
            r_success[kk] = np.sum(preds_agg==y)/n
        agg.params["r"]=np.median(agg.params["r_range"][np.where(r_success == np.max(r_success))])
        agg.params["weights"] = smax(agg.params["clf_scores"],agg.params["r"])
        if (display == True):        
            plt.figure()
            plt.plot(agg.params["r_range"],r_success)
            plt.show() 
            
def regulGridSearch(agg,clf_preds,y, display=False):
    r_success = np.zeros((agg.params["regul_range"].size,))
    (Ns,n) = clf_preds.shape
    for kk in range(agg.params["regul_range"].size):
        agg.params["clf_meta"] = LogisticRegression(penalty='l2', C=agg.params["regul_range"][kk])
        agg.params["clf_meta"].fit(clf_preds.T,y)
        preds_agg = agg.predict(clf_preds) 
        r_success[kk] = np.sum(preds_agg==y)/n
    agg.params["hyper"]=np.median(agg.params["regul_range"][np.where(r_success == np.max(r_success))])
    agg.params["clf_meta"] = LogisticRegression(penalty='l2', C=agg.params["hyper"])
    agg.params["clf_meta"].fit(clf_preds.T,y)
    if (display == True):        
        plt.figure()
        plt.plot(agg.params["regul_range"],r_success)
        plt.show()         

class Aggregator:
    """
    This class defines a general scheme for classifier aggregation methods. It
    is meant to have similar functionalities as classifier classes from sklearn.
    """
    #Constructor
    def __init__(self, name, Ns, n_class, params = None):
        self.name = name
        self.Ns = Ns
        self.n_class = n_class
        if (params is None):
            self.params = {}
        else:
            self.params = params
            
        #For each type of aggregator, set default parameters or check if the provided one are correct.
        if (name == 'naive'):
            if ("cond_pdf" in self.params):
                if (self.params["cond_pdf"].shape[1] != self.n_class):
                    raise ValueError('Incorrect array size. Conditional probabilities is 3D array with shape (Ns,n_class,n_class)')
            else:
                self.params["cond_pdf"]=None  
            if ("prior" in self.params):
                if (self.params["prior"].shape != (self.n_class,)):
                    raise ValueError('Incorrect array size. Prior probabilities is 1D array with shape (n_class,)')
            else:
                self.params["prior"]=None
                

        elif (name == 'spocc'):
            if ("tnorm" in self.params):
                if (self.params["tnorm"] not in ['Aczel-Alsina','convex']):
                    raise ValueError('Unknown tnorm type.')
            else:
                self.params["tnorm"]='convex'            
            if ("cond_pdf" in self.params):
                if (self.params["cond_pdf"].shape[1] != self.n_class):
                    raise ValueError('Incorrect array size. Conditional probabilities is 3D array with shape (Ns,n_class,n_class)')
            else:
                self.params["cond_pdf"]=None
            if ("cond_possib" in self.params):
                if (self.params["cond_possib"].shape[1] != self.n_class):
                    raise ValueError('Incorrect array size. Conditional possibilities is 3D array with shape (Ns,n_class,n_class)')
            else:
                self.params["cond_possib"]=None                
            if ("hyper" in self.params):
                if not(isinstance(self.params["hyper"], Number)):
                    raise ValueError('Invalid type for parameter hyper which must be a number')
                else:
                    if (self.params["hyper"]<0):
                        raise ValueError('Invalid value for parameter hyper which must be positive.')
            else:
                self.params["hyper"] = 0.0  
            if ("hyper_range" in self.params):
                if (self.params["hyper_range"].ndim !=1):
                    raise ValueError('Invalid array dimension. hyper_range must be a 1D array')
            else:
                self.params["hyper_range"] = np.logspace(-2,1,101)     
                
        elif (name == 'adaspocc'):
            self.params["alpha"]=None
            self.params["dendro"]=None
            if ("clf_scores" in self.params):
                if (self.params["clf_scores"].shape != (self.Ns,)):
                    raise ValueError('Incorrect shape for parameter clf_scores which must have shape (Ns,).')
            else:
                self.params["clf_scores"] = None            
            if ("tnorm" in self.params):
                if (self.params["tnorm"] not in ['convex','Aczel-Alsina']):
                    raise ValueError('Unknown tnorm type.')
            else:
                self.params["tnorm"]='convex'            
            if ("cond_pdf" in self.params):
                if (self.params["cond_pdf"].shape[1] != self.n_class):
                    raise ValueError('Incorrect array size. Conditional probabilities is 3D array with shape (Ns or more,n_class,n_class)')
            else:
                self.params["cond_pdf"]=None
            if ("cond_possib" in self.params):
                if (self.params["cond_possib"].shape[1] != self.n_class):
                    raise ValueError('Incorrect array size. Conditional possibilities is 3D array with shape (Ns or more,n_class,n_class)')
            else:
                self.params["cond_possib"]=None  
            if ("r" in self.params):
                if not(isinstance(self.params["r"], Number)):
                    raise ValueError('Invalid type for parameter r which must be a number')
                else:
                    if (self.params["r"]<0):
                        raise ValueError('Invalid value for parameter r which must be positive.')
            else:
                self.params["r"] = 10.0
                
            if ("hyper" in self.params):
                if not(isinstance(self.params["hyper"], Number)):
                    raise ValueError('Invalid type for parameter hyper which must be a number')
                else:
                    if (self.params["hyper"]<0):
                        raise ValueError('Invalid value for parameter hyper which must be positive.')
            else:
                self.params["hyper"] = 0.0  
            if ("hyper_range" in self.params):
                if (self.params["hyper_range"].ndim !=1):
                    raise ValueError('Invalid array dimension. hyper_range must be a 1D array')
            else:
                self.params["hyper_range"] = np.logspace(-2,1,101)  
            if ("alpha_range" in self.params):
                if (self.params["alpha_range"].ndim !=1):
                    raise ValueError('Invalid array dimension. alpha_range must be a 1D array')
            else:
                self.params["alpha_range"] = np.linsapce(0,1,101)                 
                
        elif (name == 'stacked_logreg'):
            if ("hyper" in self.params):
                if not(isinstance(self.params["hyper"], Number)):
                    raise ValueError('Invalid type for parameter hyper which must be a number')
                else:
                    if (self.params["hyper"]<0) :
                        raise ValueError('Invalid value for parameter hyper which must be positive.')
            else:
                self.params["hyper"] = 1.0            
            if ("clf_meta" in self.params):
                if not(isinstance(self.params["clf_meta"], LogisticRegression)):
                    raise ValueError('Invalid type of object. clf_meta must be an instance of LogisticRegression from sklearn.')
            else:
                self.params["clf_meta"] = LogisticRegression(penalty='l2', C=self.params["hyper"])
            if ("regul_range" in self.params):
                if (self.params["regul_range"].ndim !=1):
                    raise ValueError('Invalid array dimension. regul_range must be a 1D array')
            else:
                self.params["regul_range"] = np.logspace(-2,2,101)                
                
        elif (name == 'weighted_vote'): 
            if ("clf_scores" in self.params):
                if (self.params["clf_scores"].shape != (self.Ns,)):
                    raise ValueError('Incorrect shape for parameter clf_scores which must have shape (Ns,).')
            else:
                self.params["clf_scores"] = None
            if ("r" in self.params):
                if not(isinstance(self.params["r"], Number)):
                    raise ValueError('Invalid type for parameter r which must be a number')
                else:
                    if (self.params["r"]<0):
                        raise ValueError('Invalid value for parameter r which must be positive.')
            else:
                self.params["r"] = 10.0                
            if ("r_range" in self.params):
                if (self.params["r_range"].ndim !=1):
                    raise ValueError('Invalid array dimension. r_range must be a 1D array')
            else:
                self.params["r_range"] = np.logspace(-1,2,201)
            if("expo" in self.params):
                if (type(self.params["expo"])!=bool):
                     raise ValueError('Parameter expo must be a bool.')
            else:
                self.params["expo"]=True
                
        elif (name == 'selection'):
            if ("clf_scores" in self.params):
                if (self.params["clf_scores"].shape != (self.Ns,)):
                    raise ValueError('Incorrect shape for parameter clf_scores which must have shape (Ns,).')
            else:
                self.params["clf_scores"] = None
            if ("select" in self.params):
                if (type(self.params["select"]) is not int):
                    raise ValueError('Invalid type for parameter select which must be an integer.')
            else:
                self.params["select"] = None
        elif (name == 'bayes'):
            size = (self.n_class,)
            for i in range(Ns):
                size += (self.n_class,)
            self.params["cond_probas"] = np.zeros(size)
        else:
            raise ValueError('Unknown type of Aggregator.')
            
    #Methods
            
    def fit(self,clf_preds,y):
        if (self.name == 'stacked_logreg'):
            self.params["clf_meta"].fit(clf_preds.T,y)
        if (self.name == 'naive'):
            if (self.params["cond_pdf"] is None):
                #Estimation of conditional probabilities (predicted class given actual class)
                self.params["cond_pdf"] = np.zeros((self.Ns,self.n_class,self.n_class))
                for i in range(self.Ns):#Classifier number
                    for j in range(self.n_class):#True class index
                        ind = np.where(y==j)[0]
                        for k in range(self.n_class):#Predicted class index
                            self.params["cond_pdf"][i][k,j] += (np.sum(clf_preds[i,ind]==k)+1)/(ind.size+self.n_class) #Laplace add one
            if (self.params["prior"] is None):
                self.params["prior"] = np.zeros((self.n_class,))
                for i in range(self.n_class):
                    self.params["prior"][i] = np.sum(y==i)/y.size
        if (self.name in ['spocc']):
            if (self.params["cond_pdf"] is None):
                #Estimation of conditional probabilities (actual class given predicted class)
                self.params["cond_pdf"] = np.zeros((self.Ns,self.n_class,self.n_class))
                for i in range(self.Ns):#Classifier number    
                    for j in range(self.n_class):#Predicted class index
                        ind = np.where(clf_preds[i]==j)[0]
                        for k in range(self.n_class):#True class index
                            self.params["cond_pdf"][i][j,k] += (np.sum(y[ind]==k)+1)/(ind.size+self.n_class) #Laplace add one
                
                #Transformation into possibility distributions
                self.params["cond_possib"] = np.zeros(self.params["cond_pdf"].shape)
                for i in range(self.Ns):#Classifier number
                    for j in range(self.n_class):#Predicted class index
                        self.params["cond_possib"][i][j] = prob2possib(self.params["cond_pdf"][i][j])
                        
        if (self.name == 'adaspocc'):
            if (self.params["alpha"] is None):
                self.params["alpha"] = np.zeros((self.Ns,)) 
            if (self.params["clf_scores"] is None):    
                self.params["clf_scores"] = np.zeros((self.Ns,))
                for i in range(self.Ns):#Classifier number
                    self.params["clf_scores"][i] = np.sum(clf_preds[i]==y)
                    self.params["clf_scores"][i] /= len(y)
            if (self.params["cond_pdf"] is None):
                #Estimation of conditional probabilities (actual class given predicted class)
                self.params["cond_pdf"] = np.zeros((self.Ns,self.n_class,self.n_class))
                for i in range(self.Ns):#Classifier number    
                    for j in range(self.n_class):#Predicted class index
                        ind = np.where(clf_preds[i]==j)[0]
                        for k in range(self.n_class):#True class index
                            self.params["cond_pdf"][i][j,k] += (np.sum(y[ind]==k)+1)/(ind.size+self.n_class) #Laplace add one
                      
                #Transformation into possibility distributions
                self.params["cond_possib"] = np.zeros(self.params["cond_pdf"].shape)
                for i in range(self.Ns):#Classifier number
                    for j in range(self.n_class):#Predicted class index
                        self.params["cond_possib"][i][j] = prob2possib(self.params["cond_pdf"][i][j])
                self.params["cond_possib_saved"] = copy.deepcopy(self.params["cond_possib"])
                #Discounting
                for i in range(self.Ns):
                    self.params["cond_possib"][i] = (1-self.params["alpha"][i])*self.params["cond_possib"][i] + self.params["alpha"][i]
            (Ns,n) = clf_preds.shape        
            HAC_aggregation_fit(clf_preds,self,self.n_class,y,rho=1/n)
            (n_combi,n) = self.params["dendro"].shape
            self.params["hyper"] = self.params["hyper"] + np.zeros((n_combi,))
                        
        if (self.name in ['weighted_vote', 'selection']):
            if (self.params["clf_scores"] is None):
                self.params["clf_scores"] = np.zeros((self.Ns,))
                for i in range(self.Ns):#Classifier number
                    self.params["clf_scores"][i] = np.sum(clf_preds[i]==y)
                    self.params["clf_scores"][i] /= len(y)
        
        if (self.name in ['weighted_vote']):        
            if (self.params["expo"]):
                self.params["weights"] = smax(self.params["clf_scores"],self.params["r"])
            else:
                self.params["weights"] = self.params["clf_scores"]

        if (self.name == 'selection'):
            if (self.params["select"] is None):
                self.params["select"] = np.argmax(self.params["clf_scores"])
                
        if (self.name == 'bayes'):
            (Ns,n) = clf_preds.shape
            for i in range(n):
                self.params["cond_probas"][tuple(clf_preds[:,i])][y[i]] += 1
            self.params["cond_probas"] /= y.size
             
            
    def predict(self, clf_preds):
        if (self.name == 'stacked_logreg'):
            preds = self.params["clf_meta"].predict(clf_preds.T)
        if (self.name == 'naive'):
            n = clf_preds.shape[1]
            preds = np.zeros((n,))
            for i in range(n):
                preds[i] = combinedPred(clf_preds[:,i],self.params["cond_pdf"],self.params["prior"],method='indep')
        if (self.name == 'spocc'):
            n = clf_preds.shape[1]
            preds = np.zeros((n,))
            for i in range(n):
                preds[i] = combinedPredPossib(clf_preds[:,i],self.params["cond_possib"],self.n_class,tnorm=self.params["tnorm"],theta=self.params["hyper"])
        if (self.name == 'adaspocc'):
            preds = HAC_aggregation_predict(clf_preds,self,self.n_class,rho=1)
        if (self.name == 'weighted_vote'):
            preds = weighted_vote(clf_preds,self.params["weights"],self.n_class)
        if (self.name == 'selection'):
            preds = clf_preds[self.params["select"],:]
        if (self.name == 'bayes'):
            (Ns,n) = clf_preds.shape
            preds = np.zeros((n,))
            for i in range(n):
                preds[i] = np.argmax( self.params["cond_probas"][tuple(clf_preds[:,i])] )
        return preds
    
    def gridSearch(self, clf_preds, y, display = False):
        if (self.name in ['spocc']):
            theta_success = np.zeros((self.params["hyper_range"].size,))       
            for j in range(self.params["hyper_range"].size):
                self.params["hyper"]=self.params["hyper_range"][j]
                comb_pred = self.predict(clf_preds)
                theta_success[j] = np.sum(comb_pred==y)/y.size
            theta = np.median(self.params["hyper_range"][np.where(theta_success == np.max(theta_success))])
            self.params["hyper"]=theta
            if (display == True):
                plt.figure()
                plt.plot(self.params["hyper_range"],theta_success)
                plt.show()
        if (self.name == 'adaspocc'):
            heuris_adaspoccGridSearch(self,clf_preds,y)
            rhoGridSearch(self,clf_preds,y)
        if (self.name == 'weighted_vote'):
            rGridSearch_expoWeights(self,clf_preds,y)
        if (self.name == 'stacked_logreg'):    
            regulGridSearch(self,clf_preds,y)
        
class Predictor():
    """
    This class is a wrapper for base classifiers or aggregators. It allows to
    associate to each such object some arrays which contain results of the 
    experiments.
    """
    
    def __init__(self, name, kind, machine, rate_resolution, style='k'):
        self.name = name
        self.kind = kind
        self.machine = machine
        self.hist_size = int(1.0/rate_resolution)
        self.acc_hist = np.zeros((self.hist_size,))
        self.acc_list = np.zeros((0,)) 
        self.conf_matrix_list = []
        self.cp_intervals = np.ones((self.hist_size,))
        self.style = style
        self.hist_frontiers = np.arange(rate_resolution,1.0+rate_resolution,rate_resolution)
        self.bin_centers = self.hist_frontiers-rate_resolution/2
        self.isClone = False
        
        
    def stats(self,):
        exp = np.mean(self.acc_list) 
        std = np.std(self.acc_list) 
        return exp, std
    
    def confMatrix(self,):
        if (self.conf_matrix_list[0].ndim == 2):
            mat = np.mean(np.array(self.conf_matrix_list),axis=0)
        elif (self.conf_matrix_list[0].ndim == 3):
            mat = np.mean(np.array(self.conf_matrix_list),axis=(0, 1))
        else:
            raise ValueError('Unexpected number of dimensions for confusion matrix of predictor.')
        return mat
    
    def clone(self, shadow=True):
        if shadow: #copy of the pointer
            machine = self.machine
        else: #new instance with identical profile
            machine = copy.deepcopy(self.machine)
        out = copy.deepcopy(self)
        out.machine = machine
        out.isClone = True
        return out
    
class Adversary():
    """
    """
    def __init__(self, clf, n_class, theta=0.5):
        self.clf = clf
        self.theta = theta
        self.n_class = n_class
        
    def fit(self,X,y):
        if not hasattr(self.clf,"classes_"): #check if already fitted
            self.clf.fit(X,y)
            
    def predict(self,X):
        (n,d) = X.shape
        cheat = np.random.binomial(1, self.theta,(n,))
        ind = np.where(cheat==1)
        modif = np.random.randint(0,self.n_class,(ind[0].size,))
        preds = self.clf.predict(X)
        shift_needed = np.where(np.equal(preds[ind],modif))
        modif[shift_needed] += 1 
        modif[shift_needed] %= self.n_class
        preds[ind] = modif
        return preds
        
    def score(self,X,y):
        return np.sum(self.predict(X)==y)/y.size
    
class NoisyClf():
    """
    """
    def __init__(self, clf, n_class, theta=0.5):
        self.clf = clf
        self.theta = theta
        self.n_class = n_class
        
    def fit(self,X,y):
        if not hasattr(self.clf,"classes_"): #check if already fitted
            self.clf.fit(X,y)
            
    def predict(self,X):
        (n,d) = X.shape
        perturbation = np.random.binomial(1, self.theta,(n,))
        ind = np.where(perturbation==1)
        modif = np.random.randint(0,self.n_class,(ind[0].size,))
        preds = self.clf.predict(X)
        preds[ind] = modif
        return preds
          
    def score(self,X,y):
        return np.sum(self.predict(X)==y)/y.size
    

def pdf2cdf(pdf):
    """
    This function turns a set of joint pdfs to a set of joint cdfs.
    Warning: only works for a pair of classifiers.
    input pdf is a collection of conditional 2D joint cdfs (given true class).
    """
    (Ns_loc,n_class1,n_class2) = pdf.shape
    if (n_class1 != n_class2):
        raise ValueError('The second and third dimensions must agree (number of classes).')
    cdf = np.zeros((Ns_loc,n_class1,n_class1))
    for i in range(Ns_loc):
        for j in range(n_class1):
            cdf[i][:,j] = np.cumsum(pdf[i][:,j])
    return cdf


def combinedPred(preds,cond_pdfs,priors,method='indep'):
    """
    Computes the combined predictions based on those returned by classifiers 
    and the naive Bayes aggregation.
    
    Parameters
    ----------    
     - preds : a 2D numpy array of int
         It containing the predictions returned by each 
         classifier individually. The 1st dimension is the number of classifier.
         The second dimension is the number of examples from which the predictions 
         are obtained.
    
    - cond_pdfs : 3D numpy array of float
        It is a collection of conditional pdfs of predicted class given true 
        class for each classifier. The 1st dimension is the classifier index.
        The second dimension is the predicted class index. The third dimension
        is the true class index
        
    - priors : 1D numpy array of float
        Probabilities of true classes.
        
    - method : indep
        

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

    (Ns_loc,n_class1,n_class2) = cond_pdfs.shape
    pred_out = np.zeros((n_loc,),dtype=int)
    if (n_class1 != n_class2):
        raise ValueError('The second and third dimensions must agree (number of classes).')    
    for i in range(n_loc):#loop on examples
        pdf = np.asarray([])
        for j in range(Ns_loc):     
            pdf = np.append(pdf,cond_pdfs[j,my_preds[j,i],:])
        pdf = np.reshape(pdf,(Ns_loc,n_class1))
        comb_pdf = np.prod(pdf,axis=0)
        pred_out[i] = np.argmax(np.multiply(comb_pdf,priors))
    return pred_out