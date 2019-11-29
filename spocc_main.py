#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:30:39 2017

@author: johnklein
"""
import copy
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.stats import beta
from os import getcwd
from os.path import abspath, join, pardir
from sklearn.dummy import DummyClassifier
from functools import partial
import sys

if abspath(join(getcwd(), pardir)) not in sys.path:
    sys.path.append(abspath(join(getcwd(), pardir)))
from datasets import get_split_data, simple_split, synthDataset, Oracle
from aggregation import Aggregator, Predictor, Adversary, NoisyClf
        

def main_loop(random_state,n,Data,mode,Ns,predictors,sent,iter_max):
    dataset = Data.name
    epsi = 1e-7    
    degenerate = True
    while (degenerate == True):
        Data.genData(random_state=random_state)
        n_class = Data.n_class
                    
        #Spliting data for each local clf
        (X_loc,y_loc) = get_split_data(Data.X,Data.y,dataset,mode,Data.Ns,n_class)
        
        #Selecting the send examples to central machine for those methods with budgeted training
        X_loc_small = []
        y_loc_small = []
        for i in range(Data.Ns):
            X_s, y_s, X_i, y_i = simple_split(X_loc[i],y_loc[i],sent,n_class,shuffle=True)
            X_loc_small.append(X_i)
            y_loc_small.append(y_i)
            if (i==0):
                X_sent = copy.deepcopy(X_s)
                y_sent = copy.deepcopy(y_s)               
            else:
                X_sent = np.vstack((X_sent, copy.deepcopy(X_s)))
                y_sent = np.hstack((y_sent, copy.deepcopy(y_s)))
        
        degenerate = False      
        for i in range(Data.Ns):
            for j in range(n_class):
                if (np.sum(y_loc_small[i]==j)<1):
                    degenerate = True                    
                if (np.sum(y_sent==j)==0):
                    degenerate = True
    
    #Find who is who
    clf_ind = []                
    ctl_clf_ind = []
    others_ind = []
    oracle_ind = []
    for i in range(len(predictors)):
        if (predictors[i].kind == 'base_clf'):
            clf_ind.append(i)
        elif (predictors[i].kind == 'global'):
            ctl_clf_ind.append(i)
        elif (predictors[i].kind == 'oracle'):
            oracle_ind.append(i)
        else:
            others_ind.append(i)      
    
    score_clf = np.zeros((Ns,))
    preds = np.zeros((Ns,y_sent.size),dtype=int)
    for i in range(Data.Ns):#Classifier number
        #Training clfs on decentralized machines on datapoints that are not sent to central machine
        predictors[clf_ind[i]].machine.fit(X_loc_small[i],y_loc_small[i])
        #Evaluation in the central machine
        score_clf[i] = predictors[clf_ind[i]].machine.score(X_sent,y_sent)
        preds[i] = predictors[clf_ind[i]].machine.predict(X_sent)
        
    #Random or dummy clfs
    for i in range(Data.Ns,Ns):
        if (predictors[clf_ind[i]].isClone==False):
            predictors[clf_ind[i]].machine.fit(Data.X,Data.y)
        score_clf[i] = predictors[clf_ind[i]].machine.score(X_sent,y_sent)
        preds[i] = predictors[clf_ind[i]].machine.predict(X_sent)        
        
    #Training aggregators   
    for i in others_ind:
        predictors[i].machine.fit(preds,y_sent)
        if (isinstance(predictors[i].machine,Aggregator)):
            predictors[i].machine.gridSearch(preds,y_sent)
    
    #Retrain base classifiers on the whole training set
    for i in range(Data.Ns):
        predictors[clf_ind[i]].machine.fit(X_loc[i],y_loc[i])
    #Traning in centralized fashion
    predictors[ctl_clf_ind[0]].machine.fit(Data.X,Data.y)
    
    if(degenerate == False):#Going on only if it is worth 
        #Start testing on newly generated examples
        n_test = 0
        n_test_batch = 10000
        n_success = np.zeros((len(predictors),))
        conf_matrix = np.zeros((len(predictors),n_class,n_class))
        alpha = 0.05
        clopper_pearson_interval = np.ones((len(predictors),n_class, n_class))
        iter_nb = 0
        
        #Looping until Clopper Pearson interval meets prescribed conditions for the accurracy of each prediction method
        failed = False
        while ((np.max(clopper_pearson_interval) > 0.002) and (iter_nb<iter_max)): #3e5
        
            # Test Data generation
            Data_test = synthDataset(dataset,n_test_batch)  
            Data_test.genData(random_state=iter_nb)
            n_test += Data_test.y.size
            preds = np.zeros((Ns,Data_test.y.size),dtype=int)
            
            #base clfs
            for i in range(Ns):
                preds[i,:] = predictors[clf_ind[i]].machine.predict(Data_test.X)
                conf_matrix[clf_ind[i]] += confusion_matrix(Data_test.y,preds[i,:])
                n_success[clf_ind[i]] += np.sum(preds[i,:]==Data_test.y) 
                for j in range(n_class):
                    for k in range(n_class):
                        clopper_pearson_interval[clf_ind[i],j,k] = beta.ppf(1-alpha/2,conf_matrix[clf_ind[i],j,k]+1,n_test-conf_matrix[clf_ind[i],j,k]+epsi) - beta.ppf(alpha/2,conf_matrix[clf_ind[i],j,k]+epsi,n_test-conf_matrix[clf_ind[i],j,k]+1)
            
            #non aggregation methods
            for i in ctl_clf_ind+oracle_ind:
                meta_preds = predictors[i].machine.predict(Data_test.X)
                conf_matrix[i] += confusion_matrix(Data_test.y,meta_preds)
                n_success[i] += np.sum(meta_preds==Data_test.y)
                for j in range(n_class):
                    for k in range(n_class):                
                            clopper_pearson_interval[i,j,k] = beta.ppf(1-alpha/2,conf_matrix[i,j,k]+1,n_test-conf_matrix[i,j,k]+epsi) - beta.ppf(alpha/2,conf_matrix[i,j,k]+epsi,n_test-conf_matrix[i,j,k]+1)            

            #aggregation methods
            for i in others_ind:
                meta_preds = predictors[i].machine.predict(preds)
                conf_matrix[i] += confusion_matrix(Data_test.y,meta_preds)
                n_success[i] += np.sum(meta_preds==Data_test.y)
                for j in range(n_class):
                    for k in range(n_class):                
                            clopper_pearson_interval[i,j,k] = beta.ppf(1-alpha/2,conf_matrix[i,j,k]+1,n_test-conf_matrix[i,j,k]+epsi) - beta.ppf(alpha/2,conf_matrix[i,j,k]+epsi,n_test-conf_matrix[i,j,k]+1)            
            
            clopper_pearson_interval[np.where(np.isnan(clopper_pearson_interval))]=1
            iter_nb += 1
        if (iter_nb==iter_max):
            failed = True
   
        n_success /= n_test
        conf_matrix /= n_test
        return n_success,conf_matrix,failed


def launch_test(test_params):
    """
    This function instanciates the base classifiers and the aggregators. It 
    dispatches the computation on several cores of the micro-processor (if any).
    It also takes care of experimental data saving, figure edition and so on.
    """
    ##########################################
    # PARAMETERS BELOW SHOULD NOT BE CHANGED #
    ##########################################
    rate_resolution = 0.01   
    if (test_params["dataset"] in ['moons', 'circles', 'blobs', 'neoblobs']):
        n = test_params["n"]
        if 'beta' in test_params.keys():
            mybeta = test_params["beta"]
        else:
            mybeta=0.5
        Data = synthDataset(test_params["dataset"],n, beta=mybeta)
        synth_data = True
    else:
        raise ValueError("Unknown dataset name.")
    n_class = Data.n_class
    Ns = Data.Ns
    n_adver = 0
    if "adversary" in test_params.keys():
        for i in range(len(test_params["adversary"])):
            n_adver += test_params["adversary"][i][1]
    n_bad = 0        
    if "noisy" in test_params.keys():
        for i in range(len(test_params["noisy"])):
            n_bad += test_params["noisy"][i][1]   
    if 'names' in test_params.keys():
        names = test_params["names"]
    else:
        names = []
        for i in range(Ns):
            names.append("Tree")
        
    if (test_params["tnorm"] in ['Aczel-Alsina']):
        lambda_range = np.logspace(0,1,101)
        lambda_default = 5.0
    elif (test_params["tnorm"] in ['convex']):
        lambda_range = np.linspace(0.0,1.0,101)
        lambda_default = 0.5
    else:
        raise ValueError('Unknown tnorm name.')

    if (test_params["tnorm_ada"] in ['convex']):
        lambda_range_ada = np.linspace(0.0,1.0,101)
        lambda_default_ada = 0.5     
    elif (test_params["tnorm_ada"] in ['Aczel-Alsina']):
        lambda_range_ada = np.logspace(0,1,101)
        lambda_default_ada = 1.0    
    else:
        raise ValueError('Unknown tnorm name.')
    r_range = np.logspace(-1,2,201)
    alpha_range = np.linspace(0,1,101)     
    regul_range = np.logspace(-2,2,101)
    rho_range = np.logspace(-2,2,101)
    
    # Models
    clf = []
    predictors = []
    for i in range(Ns):
        if names[i] == 'Tree':
            clf.append(DecisionTreeClassifier(max_depth=2))
        if names[i] == 'Reg Log':    
            clf.append(LogisticRegression(penalty='l2', C=1.0, solver='lbfgs'))
        if names[i] == 'NBC':    
            clf.append(GaussianNB())
        if names[i] == 'QDA':    
            clf.append(QuadraticDiscriminantAnalysis())
        if names[i] == 'SVM_lin':    
            clf.append(SVC(kernel="linear", C=0.025))        
        if names[i] == 'SVM_nlin':    
            clf.append(SVC(gamma=2, C=1))     
        if names[i] == 'GP':    
            clf.append(GaussianProcessClassifier(1.0 * RBF(1.0)))   
        if names[i] == 'RF':    
            clf.append(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)) 
        if names[i] == 'MLP':    
            clf.append(MLPClassifier(alpha=1))           
        if names[i] == 'Ada':    
            clf.append(AdaBoostClassifier())
        if names[i] == 'kNN':
            clf.append(KNeighborsClassifier(n_neighbors=5))
        predictors.append(Predictor('Base Clf '+str(i+1), 'base_clf', clf[i], rate_resolution, 'k'))
        
    adv = []
    for i in range(len(test_params["adversary"])):
        for j in range(test_params["adversary"][i][1]):
            adv.append(Adversary(clf[test_params["adversary"][i][0]], n_class, test_params["adversary"][i][2]))
            predictors.append(Predictor('Advers. Clf '+str(i+1), 'base_clf', adv[i], rate_resolution, 'k'))
            Ns += 1
    
    bad = []
    for i in range(len(test_params["noisy"])):
        for j in range(test_params["noisy"][i][1]):
            bad.append(NoisyClf(clf[test_params["noisy"][i][0]], n_class, test_params["noisy"][i][2]))
            predictors.append(Predictor('Noisy Clf '+str(i+1), 'base_clf', bad[i], rate_resolution, 'k'))
            Ns += 1    
        
    for i in range(test_params["clone"]):
        predictors.append(predictors[0].clone())
        predictors[-1].name = 'Clone ' +str(i+1)
        clf.append(predictors[-1].machine)
        Ns += 1
        
    for i in range(test_params["random"]):
        clf.append(DummyClassifier(strategy='uniform'))
        predictors.append(Predictor('Random Clf '+str(i+1), 'base_clf', clf[-1], rate_resolution, 'k'))
        Ns += 1
        
    for i in range(test_params["dummy"]):
        clf.append(DummyClassifier(strategy='most_frequent'))
        predictors.append(Predictor('Constant Clf '+str(i+1), 'base_clf', clf[-1], rate_resolution, 'k'))
        Ns += 1    
       
    selec = Aggregator('selection', Ns, n_class)
    predictors.append(Predictor('Selected Clf.', 'agg', selec, rate_resolution, 'b'))
    
    wvote = Aggregator('weighted_vote', Ns, n_class, params = {"r_range" : r_range, "expo": False}) 
    predictors.append(Predictor('Weighted Vote Ens.', 'agg', wvote, rate_resolution, 'orange'))

    expow = Aggregator('weighted_vote', Ns, n_class, params = {"r_range" : r_range}) 
    predictors.append(Predictor('Expo. Weighted Vote Ens.', 'agg', expow, rate_resolution, 'brown'))
    
    naive = Aggregator('naive', Ns, n_class, params = {"method" : 'indep'} )
    predictors.append(Predictor('Naive Bayes.', 'agg', naive, rate_resolution, '--m'))
       
    spocc = Aggregator('spocc', Ns, n_class, {"tnorm" : test_params["tnorm"], "hyper" : lambda_default, "hyper_range" : lambda_range} )
    predictors.append(Predictor('SPOCC (' + test_params["tnorm"] + ')', 'agg', spocc, rate_resolution, 'm'))
    
    adaspocc = Aggregator('adaspocc', Ns, n_class, {"tnorm" : test_params["tnorm_ada"], "hyper" : lambda_default_ada, "hyper_range" : lambda_range_ada, "alpha_range" : alpha_range, "rho" : 1, "rho_range" : rho_range} )
    predictors.append(Predictor('adaSPOCC (' + test_params["tnorm_ada"] + ')', 'agg', adaspocc, rate_resolution, 'm'))
    
    stack = Aggregator('stacked_logreg', Ns, n_class, {"regul_range" : regul_range})
    predictors.append(Predictor('Stacked Log. Reg.', 'agg', stack, rate_resolution, 'r'))
            
    clf_ctl = LogisticRegression(penalty='l2', C=1.0)
    predictors.append(Predictor('Centralized Clf.', 'global', clf_ctl, rate_resolution, '--g'))
    
    if (test_params["dataset"] not in ['20newsgroup','mnist','drive']):
        bayes = Aggregator('bayes', Ns, n_class)
        predictors.append(Predictor('Bayes Agg.', 'agg', bayes, rate_resolution, ':g'))
    
    if synth_data:
        optim = Oracle(test_params["dataset"])
        predictors.append(Predictor('Optimal Clf.', 'oracle', optim, rate_resolution, 'g'))   
             
    main_loop_partial = partial(main_loop, n=n,Data=Data,mode=test_params["mode"],Ns=Ns,predictors=predictors,sent=test_params["sent"],iter_max=test_params["iter_max"])
    results = main_loop_partial(test_params["rand_state"])
    accuracy,conf_matrix,failed_loc = results  
    for i in range(len(predictors)):    
        print(predictors[i].name+" has accuracy "+str(accuracy[i]))


def example():
    """
    This function shows how to run the set of compared aggregation methods on a synthetic dataset called neoblobs.
    """
    test_params = {
    "rand_state": 0, # seed of the random generator
    "tnorm" : 'Aczel-Alsina', #
    "tnorm_ada" : 'Aczel-Alsina',#
    "dataset" : 'neoblobs', # 
    "mode" : 'deterministic_split', # 'random_split', 'deterministic_split'
    "h" : .02,  # step size in the mesh
    "n" : 200, #  dataset size
    "sent" : 0.2, # validation set portion
    "iter_max" : 1000, # this parameter can be set to np.inf for very accurate evaluation but values such 1000 are already fairly enough.
    "random" : 0,# number of random classifiers
    "dummy" : 0,# number of dummy classifiers
    "clone" : 2,# number of clones of the first base classifier
    "adversary" : [(0,2,0.5),], # list of tuples. Each is (clf id, number of instances, bernoulli proba ),
    "noisy" : [(0,2,0.5),], # list of tuples. Each is (clf id, number of instances, bernoulli proba )
    "beta" : 0.5 # class imbalance
    }
    launch_test(test_params)
