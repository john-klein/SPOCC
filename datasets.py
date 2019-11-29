#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:14:47 2019

@author: johnklein
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.datasets import make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import genfromtxt
import pandas as pd
import pickle as pck
import _pickle as cPickle
import gzip
import sys
sys.path.insert(0, '../../synth_data/copule')

class synthDataset:
    """
    This class is a wrapper for synthetic datasets that are used in the 
    experiments. It allows to store metadata for each dataset and display them 
    using matplotlib figures.
    """
    def __init__(self, name, n, beta=0.5):
        self.name = name
        self.n = n   
        self.X = None
        self.y = None
        if (self.name == 'blobs'):
            self.d = 2
            self.n_class = 3
            self.Ns = 2
        elif (self.name == 'neoblobs'):
            self.d = 2
            self.n_class = 2
            self.Ns = 4    
            self.beta = beta
        elif (self.name == 'moons'):
            self.d = 2
            self.n_class = 2
            self.Ns = 3
        elif (self.name == 'circles'):
            self.d = 2
            self.n_class = 2
            self.Ns = 3
        else:
            raise ValueError('Unknown generating process name.')
        
    def genData(self,random_state=0):
        if (self.name == 'blobs'):
            (self.X,self.y) = gen_blobs(self.n, self.n_class, random_state=random_state)
        elif (self.name == 'neoblobs'):
            (self.X,self.y) = gen_blobs(self.n, self.n_class, random_state=random_state, beta=self.beta)
        elif (self.name == 'moons'):
            (self.X,self.y) = my_make_moons(n_samples=self.n,noise=0.3, random_state=random_state)
        elif (self.name == 'circles'):
            (self.X,self.y) = make_circles(n_samples=self.n, factor=.5, noise=.15, random_state=random_state)
       
    def fig(self, rep = None):
        fig2 = plt.figure(figsize=(4,4))
        if (self.name == 'blobs'):
            (X_all,y_all) = gen_blobs(self.n,self.n_class)
        if (self.name == 'neoblobs'):
            (X_all,y_all) = gen_blobs(self.n,self.n_class)            
        elif (self.name == 'moons'):
            (X_all,y_all) = my_make_moons(n_samples=self.n,noise=0.3)
        elif (self.name == 'circles'):
            (X_all,y_all) = make_circles(n_samples=self.n, factor=.5, noise=.15)
        else:
            raise ValueError('Unknown generating process name.')    
        plt.scatter(X_all[:, 0], X_all[:, 1], marker='o', c=y_all, s=25,cmap='tab10')
        
        fig3 = plt.figure(figsize=(4,4))
        plt.scatter(X_all[:, 0], X_all[:, 1], marker='o', c=y_all, s=25,cmap='tab10')
        x_min = np.min(X_all[:, 0])
        if (x_min<0):
            x_min *=1.05
        else:
            x_min *=0.95     
        x_max = np.max(X_all[:, 0])
        if (x_max>0):
            x_max *=1.05
        else:
            x_max *=0.95     
        y_min = np.min(X_all[:, 1])
        if (y_min<0):
            y_min *=1.05
        else:
            y_min *=0.95     
        y_max = np.max(X_all[:, 1])
        if (y_max>0):
            y_max *=1.05
        else:
            y_max *=0.95     
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        if (self.name == 'blobs'):
            plt.plot([x_min,x_max],[y_max,y_min],'--m')
        elif (self.name == 'neoblobs'):
            plt.plot([0,0],[y_max,y_min],'--m')
            plt.plot([x_min,x_max],[0,0],'--m')            
        elif (self.name == 'moons'):
            plt.plot([0,0],[y_max,y_min],'--m')
            plt.plot([1,1],[y_max,y_min],'--m')        
        elif (self.name == 'circles'):
            plt.plot([0,-y_min/np.sqrt(3)],[0,y_min],'--m')
            plt.plot([0,y_max/np.sqrt(3)],[0,y_max],'--m')
            plt.plot([0,x_min],[0,0],'--m')
        else:
            raise ValueError('Unknown generating process name.')      

        if (rep is not None):
            fig2.savefig(join(rep,self.name + '.png'),dpi=200)
            fig3.savefig(join(rep,self.name + '+split.png'),dpi=200)
        else:
            raise ValueError('Save directory "rep" not specified.') 

def simple_split(X_in,y_in,percent,n_class,shuffle=True):
    """
    This function splits a annotated dataset in two pieces. The relative size 
    of these is specified. The split is stratified.
    Parameters
    ----------    
     - X : a 2D numpy array of floats
         It contains training examples.
    
    - y : 1D numpy array of int
        It contains class labels of the training examples stored in X.
        
    - percent : float in the [0,1] range
        the proportion of the data in the first chunck of data.
        
    - n_class : int
        The number of classes.
    Returns
    -------    
    - X1 : a 2D numpy array of floats
        It contains the first chunck of training examples
        
    - y1 : 1D numpy array of int
        It containsclass labels of the training examples stored in X1.
        
    - X2 : a 2D numpy array of floats
        It contains the second chunck of training examples
        
    - y2 : 1D numpy array of int
        It containsclass labels of the training examples stored in X2.
                
    """
    cards = np.zeros((n_class,))
    inds = []
    if shuffle:
            X, y = util_shuffle(X_in, y_in, random_state=0)   
    else:
        X = X_in
        y = y_in
    for i in range(np.min(y),np.min(y)+n_class):
        cards[i-np.min(y)] = np.sum(y==i)
        proportion = int(percent*cards[i-np.min(y)])
        if (proportion == 0):
            proportion = 1
        inds = inds + list(np.where(y==i)[0][:proportion]) 
    X1 = X[inds]
    y1 = y[inds]
    inds2 = np.setdiff1d(range(len(y)),inds)
    X2 = X[inds2]
    y2 = y[inds2]   
    return X1,y1,X2,y2

class Oracle():
    """
    Optimal classifiers for each generating process.
    """    
    def __init__(self,name):
        self.name = name
        if (name not in ['blobs','moons','circles','neoblobs']):
            raise ValueError('Unknown generating process name.')
        
    def fit(self,X,y):
        #nothing to be done here
        pass
    
    def predict(self,X):

        preds = np.zeros((X.shape[0],),dtype=int)
        if (self.name == 'blobs'):
            ind = np.where((X[:,0]<0) & (X[:,1]>0))
            preds[ind] = 1
            ind = np.where((X[:,0]>0) & (X[:,1]<0))
            preds[ind] = 2
        if (self.name == 'neoblobs'):
            ind = np.where((X[:,0]<0) & (X[:,1]>0))
            preds[ind] = 1
            ind = np.where((X[:,0]>0) & (X[:,1]<0))
            preds[ind] = 1
        elif (self.name == 'moons'):
            angle = np.linspace(0,np.pi,100)
            outer_circ_x = np.cos(angle)
            outer_circ_y = np.sin(angle)
            inner_circ_x = 1 - np.cos(angle)
            inner_circ_y = 1 - np.sin(angle) - .5        
            for i in range(X.shape[0]):
                d0 = (outer_circ_x-X[i,:][0])**2 + (outer_circ_y-X[i,:][1])**2
                p0 = np.sum(np.exp(-d0/(2*0.3**2)))
                d1 = (inner_circ_x-X[i,:][0])**2 + (inner_circ_y-X[i,:][1])**2
                p1 = np.sum(np.exp(-d1/(2*0.3**2)))
                if (p1>p0):
                    preds[i]=1
        elif (self.name == 'circles'):
            d = (X[:,0])**2 + (X[:,1])**2
            preds = (d<0.75**2).astype(int)
        else:
            raise ValueError('Unknown generating process name.')
        return preds

def get_local_data(X,y,i,name):
    """
    This functions returns the piece of the dataset that will be used to train
    classifier number i.
    """
    if (name == 'blobs'):
        if (i==0):
            ind = np.where(X[:,0]<-X[:,1])[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==1):
            ind = np.where(X[:,0]>=-X[:,1])[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind] 
    elif (name == 'neoblobs'):
        if (i==0):
            ind = np.where(np.logical_or(X[:,0]<0 , X[:,1]>=0))[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==1):
            ind = np.where(np.logical_or(X[:,0]>=0 , X[:,1]>=0))[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]     
        if (i==2):
            ind = np.where(np.logical_or(X[:,0]>=0 , X[:,1]<0))[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==3):
            ind = np.where(np.logical_or(X[:,0]<0 , X[:,1]<0))[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]            
    elif (name == 'moons'):
        if (i==0):
            ind = np.where(X[:,0]<0)[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==1):
            ind = np.where((X[:,0]>=0) & (X[:,0]<1))[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==2):
            ind = np.where(X[:,0]>=1)[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]            
    elif (name == 'circles'):
        theta = np.arctan2(X[:,1], X[:,0])
        if (i==0):
            ind = np.where(theta<-np.pi/3)[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==1):
            ind = np.where((theta>=-np.pi/3) & (theta<np.pi/3))[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==2):
            ind = np.where(theta>=np.pi/3)[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]            
    else:
        raise ValueError('Unknown generating process name.')
    return X_train_loc,y_train_loc

def get_split_data(X_all,y_all,name,mode,Ns,n_class):
    """
    This functions splits the dataset into several pieces - on for each classifier
    to be trained. The splitting scheme is different for each generating process.
    """
    X=[]
    y=[]
    if (mode == 'deterministic_split'):
        for i in range(Ns):
            (X_loc,y_loc) = get_local_data(X_all,y_all,i,name)
            X.append(X_loc)
            y.append(y_loc)
    elif (mode == 'random_split'):
        X_train_folds, y_train_folds, X_test_folds, y_test_folds = kfold_data(X_all.T,y_all,Ns,n_class)
        for i in range(Ns):
            X_loc = X_test_folds[i].T
            y_loc = y_test_folds[i]
            X.append(X_loc)
            y.append(y_loc)
    else:
        raise ValueError('Unknown spliting mode name.')
    return X,y

        
def gen_blobs(n, n_class, beta=0.5, random_state=None):
    """
    This function generates a dataset from the following generating process: 
    4 gaussian 2D distributions centered on each corner of a centered square
    whose side length is 4. Each gaussian has unit variance
    The diagonal blobs generates example of class n° 0. The anti-diagonal blobs 
    generates examples of class n°1 and n°2 respectively.
    """
    if (n_class==3):
        X, y = make_blobs(n_samples=int(0.75*n),n_features=2, centers=np.asarray([[-2,-2],[-2,2],[2,-2] ]), random_state=random_state)
        X2, y2 = make_blobs(n_samples=int(0.25*n),n_features=2, centers=np.asarray([[2,2] ]), random_state=random_state)
        X = np.vstack((X,X2))
        y = np.hstack((y,y2))
    if (n_class==2):
        X1, y1 = make_blobs(shuffle=False,n_samples=int(n*beta),n_features=2, centers=np.asarray([[-1.5,-1.5],[-1.5,1.5] ]), random_state=random_state)
        X2, y2 = make_blobs(shuffle=False,n_samples=int(n*(1-beta)),n_features=2, centers=np.asarray([[-1.5,-1.5],[-1.5,1.5] ]), random_state=random_state)
        X3, y3 = make_blobs(shuffle=False,n_samples=int(n*beta),n_features=2, centers=np.asarray([[1.5,1.5],[1.5,-1.5] ]), random_state=random_state)
        X4, y4 = make_blobs(shuffle=False,n_samples=int(n*(1-beta)),n_features=2, centers=np.asarray([[1.5,1.5],[1.5,-1.5] ]), random_state=random_state)
        X = np.vstack((X1[np.where(y1==0)],X2[np.where(y2==1)],X3[np.where(y3==0)],X4[np.where(y4==1)]))
        y = np.hstack((y1[np.where(y1==0)],y2[np.where(y2==1)],y3[np.where(y3==0)],y4[np.where(y4==1)]))
        perm = np.random.permutation(len(y))
        X = X[perm]
        y = y[perm]
    return (X,y)

def my_make_moons(n_samples=100, shuffle=True, noise=None, random_state=None):
    """Make two interleaving half circles
    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    generator = check_random_state(random_state)

    angle = generator.uniform(size=(n_samples_out,))*np.pi
    outer_circ_x = np.cos(angle)
    outer_circ_y = np.sin(angle)
    angle = generator.uniform(size=(n_samples_out,))*np.pi    
    inner_circ_x = 1 - np.cos(angle)
    inner_circ_y = 1 - np.sin(angle) - .5

    X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise != None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y


def kfold_data(X,c,k,n_class):
    """
    A function to split a dataset according to the statified k fold cross validation
    principle. If there is less class representative example than folds, some
    folds may not contain any example of this class.
    """
    (d,n)=X.shape
    X_test_folds=[]
    c_test_folds=[]
    X_train_folds=[]
    c_train_folds=[]
    myrows=np.array(range(d),dtype=np.int)
    for i in range(k):
        for j in range(n_class):
            mycols=np.where(c==j)
            X_loc=X[myrows[:, np.newaxis], mycols]
            c_loc=c[mycols]
            (d,n_loc)=X_loc.shape
            fold_size=int(n_loc/k)
            if (fold_size>0):
                mycols=np.array(range(i*fold_size,(i+1)*fold_size),dtype=np.int)
            else:
                if (i<len(c_loc)):
                    mycols=np.array([i],dtype=np.int)
                else:
                    mycols=np.array([],dtype=np.int)
            if (j==0):
                X_test_fold=X_loc[myrows[:, np.newaxis], mycols]
                c_test_fold=c_loc[mycols]
            else :
                X_test_fold=np.hstack((X_test_fold,X_loc[myrows[:, np.newaxis], mycols]))
                c_test_fold=np.hstack((c_test_fold,c_loc[mycols]))
            mycols=np.setdiff1d(np.arange(0,n_loc), mycols)
            if (j==0):
                X_train_fold=X_loc[myrows[:, np.newaxis], mycols]
                c_train_fold=c_loc[mycols]
            else :
                X_train_fold=np.hstack((X_train_fold,X_loc[myrows[:, np.newaxis], mycols]))
                c_train_fold=np.hstack((c_train_fold,c_loc[mycols]))
        X_test_folds.append(X_test_fold)
        c_test_folds.append(c_test_fold)
        X_train_folds.append(X_train_fold)
        c_train_folds.append(c_train_fold)        
    return X_train_folds, c_train_folds, X_test_folds, c_test_folds


def pca3_split(X,y,n_class,Ns,inds_in):
    """
    This function splits the dataset into Ns pieces. 
    The split is deterministic. The feature space is divided into Ns regions.
    Each region is separated from another by a hyerplane that is orthogonal
    to the eigenvector number dim returned by a PCA.
    """   
    pca = PCA(n_components=X.shape[1])
    ind_loc = []
    for i in range(n_class):
        ind_class = np.where(y==i)[0]
        X_class = X[ind_class]
        pca.fit(X_class)
        u = pca.components_[0].T
        v_loc = np.dot(X_class,u)
        inds = np.argsort(v_loc.flatten())
        width = int(len(v_loc)/Ns)
        for j in range(Ns):
            ind = inds[j*width:(j+1)*width]
            if (i==0):
                ind_loc.append(inds_in[ind_class[ind]])
            else:
                ind_loc[j] = np.hstack((ind_loc[j],inds_in[ind_class[ind]]))
    return ind_loc

def load_mnist(filename):
    f = gzip.open(filename, 'rb')
    train_set, valid_set, test_set = cPickle.load(f, encoding='iso-8859-1')
    f.close()
    X = np.vstack((train_set[0],valid_set[0],test_set[0]))
    y = np.hstack((train_set[1],valid_set[1],test_set[1]))
    return X,y

class realDataset:
    """
    This class is a wrapper for synthetic datasets that are used in the 
    experiments. It allows to store metadata for each dataset and display them 
    using matplotlib figures.
    """
    def __init__(self, name, n, X, y, d, n_class, Ns=2):
        self.name = name
        self.n = n   
        self.X = X
        self.y = y
        self.d = d
        self.n_class = n_class
        self.Ns = Ns


def getRealData(dataset,rep,quick_test=False): 
   """
   This function loads real datasets. These datasets must be in directory rep
   and have appropriate names. The names are the default ones when they are
   downloaded from their corresponding repositeries.
   """
   if (dataset == 'mnist'):
       (X,y) = load_mnist(join(rep,'mnist.pkl.gz'))
       (n,d) = X.shape


   elif (dataset == '20newsgroup'):
       filename = join(rep,'20newsgroup_reduced')
       with open(filename, 'rb') as fp:
           X, y = pck.load(fp)
    
   elif (dataset == 'satellite'):
        f = open(join(rep,'sat.trn.txt'), 'r')
        rows = []
        for line in f:
            row = line.split()
            for j in range(len(row)):
                row[j]=int(row[j])
            rows.append(row)
        rows = np.asarray(rows,dtype=int)
        X = rows[:,:len(row)-1]
        y = rows[:,len(row)-1]
        
        f = open(join(rep,'sat.tst.txt'), 'r')
        rows = []
        for line in f:
            row = line.split()
            for j in range(len(row)):
                row[j]=int(row[j])
            rows.append(row)
        rows = np.asarray(rows,dtype=int)
        X = np.vstack((X,rows[:,:len(row)-1]))
        y = np.hstack((y,rows[:,len(row)-1]))
        #relabeling correctly the class indexes
        y -= 1
        ind = np.where(y==6)[0]
        y[ind] -=1
        
   elif (dataset == 'wine'):
        data = genfromtxt(join(rep,'winequality-red.csv'), delimiter=';')
        data = data[1:]
        X = data[:,:11]
        y = data[:,11].astype(int)
        data = genfromtxt(join(rep,'winequality-white.csv'), delimiter=';')
        data = data[1:]       
        X = np.vstack((X,data[:,:11]))
        y = np.hstack((y,data[:,11].astype(int)))
        #Binarizing the problem to avoid very imbalanced class issues
        ind = np.where(y<=5)[0]
        y[ind]=0
        ind = np.where(y>5)[0]
        y[ind]=1

   elif (dataset == 'spam'):
        data = genfromtxt(join(rep,'spambase.data.txt'), delimiter=',')
        X = data[:,:57]
        y = data[:,57].astype(int)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)        
        
   elif (dataset == 'avila'):
        data = genfromtxt(join(rep,'avila/avila-tr.txt'), delimiter=',')
        X = data[:,:10]
        y = []
        f = open(join(rep,'avila/avila-tr.txt'), 'r')
        rows =[]
        for line in f:
            row = line.split()
            rows.append(row)
        for i in range(len(rows)):
            if (rows[i][0][-1]=='A'):
                y.append(0)
            if (rows[i][0][-1]=='B'):
                y.append(0)            
            if (rows[i][0][-1]=='C'):
                y.append(0)
            if (rows[i][0][-1]=='D'):
                y.append(0)                
            if (rows[i][0][-1]=='E'):
                y.append(0)                
            if (rows[i][0][-1]=='F'):
                y.append(1)                
            if (rows[i][0][-1]=='G'):
                y.append(1)                
            if (rows[i][0][-1]=='H'):
                y.append(1)                
            if (rows[i][0][-1]=='I'):
                y.append(1)
            if (rows[i][0][-1]=='W'):
                y.append(1)
            if (rows[i][0][-1]=='X'):
                y.append(1)
            if (rows[i][0][-1]=='Y'):
                y.append(1)  
        y = np.asarray(y,dtype=int)
        
   elif (dataset == 'drive'):
        data = genfromtxt(join(rep,'Sensorless_drive_diagnosis.txt'), delimiter=' ')
        X = data[:,:48]
        y = data[:,48].astype(int)
        y -=1
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
   elif (dataset == 'particle'):
        df = pd.read_table(join(rep,'MiniBooNE_PID.txt'), sep='\s+', engine='python', header=None)
        X = df.values
        cards = genfromtxt(join(rep,'MiniBooNE_PID_cards.txt'), delimiter=' ')
        y = np.ones((X.shape[0],),dtype=int)
        y[:int(cards[0])]=0
        
   else:
       raise ValueError('Unknown dataset name.')
                
   if (quick_test == True):
        ######### MAKING IT FASTER ########
        ind0 = np.where(y==y.min())[0]
        ind0 = ind0[:int(ind0.size/2)]
        ind1 = np.where(y==y.min()+1)[0]
        ind1 = ind1[:int(ind1.size/2)]      
        ind = np.hstack([ind0,ind1])
        X = X[ind]
        y = y[ind]
        ###################################    

   (n,d) = X.shape
    
   #Number of classes
   n_class = y.max()-y.min()+1
   Dataset = realDataset(dataset, n, X, y, d, n_class)
   return Dataset

def kfold_data_inds(X,c,k,n_class):
    """
    A function to split a dataset according to the statified k fold cross validation
    principle. If there is less class representative example than folds, some
    folds may not contain any example of this class.
    """
    (d,n)=X.shape
    ind_test_folds=[]
    ind_train_folds=[]
    for i in range(k):
        for j in range(n_class):
            mycols=np.where(c==j)[0]
            n_loc=mycols.size
            fold_size=int(n_loc/k)
            if (fold_size>0):
                test_cols=mycols[range(i*fold_size,(i+1)*fold_size)]
            else:
                if (i<n_loc):
                    test_cols=np.array(mycols[i],dtype=np.int)
                else:
                    test_cols=np.array([],dtype=np.int)
            if (j==0):
                ind_test_fold = test_cols
            else :
                ind_test_fold = np.hstack((ind_test_fold,test_cols))
            train_cols=np.setdiff1d(mycols,test_cols)
            if (j==0):
                ind_train_fold = train_cols
            else :
                ind_train_fold = np.hstack((ind_train_fold,train_cols))
        ind_test_folds.append(ind_test_fold)
        ind_train_folds.append(ind_train_fold)       
    return ind_train_folds, ind_test_folds

def simple_split_inds(y,percent,n_class,inds_in):
    cards = np.zeros((n_class,))
    inds = []
    for i in range(np.min(y),np.min(y)+n_class):
        cards[i-np.min(y)] = np.sum(y==i)
        proportion = int(percent*cards[i-np.min(y)])
        if (proportion == 0):
            proportion = 1
        inds = inds + list(np.where(y==i)[0][:proportion]) 
    ind1 = inds_in[inds]
    inds2 = np.setdiff1d(range(len(y)),inds)
    ind2 = inds_in[inds2]  
    return ind1,ind2


