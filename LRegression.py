

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('crime_data_processed.csv')


def prepare(data):
    X=data.as_matrix(columns=['avg_hatecrimes_per_100k_fbi','share_voters_voted_trump'])
    
    maxX = np.max(X, axis=0)
    minX = np.min(X, axis=0)
    X = (X-minX)/(maxX-minX)
    X = np.insert(X, 0, 1, axis=1)
    y=data.as_matrix(columns=data.columns[12:13])
    return X,y

X,y = prepare(data)
    


def errormean(X,y,w):
    sUm = np.sum((y.T-w@X.T)**2)
    div = (1/(2*X.shape[0]))
    return div*sUm

def gradmean(X,y,w):
    sUm = np.sum((y.T-w@X.T).T*X,axis=0,keepdims=True)
    div = -(1/X.shape[0])
    return div*sUm


def fIt(X,y,kappa,iter):
    w = np.ones((1,X.shape[1]))
    E = []
    for loop in range(iter):
        E.append(errormean(X,y,w))
        w = w - kappa*(gradmean(X,y,w))
    return w,E

w,E = fIt(X,y,0.5,100)
plt.plot(E)
plt.show()



