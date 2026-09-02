
import numpy as np
import sys
sys.path.insert(0, "src")

def euclidean(A,b):
    return np.sqrt(np.sum((A-b)**2,axis=1))

def test_euclidean_distance():
    A=np.array([[0,0],[3,4],[1,1]],float)
    d=euclidean(A,np.array([0,0],float))
    assert np.allclose(d,[0,5,np.sqrt(2)])

def test_knn_mean():
    y=np.array([2.,4.,6.,8.])
    nearest=np.array([0,1])
    assert np.isclose(y[nearest].mean(),3.)

def test_gaussian_kernel():
    d2=np.array([0.,1.,4.])
    w=np.exp(-d2/2)
    assert w[0] > w[1] > w[2]
