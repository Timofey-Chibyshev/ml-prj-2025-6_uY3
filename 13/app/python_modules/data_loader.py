import scipy.io
import numpy as np
from pyDOE import lhs

def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data

def get_values(data, N_u, N_r):
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact_values = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    
    X_setka = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    U_setka = Exact_values.flatten()[:,None]
    
    X_0 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    U_0 = Exact_values[0:1,:].T
    
    X_x_min = np.hstack((X[:,0:1], T[:,0:1]))
    X_x_max = np.hstack((X[:,-1:], T[:,-1:]))
    X_b = np.vstack([X_x_min,X_x_max])
    
    U_b_x_min = Exact_values[:,0:1]
    U_b_x_max = Exact_values[:,-1:]
    U_b = np.vstack([U_b_x_min,U_b_x_max])
    
    X_u_train = np.vstack([X_0,X_b])
    U_train = np.vstack([U_0,U_b])
    
    lb = X_setka.min(axis=0)
    ub = X_setka.max(axis=0)
    
    X_f_train = lb + (ub - lb)*lhs(2, N_r)
    
    idx = np.random.choice(X_u_train.shape[0], N_u)
    X_u_train = X_u_train[idx,:]
    U_train = U_train[idx,:]
    
    return x, t, X, T, Exact_values, X_setka, U_setka, X_u_train, U_train, X_f_train, ub, lb