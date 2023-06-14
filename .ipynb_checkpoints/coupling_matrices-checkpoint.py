import numpy as np

import scipy.linalg as spl

import sys
p = '/global/homes/q/qnicolas'
if p not in sys.path:
    sys.path.append(p)
from perlmutterNotebooks.calc_GW_LRF import g ,Lv,cp,Rd,T0,make_interp_matrix,make_A_damped


def gaussianfilter_matrix(n):
    F = np.zeros((n,n))
    F[0,0] = 1; F[-1,-1] = 1
    F[1,:3] = [0.25,0.5,0.25]; F[-2,-3:] = [0.25,0.5,0.25]
    for i in range(2,n-2):
        F[i,i-2:i+3] = np.array([1.,4.,6.,4.,1.])/16.
    #return F
    return np.eye(n)


def rhsmatrix(z,zrce,k,U0,MM):
    dz = z[1]-z[0]
    n=len(z)
    
    itp1_matrix = np.block([[make_interp_matrix(z,zrce[:26]),np.zeros((26,len(z)))],[np.zeros((14,len(z))),make_interp_matrix(z,zrce[:14])]])
    itp2_matrix = make_interp_matrix(zrce[:26],z)
    itp3_matrix = np.block([[itp2_matrix,np.zeros((len(z),14))],[np.zeros((len(z),26)),make_interp_matrix(zrce[:14],z)]])

    zeroout_matrix = np.eye(n)
    zeroout_matrix[45:,45:] = 0.
    zeroout_matrix_2 = np.block([[zeroout_matrix,np.zeros((n,n))],[np.zeros((n,n)),zeroout_matrix]])
    gaussianfilter = np.block([[gaussianfilter_matrix(n),np.zeros((n,n))],[np.zeros((n,n)),gaussianfilter_matrix(n)]])
    
    MMinvitp = np.linalg.multi_dot((zeroout_matrix_2,gaussianfilter,itp3_matrix, spl.inv(MM/86400), itp1_matrix))
    
    zeroout2 = np.eye(40)
    zeroout2[20:26] = 0.
    MMinvitp = np.linalg.multi_dot((itp3_matrix,zeroout2, spl.inv(MM/86400), itp1_matrix))
    
    rhsmatrix = np.block([[np.zeros((n,n)),np.zeros((n,2*n))],
                          [np.zeros((2*n,n)),-1j*k*U0*MMinvitp]
                         ])
    
    rhsmatrix[0] = 0.
    rhsmatrix[len(z)-1] = 0.
    return rhsmatrix

def lhsmatrix_rhsvec(z,ds0dz,dq0dz,k,hhatk,U0):
    n=len(z)
    A = make_A(z,ds0dz*g/T0/U0**2,k)
    b = 1j*np.zeros(3*n)
    b[0] = 1j*k*U0*hhatk    
    
    I1 = np.eye(n)
    I1[0,0]=0.;I1[-1,-1]=0.
    
    zeroout_matrix = np.eye(n)
    zeroout_matrix[45:,45:] = 0.
    
    return np.block([[A,-g/T0/U0**2*I1,np.zeros((n,n))],[np.dot(zeroout_matrix,np.diag(ds0dz)),-np.eye(n),np.zeros((n,n))],[np.dot(zeroout_matrix,np.diag(dq0dz)*1000),np.zeros((n,n)),-np.eye(n)]]),b

################################################################################################################################################    
################################################################################################################################################
##########################################################                           ###########################################################
##########################################################   WITH RAYLEIGH DAMPING   ###########################################################
##########################################################                           ###########################################################
################################################################################################################################################
################################################################################################################################################

def rhsmatrix_damped(z,zrce,k,U0,eps,MM):
    dz = z[1]-z[0]
    n=len(z)
    
    itp1_matrix = np.block([[make_interp_matrix(z,zrce[:26]),np.zeros((26,len(z)))],[np.zeros((14,len(z))),make_interp_matrix(z,zrce[:14])]])
    itp2_matrix = make_interp_matrix(zrce[:26],z)
    itp3_matrix = np.block([[itp2_matrix,np.zeros((len(z),14))],[np.zeros((len(z),26)),make_interp_matrix(zrce[:14],z)]])

    zeroout_matrix = np.eye(n)
    zeroout_matrix[45:,45:] = 0.
    zeroout_matrix_2 = np.block([[zeroout_matrix,np.zeros((n,n))],[np.zeros((n,n)),zeroout_matrix]])
    gaussianfilter = np.block([[gaussianfilter_matrix(n),np.zeros((n,n))],[np.zeros((n,n)),gaussianfilter_matrix(n)]])
    
    MMinvitp = np.linalg.multi_dot((zeroout_matrix_2,gaussianfilter,itp3_matrix, spl.inv(MM/86400), itp1_matrix))
    
    #zeroout2 = np.eye(40)
    #zeroout2[20:26] = 0.
    #MMinvitp = np.linalg.multi_dot((itp3_matrix,zeroout2, spl.inv(MM/86400), itp1_matrix))
    
    rhsmatrix = np.block([[np.zeros((n,n)),np.zeros((n,2*n))],
                          [np.zeros((2*n,n)),-(1j*k*U0+eps)*MMinvitp]
                         ])
    
    rhsmatrix[0] = 0.
    rhsmatrix[len(z)-1] = 0.
    return rhsmatrix

def lhsmatrix_rhsvec_damped(z,ds0dz,dq0dz,k,hhatk,U0,eps):
    n=len(z)
    A = make_A_damped(z,ds0dz*g/T0/U0**2,k,U0,eps)
    b = 1j*np.zeros(3*n)
    b[0] = 1j*k*U0*hhatk    
    
    I1 = np.eye(n)
    I1[0,0]=0.;I1[-1,-1]=0.
    
    zeroout_matrix = np.eye(n)
    zeroout_matrix[45:,45:] = 0.
    
    return np.block([[A,- (1/(1-1j*eps/(k*U0)))**2 * g/T0/U0**2*I1,np.zeros((n,n))],[np.dot(zeroout_matrix,np.diag(ds0dz)),-np.eye(n),np.zeros((n,n))],[np.dot(zeroout_matrix,np.diag(dq0dz)*1000),np.zeros((n,n)),-np.eye(n)]]),b
