import numpy as np

import sys
p = '/global/homes/q/qnicolas'
if p not in sys.path:
    sys.path.append(p)

from orographicPrecipitation.observationsAndERA5.linearWaveModels import second_derivative_matrix

g = 9.81
Lv = 2.5e6
cp=1004.
Rd = 287.
T0=300.

def make_A(z,lz2,k):
    n = len(z)
    dz = z[1]-z[0]
    
    sgnk = np.sign(k)
    if k==0:
        sgnk=1
    if lz2[-1] < k**2:
        mtop = 1j*np.sqrt(k**2-lz2[-1])
    else:
        mtop = sgnk*np.sqrt(lz2[-1]-k**2)
    
    D2 = second_derivative_matrix(n,1) #Matrix of second differentiation
    A = D2 + dz**2*np.diag(lz2-k**2)
    A = A.astype('complex')
    
    A[0]   = np.zeros(n)
    A[0,0] = 1
    A[-1]  = np.zeros(n)
    #A[-1,-1] = 1;A[-1,-2] = -1
    A[-1,-3:] = np.array([1,-4,3])/2
    A[-1,-1] -=  dz*1j* mtop    
    
    return A

def gw_mode_forced(z,lz2,k,hhatk,U0,qhatk):
    """Computes one wave mode by solving the linear wave equation:
    d2/dz2(w_hat) + (l(z)^2-k^2)w_hat = q_hat, subject to BCs
    w_hat(k,z=0) = ikU(z=0)h_hat(k) 
    & d w_hat(k,ztop) = i m(ztop) w_hat(k,ztop), where m(ztop) is defined to satisfy a radiation BC or an evanescent BC at the top
    """
    n = len(z)
    dz = z[1]-z[0]
    
    A = make_A(z,lz2,k)
    
    b = 1j*np.zeros(n)
    b[0] = 1j*k*U0*hhatk
    b[1:-1]= dz**2 * qhatk[1:-1]
    
    return np.linalg.solve(A,b)

def nmodes_forced(z,mui,wi,mi,hhatk,U0,qhatk):
    n = len(z)
    dz = z[1]-z[0]

    b = 1j*np.zeros(n)
    b[0] = 1j*k*U0*hhatk
    b[1:-1]= dz**2 * qhatk[1:-1]
    
    coefs_true = np.dot(np.conj(mi),b)/mui
    return np.dot(coefs_true,wi)

def make_interp_matrix(zorig,zdest):
    return np.array([[np.interp(zdest[j],zorig,np.eye(len(zorig))[i],left=0.,right=0.) for i in range(len(zorig))] for j in range(len(zdest))])
def coupled_gw_mode(z,zrce,ds0dz,dq0dz,k,hhatk,U0,MM,coupling='full',itp_matrices=None):
    """Computes one wave mode by solving the linear wave equation:
    d2/dz2(w_hat) + (l(z)^2-k^2)w_hat = q_hat, subject to BCs
    w_hat(k,z=0) = ikU(z=0)h_hat(k) 
    & d w_hat(k,ztop) = i m(ztop) w_hat(k,ztop), where m(ztop) is defined to satisfy a radiation BC or an evanescent BC at the top
    q_hat is coupled here.
    """
    n = len(z)
    dz = z[1]-z[0]
    
    A = make_A(z,ds0dz*g/T0/U0**2,k)
    
    b = 1j*np.zeros(n)
    b[0] = 1j*k*U0*hhatk
    
    strat_matrix = np.vstack((np.diag(ds0dz),np.diag(dq0dz)*1000))
    
    if itp_matrices is None:
        itp1_matrix = np.block([[make_interp_matrix(z,zrce[:26]),np.zeros((26,len(z)))],[np.zeros((14,len(z))),make_interp_matrix(z,zrce[:14])]])
        itp2_matrix = make_interp_matrix(zrce[:26],z)
        itp3_matrix = np.block([[itp2_matrix,np.zeros((len(z),14))],[np.zeros((len(z),26)),make_interp_matrix(zrce[:14],z)]])
    else: 
        itp1_matrix,itp2_matrix,itp3_matrix = itp_matrices
    
    #if coupling=='full':
    #    rhs_matrix = g/T0/U0**2 * np.linalg.multi_dot((itp2_matrix,MM[:26]/86400,np.linalg.inv(MM/86400-1j*k*U0*np.eye(40)),itp1_matrix,strat_matrix))
    #    A[1:-1] -= dz**2*rhs_matrix[1:-1]
    #    ww = np.linalg.solve(A,b)
    #    Tq = np.linalg.multi_dot((itp3_matrix,np.linalg.inv(MM/86400-1j*k*U0*np.eye(40)),itp1_matrix,strat_matrix,ww))
    #    QcQq = np.linalg.multi_dot((itp3_matrix,MM,itp1_matrix,Tq))
    #    return ww,Tq[:len(z)],Tq[len(z):]/1e3,QcQq[:len(z)],QcQq[len(z):]
    if coupling=='full':
        MMitp = np.linalg.multi_dot((itp3_matrix,MM,itp1_matrix))
        rhs_matrix = g/T0/U0**2 * np.linalg.multi_dot((MMitp[:len(z)]/86400,np.linalg.inv(MMitp/86400-1j*k*U0*np.eye(2*len(z))),strat_matrix))
        A[1:-1] -= dz**2*rhs_matrix[1:-1]
        ww = np.linalg.solve(A,b)
        Tq = np.linalg.multi_dot((np.linalg.inv(MMitp/86400-1j*k*U0*np.eye(2*len(z))),strat_matrix,ww))
        QcQq = np.dot(MMitp,Tq)
        return ww,Tq[:len(z)],Tq[len(z):]/1e3,QcQq[:len(z)],QcQq[len(z):]
    elif coupling=='noq':
        MMitp = np.linalg.multi_dot((itp3_matrix,MM,itp1_matrix))
        rhs_matrix = g/T0/U0**2 * np.linalg.multi_dot((MMitp[:len(z)]/86400,np.linalg.inv(np.vstack((MMitp[:len(z)],np.zeros((len(z),2*len(z)))))/86400-1j*k*U0*np.eye(2*len(z))),strat_matrix))
        A[1:-1] -= dz**2*rhs_matrix[1:-1]
        ww = np.linalg.solve(A,b)
        Tq = np.linalg.multi_dot((np.linalg.inv(np.vstack((MMitp[:len(z)],np.zeros((len(z),2*len(z)))))/86400-1j*k*U0*np.eye(2*len(z))),strat_matrix,ww))
        QcQq = np.dot(MMitp,Tq)
        return ww,Tq[:len(z)],Tq[len(z):]/1e3,QcQq[:len(z)],QcQq[len(z):]        
    else:
        ww = np.linalg.solve(A,b)        
        Tq = -1/(1j*k*U0)*np.dot(strat_matrix,ww)
        QcQq = np.linalg.multi_dot((itp3_matrix,MM,itp1_matrix,Tq))
        b[1:-1] += dz**2*g/T0/U0**2*QcQq[:len(z)][1:-1]/86400
        ww = np.linalg.solve(A,b)
        Tq = np.linalg.multi_dot((itp3_matrix,np.linalg.inv(MM/86400-1j*k*U0*np.eye(40)),itp1_matrix,strat_matrix,ww))
        return ww,Tq[:len(z)],Tq[len(z):]/1e3,QcQq[:len(z)],QcQq[len(z):]