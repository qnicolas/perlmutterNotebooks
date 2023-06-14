import numpy as np
from scipy.integrate import odeint,solve_ivp
import sys
p = '/global/homes/q/qnicolas'
if p not in sys.path:
    sys.path.append(p)
from tools.generalTools import qsat

g = 9.81
Lv = 2.5e6
cp=1004.
Rd = 287.
Rv = 461.

Tm = 250.
Tt = 200.


def Qrad_Jpm3ps(T,p):
    rho = p/(Rd*T)
    cst = -rho*cp*1/86400 # 1 K/day
    return (T>=Tm)*cst + (T>Tt)*(T<Tm)*0.5*(1+np.cos(np.pi*(Tm-T)/(Tm-Tt)))*cst + (T<=Tt)*1e-6


def calc_Gamma_m(T,qs):
    return g/cp * (1+qs*Lv/(Rd*T))/(1+qs*Lv**2/(cp*Rv*T**2))

def calc_gamma_mWRONG(T,Gamma_m):
    """This is the Singh version. Pretty sure it contains a mistake"""
    return Lv/(Rv*T**2)*(Gamma_m-g/(Rd*T))

def calc_gamma_m(T,Gamma_m):
    return Lv/(Rv*T**2)*Gamma_m - g/(Rd*T)
    
def calc_b(T,qs):
    return Lv**2*qs/(cp*Rv*T**2+Lv**2*qs)

class ZBP():
    def __init__(self,zz,ww,Ts = 293., ps=920e2, epsilon = 0.6e-3, mu=1.5):
        self.zz = zz
        self.dz = zz[1]-zz[0]
        self.ww = ww
        self.Ts = Ts
        self.ps = ps
        self.epsilon=epsilon
        self.delta=epsilon
        self.mu=mu
        
    def idx(self,z):
        return int((z-self.zz[0])/self.dz+1e-3)
        
    def calc_RH(self,T,p,qs,Gamma_m,w):
        gamma_m = calc_gamma_m(T,Gamma_m)
        b = calc_b(T,qs)
        epsilon_hat = self.epsilon/gamma_m
        delta_hat = self.delta/gamma_m
        
        rho = p/(Rd*T)
        w_hat = Lv*qs*gamma_m/Qrad_Jpm3ps(T,p)*rho*w
        
        a3 = epsilon_hat*b*self.mu/(1+self.mu)*w_hat
        a2 = epsilon_hat*b/delta_hat/(1+self.mu) + (epsilon_hat*(1-b) - (1+2*epsilon_hat*b)*(self.mu/(1+self.mu)))*w_hat
        a1 = -(1 + (1+epsilon_hat*b)/(delta_hat*(1+self.mu))) + (epsilon_hat*(b-2) + (1+epsilon_hat*b)*(1+2*self.mu)/(1+self.mu))*w_hat
        a0 = 1 + (epsilon_hat*(1-b) - 1)*w_hat
    
        roots = np.roots(np.array([a3,a2,a1,a0]))
        crit = (np.real(roots)>=0)*(np.real(roots)<=1)*(np.abs(np.imag(roots))<1e-5)
        if np.sum(crit)==0:
            raise ValueError('issue in RH root, roots = ',roots)
    
        candidate_idx = np.argmax(crit)
        RH = np.real(roots[candidate_idx])
        if (RH < 0) or (RH>1):
            raise ValueError(RH)
        return RH
    
    def calc_Gamma(self,T,p,w):
        qs = qsat(T,p/100)
        Gamma_m = calc_Gamma_m(T,qs)
        RH = self.calc_RH(T,p,qs,Gamma_m,w)
        return Gamma_m + self.epsilon*Lv*qs*(1-RH)/(cp+qs*Lv**2/(Rv*T**2))
        
    def dTPdz(self,TP,z):
        w = np.interp(z,self.zz,self.ww)
        return np.array([-self.calc_Gamma(TP[0],TP[1],w),-TP[1]*g/Rd/TP[0]])

    def integrate(self):
        TP = odeint(self.dTPdz,np.array([self.Ts,self.ps]),self.zz)
        TT = TP[:,0]
        TT[TT<Tt] = Tt
        self.TT = TT
        self.pp = TP[:,1]
        self.qsat_ = qsat(self.TT,self.pp/100)
        self.Gamma_m_ = calc_Gamma_m(self.TT,self.qsat_)
        self.RH_ = np.array([self.calc_RH(self.TT[i],self.pp[i],self.qsat_[i],self.Gamma_m_[i],self.ww[i]) for i in range(len(self.zz))])
        self.qq = self.RH_*self.qsat_
        self.rho_ = self.pp/Rd/self.TT
    
    def calc_r(self,T,p,z,w):
        i=self.idx(z)
        qs = self.qsat_[i]
        Gamma_m = self.Gamma_m_[i]
        rho = self.rho_[i]
        RH = self.RH_[i]
        
        b = calc_b(T,qs)
        gamma_m = calc_gamma_m(T,Gamma_m)
        
        r = self.delta*(1+self.mu)*(1-RH)/(RH*(gamma_m+self.epsilon*b*(1-RH)))
        return r
    
    def calc_precip(self):
        if not hasattr(self,'TT'):
            raise ValueError('You need to call integrate() first.')
        r_ = np.array([self.calc_r(self.TT[i],self.pp[i],self.zz[i],self.ww[i]) for i in range(len(self.zz))])
        
        qsat_ = qsat(self.TT,self.pp/100)
        Gamma_m_ = calc_Gamma_m(self.TT,qsat_)
        RH_ = np.array([self.calc_RH(self.TT[i],self.pp[i],qsat_[i],Gamma_m_[i],self.ww[i]) for i in range(len(self.zz))])
        self.r_ = r_
        snet = -1/r_ * ( Qrad_Jpm3ps(self.TT,self.pp)/Lv - self.mu*self.delta*(self.pp/Rd/self.TT)*self.ww*(1-RH_)*qsat_)
        snet[self.TT<=Tt] = 0.
        
        self.snet = snet
        self.precip = np.trapz(self.snet,self.zz)
        
    def calc_Q(self):
        if not hasattr(self,'snet'):
            raise ValueError('You need to call calc_precip() first.')    
        self.Q = Lv*self.snet + Qrad_Jpm3ps(self.TT,self.pp)

    
class ZBPls(ZBP):
    """ZBP with a fixed large-scale forcing defined by alpha_q and alpha_s"""
    def __init__(self,zz,ww,alpha_s,alpha_q,Ts = 293., ps=920e2, epsilon = 0.6e-3, mu=1.5):
        super().__init__(zz,ww,Ts,ps,epsilon,mu)
        self.alpha_s = alpha_s
        self.alpha_q = alpha_q
            
    def calc_RH(self,T,p,qs,Gamma_m,w,a_s,a_q):
        gamma_m = calc_gamma_m(T,Gamma_m)
        b = calc_b(T,qs)
        epsilon_hat = self.epsilon/gamma_m
        delta_hat = self.delta/gamma_m
        
        rho = p/(Rd*T)
        
        w_hat = Lv*qs*gamma_m/Qrad_Jpm3ps(T,p)*rho*w
        a_s_hat = a_s/Qrad_Jpm3ps(T,p)
        a_q_hat = a_q/Qrad_Jpm3ps(T,p)
        
        a3 = epsilon_hat*b*self.mu/(1+self.mu)*w_hat
        a2 = epsilon_hat*b/delta_hat/(1+self.mu)*(1-a_s_hat) + (epsilon_hat*(1-b) - (1+2*epsilon_hat*b)*(self.mu/(1+self.mu)))*w_hat
        a1 = -(1 + (1+epsilon_hat*b)/(delta_hat*(1+self.mu)))*(1-a_s_hat) + (epsilon_hat*(b-2) + (1+epsilon_hat*b)*(1+2*self.mu)/(1+self.mu))*w_hat + a_q_hat*(epsilon_hat*(1-b)+self.mu*delta_hat)/delta_hat/(1+self.mu)
        a0 = 1-a_s_hat + (epsilon_hat*(1-b) - 1)*w_hat - a_q_hat*(epsilon_hat*(1-b)+self.mu*delta_hat-1)/delta_hat/(1+self.mu)
    
        roots = np.roots(np.array([a3,a2,a1,a0]))
        crit = (np.real(roots)>=0)*(np.real(roots)<=1)*(np.abs(np.imag(roots))<1e-5)
        if np.sum(crit)==0:
            raise ValueError('issue in RH root, roots = ',roots,'w=',w,'as,aq=',a_s,a_q)
    
        candidate_idx = np.argmax(crit)
        RH = np.real(roots[candidate_idx])
        if (RH < 0) or (RH>1):
            raise ValueError(RH)
        return RH       
    
    def calc_Gamma(self,T,p,w,a_s,a_q):
        qs = qsat(T,p/100)
        Gamma_m = calc_Gamma_m(T,qs)
        RH = self.calc_RH(T,p,qs,Gamma_m,w,a_s,a_q)
        return Gamma_m + self.epsilon*Lv*qs*(1-RH)/(cp+qs*Lv**2/(Rv*T**2))
        
    def dTPdz(self,TP,z):
        w = np.interp(z,self.zz,self.ww)
        a_s = np.interp(z,self.zz,self.alpha_s)
        a_q = np.interp(z,self.zz,self.alpha_q)
        return np.array([-self.calc_Gamma(TP[0],TP[1],w,a_s,a_q),-TP[1]*g/Rd/TP[0]])

    def integrate(self):
        TP = odeint(self.dTPdz,np.array([self.Ts,self.ps]),self.zz)
        TT = TP[:,0]
        TT[TT<Tt] = Tt
        self.TT = TT
        self.pp = TP[:,1]
        qsat_ = qsat(self.TT,self.pp/100)
        Gamma_m_ = calc_Gamma_m(self.TT,qsat_)
        self.RH_ = np.array([self.calc_RH(self.TT[i],self.pp[i],qsat_[i],Gamma_m_[i],self.ww[i],self.alpha_s[i],self.alpha_q[i]) for i in range(len(self.zz))])
        self.qq = self.RH_*qsat_
        self.rho_ = self.pp/Rd/self.TT
    
    def calc_r(self,T,p,w,a_s,a_q):
        if w==0.:
            return 1.
        qs = qsat(T,p/100)
        Gamma_m = calc_Gamma_m(T,qs)
        rho = p/Rd/T
        
        RH = self.calc_RH(T,p,qs,Gamma_m,w,a_s,a_q)  
        b = calc_b(T,qs)
        gamma_m = calc_gamma_m(T,Gamma_m)
        
        r = (self.delta*(1+self.mu)*(1-RH) - a_q/(rho*w*Lv*qs))/(RH*(gamma_m+self.epsilon*b*(1-RH)) - a_q/(rho*w*Lv*qs))
        return r
    
    def calc_precip(self):
        if not hasattr(self,'TT'):
            raise ValueError('You need to call integrate() first.')
        r_ = np.array([self.calc_r(self.TT[i],self.pp[i],self.ww[i],self.alpha_s[i],self.alpha_q[i]) for i in range(len(self.zz))])
        qsat_ = qsat(self.TT,self.pp/100)
        self.r_ = r_
        snet = -1/r_ * ( (Qrad_Jpm3ps(self.TT,self.pp) - self.alpha_s)/Lv - self.mu*self.delta*self.rho_*self.ww*(1-self.RH_)*qsat_)
        snet[self.TT<=Tt] = 0.
        
        self.snet = snet
        self.precip = np.trapz(self.snet,self.zz)

    def calc_Q(self):
        if not hasattr(self,'snet'):
            raise ValueError('You need to call calc_precip() first.')    
        self.Q = Lv*self.snet + Qrad_Jpm3ps(self.TT,self.pp) - self.alpha_s
    
    
    
# class ZBPls(ZBP):
#     """ZBP with a horizontal temperature and moisture gradients that depend on the temperature/moisture anomalies to a reference state. A bit ill-behaved."""
#     def __init__(self,zz,ww,alpha,TTref,qqref,Ts = 293., ps=920e2, epsilon = 0.6e-3, mu=1.5):
#         super().__init__(zz,ww,Ts,ps,epsilon,mu)
#         self.alpha = alpha
#         self.TTref = TTref
#         self.qqref = qqref        
#         self.RH_ = self.zz*np.nan
#             
#     def calc_RH(self,T,p,qs,Gamma_m,w,a,Tref,qref):
#         #if w<1e-4:
#         #    w=0.
#         gamma_m = calc_gamma_m(T,Gamma_m)
#         b = calc_b(T,qs)
#         epsilon_hat = self.epsilon/gamma_m
#         delta_hat = self.delta/gamma_m
#         
#         rho = p/(Rd*T)
#         
#         w_hat = Lv*qs*gamma_m/Qrad_Jpm3ps(T,p)*rho*w
#         a_s_hat = a*cp*(T-Tref)/Qrad_Jpm3ps(T,p)
#         a_q_hat = a*Lv*qref/Qrad_Jpm3ps(T,p)
#         qref_hat = qref/qs
#         
#         a3 = epsilon_hat*b*self.mu/(1+self.mu)*w_hat
#         a2 = epsilon_hat*b/delta_hat/(1+self.mu)*(1-a_s_hat) + (epsilon_hat*(1-b) - (1+2*epsilon_hat*b)*(self.mu/(1+self.mu)))*w_hat + a_q_hat*(epsilon_hat*(1-b)+self.mu*delta_hat)/(delta_hat*(1+self.mu)*qref_hat)
#         a1 = -(1 + (1+epsilon_hat*b)/(delta_hat*(1+self.mu)))*(1-a_s_hat) + (epsilon_hat*(b-2) + (1+epsilon_hat*b)*(1+2*self.mu)/(1+self.mu))*w_hat - a_q_hat*(epsilon_hat*(1-b)+self.mu*delta_hat)/delta_hat/(1+self.mu) - a_q_hat*(epsilon_hat*(1-b)+self.mu*delta_hat-1)/(delta_hat*(1+self.mu)*qref_hat)
#         a0 = 1-a_s_hat + (epsilon_hat*(1-b) - 1)*w_hat + a_q_hat*(epsilon_hat*(1-b)+self.mu*delta_hat-1)/delta_hat/(1+self.mu)
#     
#         roots=np.roots(np.array([a3,a2,a1,a0]))
#         crit = (np.real(roots)>=0)*(np.real(roots)<=1)*(np.abs(np.imag(roots))<1e-5)
#         #crit = (np.real(roots)>=0)*(np.abs(np.imag(roots))<1e-8)
#         if np.sum(crit)==0:
#             raise ValueError('issue in RH root, roots = ',roots,'w=',w,'a=',a)
#     
#         candidate_idx = np.argmax(crit)
#         RH = np.real(roots[candidate_idx])
#         return RH
#         #return np.minimum(1.,RH)
#     
#     def calc_Gamma(self,T,p,w,a,Tref,qref):
#         qs = qsat(T,p/100)
#         Gamma_m = calc_Gamma_m(T,qs)
#         RH = self.calc_RH(T,p,qs,Gamma_m,w,a,Tref,qref)
#         return Gamma_m + self.epsilon*Lv*qs*(1-RH)/(cp+qs*Lv**2/(Rv*T**2))
#         
#     def dTPdz(self,z,TP):
#         w = np.interp(z,self.zz,self.ww)
#         a = np.interp(z,self.zz,self.alpha)
#         Tref = np.interp(z,self.zz,self.TTref)
#         qref = np.interp(z,self.zz,self.qqref)
#         Gamma = self.calc_Gamma(TP[0],TP[1],w,a,Tref,qref)
#         return np.array([-Gamma,-TP[1]*g/Rd/TP[0]])
# 
#     def integrate(self):
#         res = solve_ivp(self.dTPdz,(self.zz[0],self.zz[-1]),np.array([self.Ts,self.ps]),method='RK45',t_eval=self.zz,rtol=1e-9,atol=1e-9)#,rtol=1e-8,atol=1e-8
#         #print(res.message)
#         TP = res.y.transpose()
#         #TP = odeint(self.dTPdz,np.array([self.Ts,self.ps]),self.zz,tfirst=True)
#         TT = TP[:,0]
#         TT[TT<Tt] = Tt
#         self.TT = TT
#         self.pp = TP[:,1]
#         self.qsat_ = qsat(self.TT,self.pp/100)
#         self.Gamma_m_ = calc_Gamma_m(self.TT,self.qsat_)
#         self.RH_ = np.array([self.calc_RH(self.TT[i],self.pp[i],self.qsat_[i],self.Gamma_m_[i],self.ww[i],self.alpha[i],self.TTref[i],self.qqref[i]) for i in range(len(self.zz))])
#         self.qq = self.RH_*self.qsat_
#         self.rho_ = self.pp/Rd/self.TT
#     
# #    def calc_r(self,T,p,z,w,a,Tref,qref):
# #        #if w<1e-4:
# #        #    w=0.
# #        i=self.idx(z)
# #        qs = self.qsat_[i]
# #        Gamma_m = self.Gamma_m_[i]
# #        rho = self.rho_[i]
# #        
# #        RH = self.RH_[i]
# #        b = calc_b(T,qs)
# #        gamma_m = calc_gamma_m(T,Gamma_m)
# #        
# #        r = (rho*w*Lv*self.delta*(1+self.mu)*(1-RH) + a*(qref/qs - RH) )/(rho*w*Lv*RH*(gamma_m+self.epsilon*b*(1-RH)) + a*(qref/qs - RH))
# #        return r
#     
# #    def calc_r2(self,T,p,z,w,a,Tref,qref):
# #        i=self.idx(z)
# #        qs = self.qsat_[i]
# #        Gamma_m = self.Gamma_m_[i]
# #        rho = self.rho_[i]
# #        RH = self.RH_[i]
# #        b = calc_b(T,qs)
# #        gamma_m = calc_gamma_m(T,Gamma_m)
# #        
# #        exp = ((self.epsilon*(1-b)+self.mu*self.delta)*(1-RH) - gamma_m)/((Qrad_Jpm3ps(T,p) - a*(cp*(T-Tref)))/(Lv*rho*w*qs) - self.mu*self.delta*(1-RH))
# #        
# #        r = ((rho*w*Lv)*self.delta*(1+self.mu)*(1-RH) + a*(qref/qs - RH))/((rho*w*Lv)*RH*(gamma_m+self.epsilon*b*(1-RH)) + a*(qref/qs - RH))
# #        return r
#     
# #    def calc_precip(self):
# #        if not hasattr(self,'TT'):
# #            raise ValueError('You need to call integrate() first.')
# #        r_ = np.array([self.calc_r(self.TT[i],self.pp[i],self.zz[i],self.ww[i],self.alpha[i],self.TTref[i],self.qqref[i]) for i in range(len(self.zz))])
# #
# #        self.r_ = r_
# #        snet = -1/r_ * ( (Qrad_Jpm3ps(self.TT,self.pp) - self.alpha*cp*(self.TT-self.TTref))/Lv - self.mu*self.delta*self.rho_*self.ww*(1-self.RH_)*self.qsat_)
# #        snet[self.TT<=Tt] = 0.
# #        
# #        self.snet = snet
# #        self.precip = np.trapz(self.snet,self.zz)
# 
#     def calc_precip(self):
#         if not hasattr(self,'TT'):
#             raise ValueError('You need to call integrate() first.')
#             
#         self.snet = (self.rho_*self.ww*(np.gradient(self.TT,self.zz)*cp+g ) - Qrad_Jpm3ps(self.TT,self.pp) + self.alpha*cp*(self.TT-self.TTref))/Lv
#         self.snet[self.TT<=Tt] = 0.
#         self.precip = np.trapz(self.snet,self.zz)