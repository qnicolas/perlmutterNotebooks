import time
ti = time.time()

import numpy as np
import xarray as xr
import scipy.linalg as spl
from scipy.io import loadmat

import sys
p = '/global/homes/q/qnicolas'
if p not in sys.path:
    sys.path.append(p)
from perlmutterNotebooks.calc_GW_LRF import g ,Lv,cp,Rd,T0,coupled_gw_mode_damped
from perlmutterNotebooks.coupling_matrices import lhsmatrix_rhsvec_damped,rhsmatrix_damped

##############################################################
######################### GW PARAMETERS ######################
##############################################################
U0=10
kSAM = 1e-7
hhatSAM = 50.
eps=3/86400
prefix = "MOUNTAINWAVE_128x128x64_k1e-4_h50_U10"


jobno = sys.argv[1]
ncfilename = "/pscratch/sd/q/qnicolas/GWdata/GW_%s.nc"%jobno

mode = sys.argv[2]

if mode=='init':
    print("Initializing Gravity wave ...")
elif mode=='update':
    print("Updating Gravity wave ...")
    iteration_no = int(sys.argv[3])
else:
    raise ValueError("Second argument is either 'init' or 'update'")
    
print("Parameters: U0 = %.1f, k = %.3e, hhat = %.3f"%(U0,kSAM,hhatSAM))

###############################################################
###### LOAD Z. KUANG LRF AND REVERSE POSITIVE EIGENVALUE ######
###############################################################
kuangdata = loadmat("/global/u2/q/qnicolas/orographicPrecipitation/steadyLRF.mat")
Mkuang=kuangdata['M']
zrce = kuangdata['z'][:,0]
#Reverse positive eigenvalue
lambdas,P = spl.eig(Mkuang)
lambdas[5]*=-1
Pm1 = spl.inv(P)
Mkuang2 = np.real(np.dot(np.dot(P,np.diag(lambdas)),Pm1))

M_coupling = Mkuang2

################################################################
### LOAD SAM GRID INFO AND STABILITY/MOISTURE STRATIFICATION ###
################################################################
stat = xr.open_dataset("/pscratch/sd/q/qnicolas/SAMdata/OUT_STAT/RCE_128x128x64_fixrad_rce.nc")

samgrid = stat.z.data
zz_full = np.concatenate(([0.],samgrid,[27300,27500])) # increase top resolution for BC
n = len(zz_full)

Nsam = np.sqrt(np.abs(g/T0*stat.DSE[-240:].mean('time').differentiate('z')))
Nsam[0] = Nsam[1]
Nsam[-2:] = Nsam[-3]
Nsam_full = np.concatenate(([Nsam[0]],Nsam,[Nsam[-1],Nsam[-1]]))

dqdzsam = stat.QV[-240:].mean('time').differentiate('z')/1e3
dqdzsam_full = np.concatenate(([dqdzsam[0]],dqdzsam,[dqdzsam[-1],dqdzsam[-1]]))




if mode == 'init':
    ##############################################################
    ######################### CALCULATE GW #######################
    ##############################################################
    #w0 = gw_mode_forced(zz_full,Nsam_full**2/U0**2,kSAM,hhatSAM,U0,0*zz_full)
    w1,T1,q1,Qc1,Qq1 = coupled_gw_mode_damped(zz_full,zrce,Nsam_full**2*T0/g,dqdzsam_full,kSAM,hhatSAM,U0,eps,M_coupling,coupling='full',itp_matrices=None)

    ##############################################################
    ################# STORE GW INFO IN NETCDF FILE ###############
    ##############################################################
    
    gw_xr = xr.Dataset(data_vars=dict(
                           w_re=(["iteration","z"], np.real(w1).reshape((1,n)), {"units":"m/s"}),
                           w_im=(["iteration","z"], np.imag(w1.reshape((1,n))), {"units":"m/s"}),
                           Tprime_re=(["iteration","z"], np.real(T1).reshape((1,n)), {"units":"K"}),
                           Tprime_im=(["iteration","z"], np.imag(T1).reshape((1,n)), {"units":"K"}),
                           qprime_re=(["iteration","z"], np.real(q1).reshape((1,n)), {"units":"g/kg"}),
                           qprime_im=(["iteration","z"], np.imag(q1).reshape((1,n)), {"units":"g/kg"}),
                           Qc_re=(["iteration","z"], np.real(Qc1).reshape((1,n)), {"units":"K/s"}),
                           Qc_im=(["iteration","z"], np.imag(Qc1).reshape((1,n)), {"units":"K/s"}),
                           Qq_re=(["iteration","z"], np.real(Qq1).reshape((1,n)), {"units":"g/kg/s"}),
                           Qq_im=(["iteration","z"], np.imag(Qq1).reshape((1,n)), {"units":"g/kg/s"}),
                       ),
                       coords=dict(
                           iteration=(["iteration"], [0]),
                           z=(["z"], zz_full)
                       ),
                       attrs=dict(description="GW info, job n°%s"%jobno),
                   )
    #print(gw_xr)
    gw_xr.to_netcdf(ncfilename,mode='w')
    
    
    
    
    
elif mode == 'update':
    gw_xr = xr.open_dataset(ncfilename)
    ##############################################################
    ######################### CALCULATE GW #######################
    ##############################################################
    # First, read T,q data from the previous iteration
    prev_iter_re = xr.open_dataset("/pscratch/sd/q/qnicolas/SAMdata/OUT_STAT/"+prefix+"_real_it%i.nc"%(iteration_no-1))
    prev_iter_im = xr.open_dataset("/pscratch/sd/q/qnicolas/SAMimdata/OUT_STAT/"+prefix+"_imag_it%i.nc"%(iteration_no-1))
    
    Tprime_i_re = prev_iter_re.TABS[-24:].mean('time')-stat.TABS[-240:].mean('time')
    qprime_i_re = prev_iter_re.QV[-24:].mean('time')-stat.QV[-240:].mean('time')
    Tprime_i_im = prev_iter_im.TABS[-24:].mean('time')-stat.TABS[-240:].mean('time')
    qprime_i_im = prev_iter_im.QV[-24:].mean('time')-stat.QV[-240:].mean('time')
    
    Tprime_i = Tprime_i_re.data+1j*Tprime_i_im.data
    qprime_i = qprime_i_re.data+1j*qprime_i_im.data
    Tprime_i = np.concatenate(([0.],Tprime_i,[0.,0.]))
    qprime_i = np.concatenate(([0.],qprime_i,[0.,0.]))

    ## Test mode
    #Tprime_i_re = gw_xr.Tprime_re.isel(iteration=-1).data
    #qprime_i_re = gw_xr.qprime_re.isel(iteration=-1).data
    #Tprime_i_im = gw_xr.Tprime_im.isel(iteration=-1).data
    #qprime_i_im = gw_xr.qprime_im.isel(iteration=-1).data
    #Tprime_i = Tprime_i_re+1j*Tprime_i_im
    #qprime_i = qprime_i_re+1j*qprime_i_im
    
    zeroout_matrix = np.eye(n)
    zeroout_matrix[45:,45:] = 0.
    
    zeroTq_i = np.concatenate([np.zeros(n),np.dot(zeroout_matrix,Tprime_i),np.dot(zeroout_matrix,qprime_i)])

    AA,bb = lhsmatrix_rhsvec_damped(zz_full,Nsam_full**2*T0/g,dqdzsam_full,kSAM,hhatSAM,U0,eps)
    BB = rhsmatrix_damped(zz_full,zrce,kSAM,U0,eps,M_coupling)    
    
    wQ_i = np.concatenate([(gw_xr.w_re+1j*gw_xr.w_im).isel(iteration=-1),
                           (gw_xr.Qc_re+1j*gw_xr.Qc_im).isel(iteration=-1),
                           (gw_xr.Qq_re+1j*gw_xr.Qq_im).isel(iteration=-1)])
    wQ_ip1 = spl.solve(AA-BB, -(1j*kSAM*U0+eps)*zeroTq_i -np.dot(BB,wQ_i) + bb)
    #wQ_ip1 = spl.solve(AA, -(1j*kSAM*U0+eps)*zeroTq_i + bb)

    w1 = wQ_ip1[:n]
    Qc1 = wQ_ip1[n:2*n]
    Qq1 = wQ_ip1[2*n:3*n]
    
    ##############################################################
    ################# STORE GW INFO IN NETCDF FILE ###############
    ##############################################################
    
    gw_xr2 = xr.Dataset(data_vars=dict(
                           w_re=(["iteration","z"], np.real(w1).reshape((1,n)), {"units":"m/s"}),
                           w_im=(["iteration","z"], np.imag(w1.reshape((1,n))), {"units":"m/s"}),
                           Tprime_re=(["iteration","z"], np.real(Tprime_i).reshape((1,n)), {"units":"K"}),
                           Tprime_im=(["iteration","z"], np.imag(Tprime_i).reshape((1,n)), {"units":"K"}),
                           qprime_re=(["iteration","z"], np.real(qprime_i).reshape((1,n)), {"units":"g/kg"}),
                           qprime_im=(["iteration","z"], np.imag(qprime_i).reshape((1,n)), {"units":"g/kg"}),
                           Qc_re=(["iteration","z"], np.real(Qc1).reshape((1,n)), {"units":"K/s"}),
                           Qc_im=(["iteration","z"], np.imag(Qc1).reshape((1,n)), {"units":"K/s"}),
                           Qq_re=(["iteration","z"], np.real(Qq1).reshape((1,n)), {"units":"g/kg/s"}),
                           Qq_im=(["iteration","z"], np.imag(Qq1).reshape((1,n)), {"units":"g/kg/s"}),
                       ),
                       coords=dict(
                           iteration=(["iteration"], [iteration_no]),
                           z=(["z"], zz_full)
                       ),
                       attrs=dict(description="GW info, job n°%s"%jobno),
                   )
    
    gw_xr_new = xr.concat((gw_xr,gw_xr2),dim="iteration")
    gw_xr.close()
    gw_xr_new.to_netcdf(ncfilename)
    
###############################################################
######## STORE TEMP AND QV TENDENCIES IN SAM lsf FILE #########
###############################################################


Qc1[45:] = 0.
Qq1[45:] = 0.

tls = -(np.real(Qc1))[1:-2]
qls = -(np.real(Qq1))[1:-2]/1000
wls = 0.*tls
SAMdir = "/global/homes/q/qnicolas/SAM6.11.8/MOUNTAINWAVE/"
f = open(SAMdir+"lsf", "w")
print(' z[m] p[mb] tls[K/s] qls[kg/kg/s] uls vls wls[m/s]',file=f)
for day in (0.,1000.):
    print(' {:>4.1f},  64,{:>10.2f}   day,levels,pres0'.format(day,stat.Ps[-1]),file=f)
    for i,z in enumerate(stat.z):
        print('{:>10.3f} {:>10.3f} {: 10e} {: 10e} {:>10.3f} {:>10.3f} {:>10.3f}'.format(z,stat.p[i],tls[i],qls[i],0.,0.,wls[i]),file=f)
f.close()

tls = -(np.imag(Qc1))[1:-2]
qls = -(np.imag(Qq1))[1:-2]/1000
wls = 0.*tls
SAMdir = "/global/homes/q/qnicolas/SAM6.11.8.im/MOUNTAINWAVE/"
f = open(SAMdir+"lsf", "w")
print(' z[m] p[mb] tls[K/s] qls[kg/kg/s] uls vls wls[m/s]',file=f)
for day in (0.,1000.):
    print(' {:>4.1f},  64,{:>10.2f}   day,levels,pres0'.format(day,stat.Ps[-1]),file=f)
    for i,z in enumerate(stat.z):
        print('{:>10.3f} {:>10.3f} {: 10e} {: 10e} {:>10.3f} {:>10.3f} {:>10.3f}'.format(z,stat.p[i],tls[i],qls[i],0.,0.,wls[i]),file=f)
f.close()

if mode == 'init':
    print("Done initializing Gravity wave in %.1f s"%(time.time() - ti))
elif mode == 'update':
    print("Done updating Gravity wave in %.1f s"%(time.time() - ti))