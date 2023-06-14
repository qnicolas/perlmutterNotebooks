import time
ti = time.time()

import numpy as np
import xarray as xr
import scipy.linalg as spl
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

import sys
sys.path.append('/global/homes/q/qnicolas')
from perlmutterNotebooks.calc_GW_LRF import g ,Lv,cp,Rd,T0,coupled_gw_mode,gw_mode_forced

##############################################################
######################### GW PARAMETERS ######################
##############################################################
U0=10
kSAM = 1e-4
hhatSAM = 50.

jobno = sys.argv[1]
#iteration_no = sys.argv[2]
print("Initializing Gravity wave ...")
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

################################################################
### LOAD SAM GRID INFO AND STABILITY/MOISTURE STRATIFICATION ###
################################################################
stat = xr.open_dataset("/pscratch/sd/q/qnicolas/SAMdata/OUT_STAT/RCE_128x128x64_ref_rce.nc")

samgrid = stat.z.data
zz_full = np.concatenate(([0.],samgrid,[27300,27500])) # increase top resolution for BC
n = len(zz_full)

Nsam = np.sqrt(g/T0*stat.DSE[-240:].mean('time').differentiate('z'))
Nsam[0] = Nsam[1]
Nsam[-2:] = Nsam[-3]
Nsam_full = np.concatenate(([Nsam[0]],Nsam,[Nsam[-1],Nsam[-1]]))

dqdzsam = stat.QV[-240:].mean('time').differentiate('z')/1e3
dqdzsam_full = np.concatenate(([dqdzsam[0]],dqdzsam,[dqdzsam[-1],dqdzsam[-1]]))

##############################################################
######################### CALCULATE GW #######################
##############################################################
#w0 = gw_mode_forced(zz_full,Nsam_full**2/U0**2,kSAM,hhatSAM,U0,0*zz_full)
w1,T1,q1,Qc1,Qq1 = coupled_gw_mode(zz_full,zrce,Nsam_full**2*T0/g,dqdzsam_full,kSAM,hhatSAM,U0,Mkuang2,coupling='full',itp_matrices=None)

##############################################################
################# STORE GW INFO IN NETCDF FILE ###############
##############################################################
filename = "/pscratch/sd/q/qnicolas/GWdata/GW_%s.nc"%jobno
gw_xr = xr.Dataset(data_vars=dict(
                       w_re=(["iteration","z"], np.real(w1).reshape((1,n)), {"units":"m/s"}),
                       w_im=(["iteration","z"], np.imag(w1.reshape((1,n))), {"units":"m/s"}),
                       Tprime_re=(["iteration","z"], np.real(T1).reshape((1,n)), {"units":"K"}),
                       Tprime_im=(["iteration","z"], np.imag(T1).reshape((1,n)), {"units":"K"}),
                       qprime_re=(["iteration","z"], np.real(q1/1e3).reshape((1,n)), {"units":"kg/kg"}),
                       qprime_im=(["iteration","z"], np.imag(q1/1e3).reshape((1,n)), {"units":"kg/kg"}),
                       Qc_re=(["iteration","z"], np.real(Qc1/86400).reshape((1,n)), {"units":"K/s"}),
                       Qc_im=(["iteration","z"], np.imag(Qc1/86400).reshape((1,n)), {"units":"K/s"}),
                       Qq_re=(["iteration","z"], np.real(Qq1/86400/1e3).reshape((1,n)), {"units":"kg/kg/s"}),
                       Qq_im=(["iteration","z"], np.imag(Qq1/86400/1e3).reshape((1,n)), {"units":"kg/kg/s"}),
                   ),
                   coords=dict(
                       iteration=(["iteration"], [0]),
                       z=(["z"], zz_full)
                   ),
                   attrs=dict(description="GW info, job n°%s"%jobno),
               )
#print(gw_xr)
gw_xr.to_netcdf(filename)

##############################################################
####### STORE TEMP AND QV TENDENCIES IN SAM lsf FILE #########
##############################################################
# 
# Qc = Qc1/86400
# Qq = Qq1/86400
# Qc[45:] = 0.
# Qq[45:] = 0.
# 
# tls = -(np.real(Qc))[1:-2]
# qls = -(np.real(Qq))[1:-2]/1000
# wls = 0.*tls
# SAMdir = "/global/homes/q/qnicolas/SAM6.11.8/MOUNTAINWAVE/"
# f = open(SAMdir+"lsf", "w")
# print(' z[m] p[mb] tls[K/s] qls[kg/kg/s] uls vls wls[m/s]',file=f)
# for day in (0.,1000.):
#     print(' {:>4.1f},  64,{:>10.2f}   day,levels,pres0'.format(day,stat.Ps[-1]),file=f)
#     for i,z in enumerate(stat.z):
#         print('{:>10.3f} {:>10.3f} {: 10e} {: 10e} {:>10.3f} {:>10.3f} {:>10.3f}'.format(z,stat.p[i],tls[i],qls[i],0.,0.,wls[i]),file=f)
# f.close()
# 
# tls = -(np.imag(Qc))[1:-2]
# qls = -(np.imag(Qq))[1:-2]/1000
# wls = 0.*tls
# SAMdir = "/global/homes/q/qnicolas/SAM6.11.8.im/MOUNTAINWAVE/"
# f = open(SAMdir+"lsf", "w")
# print(' z[m] p[mb] tls[K/s] qls[kg/kg/s] uls vls wls[m/s]',file=f)
# for day in (0.,1000.):
#     print(' {:>4.1f},  64,{:>10.2f}   day,levels,pres0'.format(day,stat.Ps[-1]),file=f)
#     for i,z in enumerate(stat.z):
#         print('{:>10.3f} {:>10.3f} {: 10e} {: 10e} {:>10.3f} {:>10.3f} {:>10.3f}'.format(z,stat.p[i],tls[i],qls[i],0.,0.,wls[i]),file=f)
# f.close()

print("Done initializing Gravity wave in %.1f s"%(time.time() - ti))