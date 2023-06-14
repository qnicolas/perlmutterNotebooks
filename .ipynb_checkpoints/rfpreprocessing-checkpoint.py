import numpy as np
import pandas as pd
import xarray as xr

import sys
p = '/global/homes/q/qnicolas'
if p not in sys.path:
    sys.path.append(p)
from tools.wrfHrTools import *

def to_ar(var):
    n = len(var.pressure)
    return var.transpose('pressure','Time','south_north','distance_from_mtn').data.reshape((n,-1))
def make_df(Sim,dist1,dist2,variables='thetae',t1 = 3000,dt = 24*12,p1=24,p2=24):
    t2 = t1+dt
    plevs = Sim.datapl.P_PL[0].load()
    if variables == 'tq':
        t = change_coords_pl(Sim.datapl,Sim.datapl.T_PL)[t1:t2,:p1,:].sel(distance_from_mtn=slice(dist1,dist2)).load()
        q = change_coords_pl(Sim.datapl,Sim.datapl.Q_PL)[t1:t2,:p2,:].sel(distance_from_mtn=slice(dist1,dist2)).load()
        t=t.where(t>-1)
        q=q.where(q>-1)
        d1 = {i:"T%i"%(plevs[i]/100) for i in range(p1)} 
        d2 = {i+p1:"q%i"%(plevs[i]/100) for i in range(p2)}
    elif variables == 'thetae':
        t = xr.open_zarr(Sim.path+'wrf.THETAE.1h.1970010100-1970072000.zarr').THETAE[t1:t2,:p1,:].sel(distance_from_mtn=slice(dist1,dist2)).load()
        q = xr.open_zarr(Sim.path+'wrf.THETAESTAR.1h.1970010100-1970072000.zarr').THETAESTAR[t1:t2,:p2,:].sel(distance_from_mtn=slice(dist1,dist2)).load() 
        d1 = {i:"the%i"%(plevs[i]/100) for i in range(p1)} 
        d2 = {i+p1:"thes%i"%(plevs[i]/100) for i in range(p2)}
        
    featurenames = {**d1,**d2}
    rainnc  = xr.open_zarr(Sim.path+'wrf.SFCVARS.1h.1970010100-1970072000.zarr').RAINNC_MMDY[t1:t2].sel(distance_from_mtn=slice(dist1,dist2)).load()
    
    #convert to np arrays, remove nans
    features_ar = np.concatenate((to_ar(t),to_ar(q)),axis=0).T
    outputs_ar = rainnc.data.reshape(-1)
    idx = ~np.isnan(features_ar[:,0])
    features_ar = features_ar[idx]
    outputs_ar  = outputs_ar [idx]
    
    #convert to dataframes
    
    outputs_df = pd.DataFrame(outputs_ar)
    outputs_df = outputs_df.rename(columns={0:'precip'})
    features_df = pd.DataFrame(features_ar).rename(columns=featurenames)#[:300000]
    
    #sample light and heavy precip equally
    breakpoints = [0.]+list(10**np.linspace(-1,4,301))
    samples_per_bin=3000
    
    all_df = pd.concat([features_df,outputs_df],axis=1)
    all_df['bins'] = pd.cut(outputs_df['precip'],breakpoints,right=False,labels=breakpoints[:-1])
    
    all_df_sampled = all_df.groupby('bins').apply(lambda x: x.sample(min(samples_per_bin,len(x))).reset_index(drop=True))
    all_df_sampled = all_df_sampled.drop('bins',axis=1).reset_index().drop('bins',axis=1).drop('level_1',axis=1)
    
    return all_df_sampled

def get_blpr(Sim,dist1,dist2,t1 = 3000,dt = 24*12,bltype='DEEP'):
    t2 = t1+dt
    BL  = xr.open_zarr(Sim.path+'wrf.BL_%s.1h.1970010100-1970072000.zarr'%bltype).B_L[t1:t2].sel(distance_from_mtn=slice(dist1,dist2)).load()
    rainnc  = xr.open_zarr(Sim.path+'wrf.SFCVARS.1h.1970010100-1970072000.zarr').RAINNC_MMDY[t1:t2].sel(distance_from_mtn=slice(dist1,dist2)).load()
    
    return BL.data.reshape(-1),rainnc.data.reshape(-1)

def get_binned(a,b,bins):
    bin_centers=(bins[1:]+bins[:-1])/2
    bin_sums = np.histogram(a,bins,weights=b)[0]
    counts = np.histogram(a,bins,density=False)[0]
    return bin_centers,(bin_sums/counts)


def plot_1to1(pred_true_list,log=True):
    plt.figure(figsize=(8,8))
    if log:
        bins_pr = 10**np.linspace(0,4,101)
        plt.xscale('log');plt.yscale('log')
    else:
        bins_pr = np.linspace(0,1000,201)
    
    for (pred,true,lbl) in pred_true_list:
        predbc,true_means = get_binned(pred,true,bins_pr)
        plt.plot(true_means,predbc,label=lbl)
        
    plt.plot([1,5e2],[1,5e2],color='k',linestyle=':')
    plt.legend()       
    plt.xlabel('WRF precip');plt.ylabel('prediction (binned)')