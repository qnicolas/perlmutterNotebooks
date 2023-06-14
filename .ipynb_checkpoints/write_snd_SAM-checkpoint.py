import xarray as xr

stat_fixrad = xr.open_dataset("/pscratch/sd/q/qnicolas/SAMdata/OUT_STAT/RCE_128x128x64_fixrad_rce.nc")

tt = stat_fixrad.THETA[-240:].mean('time')
qq = stat_fixrad.QV[-240:].mean('time')
SAMdir = "/global/homes/q/qnicolas/SAM6.11.8/MOUNTAINWAVE/"
f = open(SAMdir+"snd", "w")
print(' z[m] p[mb] tp[K] q[g/kg] u[m/s] v[m/s]',file=f)
for day in (0.,1000.):
    print(' {:>4.1f},  64,{:>10.2f}   day,levels,pres0'.format(day,stat_fixrad.Ps[-1]),file=f)
    for i,z in enumerate(stat_fixrad.z):
        print('{:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.3f} {:>10.3f}'.format(z,stat_fixrad.p[i],tt[i],qq[i],0.,0.),file=f)
f.close()
