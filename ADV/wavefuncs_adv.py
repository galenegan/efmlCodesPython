"""Functions for calculating wave spectra and other wave statistics. Based
mostly on code written by Falk Feddersen and Justin Rogers, translated
into Python by Galen Egan. YW and phase method code written by Galen Egan"""

import copy
from mylib import naninterp
import numpy as np
import scipy.signal as sig
import datetime


def get_wavenumber(omega,h):
    
    #Returns wavenumber from the surface gravity wave dispersion relation
    #using Newton's method
    
    g = 9.81
    k = omega/np.sqrt(g*h)
    
    f = g*k*np.tanh(k*h) - omega**2
    
    while np.max(np.abs(f)) > 1e-10:
        dfdk = g*k*h*((1/np.cosh(k*h))**2) + g*np.tanh(k*h)
        k = k - f/dfdk
        f = g*k*np.tanh(k*h) - omega**2
        
    return k

def get_cg(k,depth):
    g=9.81
    Cp = np.sqrt((g/k)*np.tanh(k*depth))
    Cg = 0.5*Cp*(1+(k*depth)*(1-(np.tanh(k*depth))**2)/np.tanh(k*depth))
    
    return Cg

def calculate_fft2(x,nfft):
    
    X = copy.deepcopy(x)
    
    if X.ndim==1:
        n = np.size(X)
    elif X.ndim==2:
        n,m = np.shape(X)
        if m>n:
            X = X.T
            n,m = np.shape(X)
    
    num = int(np.floor(4*n/nfft) - 3)
    
    X = X - np.mean(X)
    
    jj = np.arange(0,nfft)
    
    WIN = np.hanning(np.size(jj))
    
    A = np.zeros((num,nfft),dtype = np.complex128)

    varXwindtot = 0    
    
    for ii in range(num):
        istart = int(np.ceil(ii*nfft/4))
        istop = int(np.floor(istart + nfft))  
        Xwind = X[istart:istop].squeeze()
        lenX = len(Xwind)
        Xwind = Xwind - np.mean(Xwind) #de-mean
        varXwind = np.dot(Xwind,Xwind)/lenX
        Xwind = sig.detrend(Xwind)
        
        varXwindtot = varXwindtot + varXwind
        Xwind = Xwind*WIN
        tmp = np.dot(Xwind,Xwind)/lenX
        
        if tmp == 0:
            Xwind = Xwind*0
        else:
            Xwind = Xwind*np.sqrt(varXwind/tmp)
        
        A[ii,:] = np.fft.fft(Xwind.T)/np.sqrt(nfft)
        
    
    return A
    
    

def wave_stats_spectra(u,v,p,nfft,doffu,doffp,fs,fc,rho,dirmethod):

    """Wave statistics and spectra"""
    
    U = copy.deepcopy(u)
    V = copy.deepcopy(v)
    P = copy.deepcopy(p)
    
    g = 9.81
    
    #Interpolating out nans
    P = naninterp(P) #pressure in dbar
    U = naninterp(U)
    V = naninterp(V)
    
    dbar = 1e4*np.nanmean(P)/(rho*g) + doffp#Average water depth
    
    depth = 1e4*P/(rho*g) + doffp
    
    #Making sure average depth is positive
    if dbar<0:
        raise ValueError
        
    fm,Suu = sig.welch(U,fs = fs,window = 'hamming',nperseg = nfft,detrend = 'linear')

    fm,Svv = sig.welch(V,fs = fs, window = 'hamming', nperseg = nfft,detrend = 'linear')
    
    fm,Suv = sig.csd(U,V,fs = fs, window = 'hamming', nperseg = nfft,detrend = 'linear')
    
    fm,Spp = sig.welch(P,fs = fs, window = 'hamming', nperseg = nfft,detrend = 'linear')
    
    fm,Spu = sig.csd(P,U,fs = fs, window = 'hamming', nperseg = nfft,detrend = 'linear')
    
    fm,Spv = sig.csd(P,V,fs = fs, window = 'hamming', nperseg = nfft,detrend = 'linear')
    
    df = fs/(nfft-1) #frequency resolution
    
    #Depth correction and spectral weighted averages
    f_ig_low = 1./250
    f_ig_high = 1./33.33
    f_swell_low = 1./25
    f_swell_high = 0.4
    
    i_ig = np.where((fm>f_ig_low) & (fm<f_ig_high))
    i_swell = np.where((fm>f_swell_low) & (fm<f_swell_high))
    i_all = np.where((fm>f_ig_low) & (fm<f_swell_high))
    
    omega = 2*np.pi*fm
    
    k = get_wavenumber(omega,dbar)
    
    correction = np.zeros((np.size(Spp),))
    convert = np.zeros((np.size(Spp),))
    
    ii = np.where(fm<=fc)
    
    correction[ii] = 1e4*np.cosh(k[ii]*dbar)/(rho*g*np.cosh(k[ii]*doffp))
    convert[ii] = (2*np.pi*fm[ii]/(g*k[ii]))*np.cosh(k[ii]*doffp)/(np.cosh(k[ii]*doffu))
    
    SSE = Spp*(correction**2)
    
    SSEt = SSE[ii]
    
    Suut = Suu[ii]
    Svvt = Svv[ii]
    Suvt = Suv[ii]

    fmt = fm[ii]
    
    UUpres = Suu*(convert**2) #converting to "equivalent pressure" for comparison with pressure
    VVpres = Svv*(convert**2)
    UVpres = Suv*(convert**2)
    
    PUpres = Spu*convert
    PVpres = Spv*convert
    
    
    #Cospectrum and quadrature
    coPUpres = np.real(PUpres)
    quPUpres = np.imag(PUpres)
    
    coPVpres = np.real(PVpres)
    quPVpres = np.imag(PVpres)
    
    coUVpres = np.real(UVpres)
    quUVpres = np.imag(UVpres)
    
    #coherence and phase
    cohPUpres = np.sqrt((coPUpres**2 + quPUpres**2)/(Spp*UUpres))
    phPUpres = (180/np.pi)*np.arctan2(quPUpres,coPUpres)
    cohPVpres = np.sqrt((coPVpres**2 + quPVpres**2)/(Spp*VVpres))
    phPVpres = (180/np.pi)*np.arctan2(quPVpres,coPVpres)
    cohUVpres = np.sqrt((coUVpres**2 + quUVpres**2)/(UUpres*VVpres))
    phUVpres = (180/np.pi)*np.arctan2(quUVpres,coUVpres)

    a1 = coPUpres/np.sqrt(Spp*(UUpres+VVpres))
    b1 = coPVpres/np.sqrt(Spp*(UUpres+VVpres))
    dir1 = np.degrees(np.arctan2(b1,a1))
    spread1 = np.degrees(np.sqrt(2*(1-(a1*np.cos(np.radians(dir1)) + b1*np.sin(np.radians(dir1))))))
    
    a2 = (UUpres - VVpres)/(UUpres + VVpres)
    b2 = 2*coUVpres/(UUpres+VVpres)
    
    dir2 = np.degrees(np.arctan2(b2,a2)/2)
    spread2 = np.degrees(np.sqrt(np.abs(0.5-0.5*(a2*np.cos(2*np.radians(dir2)) + b2*np.sin(2*np.radians(dir2))))))

    #Energy flux
    C = omega/k
    Cg = get_cg(k,dbar)    
    
    const = g*Cg*((np.cosh(k*dbar))**2)/((np.cosh(k*doffp))**2)
    
    
    #Energy flux by freq in cartesian coordinates
    posX = const*(0.5*(np.abs(Spp)+(UUpres-VVpres) + np.real(PUpres)))
    negX = const*(0.5*(np.abs(Spp)+(UUpres-VVpres) - np.real(PUpres)))
    posY = const*(0.5*(np.abs(Spp)+(VVpres-UUpres) + np.real(PVpres)))
    negY = const*(0.5*(np.abs(Spp)+(VVpres-UUpres) - np.real(PVpres)))
    
    posX2  = g*Cg*a1*SSE
    posY2 = g*Cg*b1*SSE
    
    Eflux = np.stack((posX2,posX,negX,posY2,posY,negY))
    
    Eflux_ss = np.nansum(Eflux[:,i_swell],axis = 1)*df
    Eflux_ig = np.nansum(Eflux[:,i_ig],axis = 1)*df
    
    #Significant wave height
    Hsigt = 4*np.sqrt(SSE[ii]*df)
    Hsig_ss = 4*np.sqrt(np.nansum(SSE[i_swell]*df))
    Hsig_ig = 4*np.sqrt(np.nansum(SSE[i_ig]*df))
    
    Hrmst = np.sqrt(8*SSE[ii]*df)
    Hrms_ss = np.sqrt(8*np.nansum(SSE[i_swell]*df))
    Hrms_ig = np.sqrt(8*np.nansum(SSE[i_ig]*df))
    Hrms_all = np.sqrt(8*np.nansum(SSE[i_all]*df))

    dirt = np.stack((dir1[ii],dir2[ii]))
    spreadt = np.stack((spread1[ii],spread2[ii]))
    
    dir_calc = dirt[dirmethod,:]

    a1t = a1[ii]
    a2t = a2[ii]
    b1t = b1[ii]
    b2t = b2[ii]
    
    a1_ss = np.nansum(a1[i_swell]*SSE[i_swell])/np.nansum(SSE[i_swell]) 
    b1_ss = np.nansum(b1[i_swell]*SSE[i_swell])/np.nansum(SSE[i_swell])
    a2_ss = np.nansum(a2[i_swell]*SSE[i_swell])/np.nansum(SSE[i_swell])
    b2_ss = np.nansum(b2[i_swell]*SSE[i_swell])/np.nansum(SSE[i_swell])
    
    dir_ss1 = np.degrees(np.arctan2(b1_ss,a1_ss))
    spread_ss1 = np.degrees(np.sqrt(2*(1-np.sqrt(a1_ss**2 + b1_ss**2))))
    
    dir_ss2 = np.degrees(np.arctan2(b2_ss,a2_ss))
    spread_ss2 = np.degrees(np.sqrt(2*(1-np.sqrt(a2_ss**2 + b2_ss**2))))
    
    #Centroid frequency
    fcentroid_swell = np.nansum(fm[i_swell]*SSE[i_swell])/np.nansum(SSE[i_swell])
    Tm_ss = 1./fcentroid_swell
    
    #peak frequency
    indx = np.argmax(SSE[i_swell])
    temp = fmt[i_swell]
    Tp_ss = 1./temp[indx]
    if np.size(Tp_ss) == 0:
        Tp_ss = np.nan
    
    #ig dir and spread
    a1_ig = np.nansum(a1[i_ig]*SSE[i_ig])/np.nansum(SSE[i_ig]) 
    b1_ig = np.nansum(b1[i_ig]*SSE[i_ig])/np.nansum(SSE[i_ig])
    a2_ig = np.nansum(a2[i_ig]*SSE[i_ig])/np.nansum(SSE[i_ig])
    b2_ig = np.nansum(b2[i_ig]*SSE[i_ig])/np.nansum(SSE[i_ig])
    
    dir_ig1 = np.degrees(np.arctan2(b1_ig,a1_ig))
    spread_ig1 = np.degrees(np.sqrt(2*(1-np.sqrt(a1_ig**2 + b1_ig**2))))
    
    dir_ig2 = np.degrees(np.arctan2(b2_ig,a2_ig))
    spread_ig2 = np.degrees(np.sqrt(2*(1-np.sqrt(a2_ig**2 + b2_ig**2))))
    
    #Centroid frequency
    fcentroid_ig = np.nansum(fm[i_ig]*SSE[i_ig])/np.nansum(SSE[i_ig])
    Tm_ig = 1./fcentroid_ig
    
    #peak frequency
    indx = np.argmax(SSE[i_ig])
    temp = fmt[i_ig]
    Tp_ig = 1./temp[indx]
    if np.size(Tp_ig) == 0:
        Tp_ig = np.nan
        
    #Centroid frequency
    fcentroid_all = np.nansum(fm[i_all]*SSE[i_all])/np.nansum(SSE[i_all])
    Tm_all = 1./fcentroid_all
    
    #All peak
    indx = np.argmax(SSE[i_all])
    temp = fmt[i_all]
    Tp_all = 1./temp[indx]
    if np.size(Tp_all) == 0:
        Tp_all = np.nan
    
    #Radiation stress
    Sxx = rho*g*((1.5+0.5*a2)*(Cg/C) - 0.5)*SSE
    Syy = rho*g*((1.5-0.5*a2)*(Cg/C) - 0.5)*SSE
    Sxy = rho*g*0.5*b2*(Cg/C)*SSE
    
    Sxx_ss = np.nansum(Sxx[i_swell])*df
    Syy_ss = np.nansum(Syy[i_swell])*df
    Sxy_ss = np.nansum(Sxy[i_swell])*df
    
    Cpu_ss = np.nansum(cohPUpres[i_swell]*SSE[i_swell])/np.nansum(SSE[i_swell])
    
    #Stokes drift
    kt = k[ii]
    omegat = omega[ii]
    
    Ust = (g*kt*Hrmst**2/(8*omegat*dbar))*np.cos(np.radians(dir_calc))
    Vst = (g*kt*Hrmst**2/(8*omegat*dbar))*np.sin(np.radians(dir_calc))
    
    Us = np.nansum(Ust)
    Vs = np.nansum(Vst)

    
    #output
    Wavestats = dict()
    Wavestats['SSEt'] = SSEt
    Wavestats['Suut'] = Suut
    Wavestats['Svvt'] = Svvt
    Wavestats['Suvt'] = Suvt
    Wavestats['fmt'] = fmt
    Wavestats['Tt'] = 1/fmt
    Wavestats['dirt'] = dirt
    Wavestats['spreadt'] = spreadt
    Wavestats['Hsigt'] = Hsigt
    Wavestats['Hsig_ss'] = Hsig_ss
    Wavestats['Hsig_ig'] = Hsig_ig
    Wavestats['Hrmst'] = Hrmst
    Wavestats['Hrms_ss'] = Hrms_ss
    Wavestats['Hrms_ig'] = Hrms_ig
    Wavestats['Hrms_all'] = Hrms_all
    Wavestats['Tm_ss'] = Tm_ss
    Wavestats['Tm_ig'] = Tm_ig
    Wavestats['Tm_all'] = Tm_all
    Wavestats['Tp_ss'] = Tp_ss
    Wavestats['Tp_ig'] = Tp_ig
    Wavestats['Tp_all'] = Tp_all
    Wavestats['Eflux_ss'] = Eflux_ss
    Wavestats['Eflux_ig'] = Eflux_ig
    Wavestats['dir_ss1'] = dir_ss1
    Wavestats['dir_ss2'] = dir_ss2
    Wavestats['dir_ig1'] = dir_ig1
    Wavestats['dir_ig2'] = dir_ig2
    Wavestats['spread_ss1'] = spread_ss1
    Wavestats['spread_ss2'] = spread_ss2
    Wavestats['spread_ig1'] = spread_ig1
    Wavestats['spread_ig2'] = spread_ig2
    Wavestats['Sxx_ss'] = Sxx_ss
    Wavestats['Sxy_ss'] = Sxy_ss
    Wavestats['Syy_ss'] = Syy_ss
    Wavestats['a1t'] = a1t
    Wavestats['a2t'] = a2t
    Wavestats['b1t'] = b1t
    Wavestats['b2t'] = b2t
    Wavestats['Cpu_ss'] = Cpu_ss
    Wavestats['depth'] = depth
    Wavestats['Us'] = Us
    Wavestats['Vs'] = Vs
    Wavestats['Ust'] = Ust
    Wavestats['Vst'] = Vst
    Wavestats['df'] = df

    return Wavestats

def yw(u,v,w,p,fs):

    """Wave-turbulence decomposition based on
    Young and Webster (2018)"""

    U = copy.deepcopy(u)
    V = copy.deepcopy(v)
    W = copy.deepcopy(w)
    P = copy.deepcopy(p)
    
    #Interpolating out nans
    P = naninterp(P) 
    U = naninterp(U)
    V = naninterp(V)
    W = naninterp(W)
    
    Pd = sig.detrend(P)
    Ud = sig.detrend(U)
    Vd = sig.detrend(V)
    Wd = sig.detrend(W)
    #Constructing matrix A
    M = len(U) 
    N = 101
    
    A = np.zeros((M,N))
    
    for m in range(M-N//2):
        A[m,:] = Pd[np.arange(m-(N-1)//2,m+N//2 + 1)]
    for m in range(M-N//2,M):
        A[m,:] = np.flipud(A[m-(M-N//2),:])
        
    hu = np.matmul(np.linalg.inv(np.matmul(A.T,A)),np.matmul(A.T,Ud))
    Uhat = np.matmul(A,hu)
    
    hv = np.matmul(np.linalg.inv(np.matmul(A.T,A)),np.matmul(A.T,Vd))
    Vhat = np.matmul(A,hv)
    
    hw = np.matmul(np.linalg.inv(np.matmul(A.T,A)),np.matmul(A.T,Wd))
    What = np.matmul(A,hw)
    
    dU = Ud - Uhat
    dV = Vd - Vhat
    dW = Wd - What
    
    yw = dict()
    yw['uu'] = np.nanmean(dU*dU)
    yw['vv'] = np.nanmean(dV*dV)
    yw['ww'] = np.nanmean(dW*dW)
    yw['uw'] = np.nanmean(dU*dW)
    yw['uv'] = np.nanmean(dU*dV)
    yw['vw'] = np.nanmean(dV*dW)
    
    yw['uu_wave'] = np.nanmean(Uhat*Uhat)
    yw['vv_wave'] = np.nanmean(Vhat*Vhat)
    yw['ww_wave'] = np.nanmean(What*What)
    yw['uw_wave'] = np.nanmean(Uhat*What)
    yw['uv_wave'] = np.nanmean(Uhat*Vhat)
    yw['vw_wave'] = np.nanmean(Vhat*What)

    return yw
    

def benilov(u,v,w,p,doffp,fs,fc,rho):

    """Benilov method wave turbulence decomposition"""
    
    U = copy.deepcopy(u)
    V = copy.deepcopy(v)
    W = copy.deepcopy(w)
    P = copy.deepcopy(p)
    
    g = 9.81
    
    #Interpolating out nans
    P = naninterp(P) 
    U = naninterp(U)
    V = naninterp(V)
    W = naninterp(W)
    
    dbar = 1e4*np.nanmean(P)/(rho*g) + doffp#Average water depth
    
    #Making sure average depth is positive
    if dbar<0:
        raise ValueError
    
    nfft = U.size

    
    #If using sig.welch/sig.csd
#    nfft = U.size//4
#    
#    fm,Suu = sig.welch(U,fs = fs,window = 'hamming',detrend = 'linear',nperseg = nfft)
#
#    fm,Svv = sig.welch(V,fs = fs, window = 'hamming',detrend = 'linear', nperseg = nfft)
#    
#    fm,Suv = sig.csd(U,V,fs = fs, window = 'hamming',detrend = 'linear', nperseg = nfft)
#    
#    fm,Sww = sig.welch(W,fs = fs,window = 'hamming',detrend = 'linear',nperseg = nfft)
#    
#    fm,Suw = sig.csd(U,W,fs = fs, window = 'hamming',detrend = 'linear', nperseg = nfft)
#    
#    fm,Svw = sig.csd(V,W,fs = fs, window = 'hamming', detrend = 'linear',nperseg = nfft)
#    
#    fm,Spp = sig.welch(P,fs = fs, window = 'hamming', detrend = 'linear',nperseg = nfft)
#    
#    fm,Spu = sig.csd(P,U,fs = fs, window = 'hamming', detrend = 'linear',nperseg = nfft)
#    
#    fm,Spv = sig.csd(P,V,fs = fs, window = 'hamming', detrend = 'linear',nperseg = nfft)
#    
#    fm,Spw = sig.csd(P,W,fs = fs, window = 'hamming', detrend = 'linear',nperseg = nfft)
    
    
    df = fs/(nfft-1)
    nnyq = int(np.floor(nfft/2 +1))
    fm = np.arange(0,nnyq)*df
#    df = np.abs(fm[1] - fm[0])

    #If using custom fft calculation function
    Amu = calculate_fft2(naninterp(U),nfft)
    Amv = calculate_fft2(naninterp(V),nfft)
    Amw = calculate_fft2(naninterp(W),nfft)
    Amp = calculate_fft2(naninterp(P),nfft)
    
    Suu = np.real(Amu*np.conj(Amu))/(nnyq*df)
    Suu = Suu.squeeze()[:nnyq]
    
    Svv = np.real(Amv*np.conj(Amv))/(nnyq*df)
    Svv = Svv.squeeze()[:nnyq]
    
    Sww = np.real(Amw*np.conj(Amw))/(nnyq*df)
    Sww = Sww.squeeze()[:nnyq]
    
    
    Suv = np.real(Amu*np.conj(Amv))/(nnyq*df)
    Suv = Suv.squeeze()[:nnyq]
    
    Suw = np.real(Amu*np.conj(Amw))/(nnyq*df)
    Suw = Suw.squeeze()[:nnyq]
    
    Svw = np.real(Amv*np.conj(Amw))/(nnyq*df)
    Svw = Svw.squeeze()[:nnyq]
    
    Spp = np.real(Amp*np.conj(Amp))/(nnyq*df)
    Spp = Spp.squeeze()[:nnyq]
    
    Spu = np.real(Amp*np.conj(Amu))/(nnyq*df)
    Spu = Spu.squeeze()[:nnyq]
    
    Spv = np.real(Amp*np.conj(Amv))/(nnyq*df)
    Spv = Spv.squeeze()[:nnyq]
    
    Spw = np.real(Amp*np.conj(Amw))/(nnyq*df)
    Spw = Spw.squeeze()[:nnyq]

    #Depth correction and spectral weighted averages
    
    f_swell_low = 1./25
    f_swell_high = .8
    i_swell = np.where((fm>f_swell_low) & (fm<f_swell_high))
    
    omega = 2*np.pi*fm

    k = get_wavenumber(omega,dbar)
    
    correction = np.zeros((np.size(Spp),))
    
    ii = np.where(fm<=fc)
    
    correction[ii] = 1e4*np.cosh(k[ii]*dbar)/(rho*g*np.cosh(k[ii]*doffp))
    
    See = Spp*(correction**2)
    Sue = Spu*correction
    Sve = Spv*correction
    Swe = Spw*correction
    
    Seet = See[ii]
    Suut = Suu[ii]
    Svvt = Svv[ii]
    Swwt = Sww[ii]
    Suvt = Suv[ii]
    Suwt = Suw[ii]
    Svwt = Svw[ii]
    Suet = Sue[ii]
    Svet = Sve[ii]
    Swet = Swe[ii]
    
    fmt = fm[ii]
    
    #Wave results
    #Auto
    S_uwave_uwave = Suet*np.conj(Suet)/Seet
    S_vwave_vwave = Svet*np.conj(Svet)/Seet
    S_wwave_wwave = Swet*np.conj(Swet)/Seet
    
    #Cross spectras
    S_uwave_wwave = Suet*np.conj(Swet)/Seet
    S_vwave_wwave = Svet*np.conj(Swet)/Seet
    S_uwave_vwave = Suet*np.conj(Svet)/Seet
    
    #Turbulence
    Supup = Suut - S_uwave_uwave
    Svpvp = Svvt - S_vwave_vwave
    Swpwp = Swwt - S_wwave_wwave
    
    Supwp = Suwt - S_uwave_wwave
    Svpwp = Svwt - S_vwave_wwave
    Supvp = Suvt - S_uwave_vwave
    
    #Integrating over spectra in wave band
    uwave_uwave_bar = np.nansum(np.real(S_uwave_uwave[i_swell])*df)
    vwave_vwave_bar = np.nansum(np.real(S_vwave_vwave[i_swell])*df)
    wwave_wwave_bar = np.nansum(np.real(S_wwave_wwave[i_swell])*df)
    
    uwave_wwave_bar = np.nansum(np.real(S_uwave_wwave[i_swell])*df)
    uwave_vwave_bar = np.nansum(np.real(S_uwave_vwave[i_swell])*df)
    vwave_wwave_bar = np.nansum(np.real(S_vwave_wwave[i_swell])*df)
    
    
    #Integrating turbulence over entire spectra
    upup_bar = 2*np.nansum(np.real(Supup)*df)
    vpvp_bar = 2*np.nansum(np.real(Svpvp)*df)
    wpwp_bar = 2*np.nansum(np.real(Swpwp)*df)
    
    upwp_bar = 2*np.nansum(np.real(Supwp)*df)
    vpwp_bar = 2*np.nansum(np.real(Svpwp)*df)
    upvp_bar = 2*np.nansum(np.real(Supvp)*df)
    
    #Outputting results
    B = dict()
    B['See'] = Seet
    B['Suu'] = Suut
    B['Svv'] = Svvt
    B['Sww'] = Swwt
    B['Suw'] = Suwt
    B['Svw'] = Svwt
    B['Suv'] = Suvt
    B['Sue'] = Suet
    B['Sve'] = Svet
    B['Swe'] = Swet
    
    B['S_uwave_uwave'] = S_uwave_uwave
    B['S_vwave_vwave'] = S_vwave_vwave
    B['S_wwave_wwave'] = S_wwave_wwave
    
    B['S_uwave_wwave'] = S_uwave_wwave
    B['S_vwave_wwave'] = S_vwave_wwave
    B['S_uwave_vwave'] = S_uwave_vwave
    
    #Wave tensor
    B['uwave_uwave_bar'] = uwave_uwave_bar
    B['vwave_vwave_bar'] = vwave_vwave_bar
    B['wwave_wwave_bar'] = wwave_wwave_bar
    
    B['uwave_vwave_bar'] = uwave_vwave_bar
    B['uwave_wwave_bar'] = uwave_wwave_bar
    B['vwave_wwave_bar'] = vwave_wwave_bar
    
    #Reynolds stress tensor
    B['upup_bar'] = upup_bar
    B['vpvp_bar'] = vpvp_bar
    B['wpwp_bar'] = wpwp_bar
    
    B['upwp_bar'] = upwp_bar
    B['vpwp_bar'] = vpwp_bar
    B['upvp_bar'] = upvp_bar
    
    B['fm'] = fmt
    B['df'] = df
    B['nfft'] = nfft
    B['fc'] = fc
    B['fs'] = fs
    
    return B

def bm_phase(adv,fs):

    """Wave turbulence decompsition based on Bricker & Monismith (2007)"""
    
    waveturb = dict()

    
    N = len(adv)
    
    #Turbulent reynolds stresses
    waveturb['uw'] = np.empty((N,))*np.NaN
    waveturb['vw'] = np.empty((N,))*np.NaN
    waveturb['uv'] = np.empty((N,))*np.NaN
    waveturb['uu'] = np.empty((N,))*np.NaN
    waveturb['vv'] = np.empty((N,))*np.NaN
    waveturb['ww'] = np.empty((N,))*np.NaN
    
    
    #Wave reynolds stresses
    waveturb['uw_wave'] = np.empty((N,))*np.NaN
    waveturb['vw_wave'] = np.empty((N,))*np.NaN
    waveturb['uv_wave'] = np.empty((N,))*np.NaN
    waveturb['uu_wave'] = np.empty((N,))*np.NaN
    waveturb['vv_wave'] = np.empty((N,))*np.NaN
    waveturb['ww_wave'] = np.empty((N,))*np.NaN
    
    waveturb['time'] = np.empty((N,),dtype = datetime.datetime)
        
    
    for ii in range(N):
        advidx = ii + list(adv.keys())[0]
       
        u = copy.deepcopy(adv[advidx]['velmaj'])
        v = copy.deepcopy(adv[advidx]['velmin'])
        w = copy.deepcopy(adv[advidx]['velz'])
        waveturb['time'][ii] = adv[advidx]['burststart']
    
        nfft = u.size
        Amu = calculate_fft2(naninterp(u),nfft)
        Amv = calculate_fft2(naninterp(v),nfft)
        Amw = calculate_fft2(naninterp(w),nfft)
        
    
        df = fs/(nfft-1)
        nnyq = int(np.floor(nfft/2 +1))
        fm = np.arange(0,nnyq)*df
       
    #    #Phase
        Uph = np.arctan2(np.imag(Amu),np.real(Amu)).squeeze()[:nnyq]
        Vph = np.arctan2(np.imag(Amv),np.real(Amv)).squeeze()[:nnyq]
        Wph = np.arctan2(np.imag(Amw),np.real(Amw)).squeeze()[:nnyq]
        
    #    #Computing the full spectra
    
        Suu = np.real(Amu*np.conj(Amu))/(nnyq*df)
        Suu = Suu.squeeze()[:nnyq]
        
        Svv = np.real(Amv*np.conj(Amv))/(nnyq*df)
        Svv = Svv.squeeze()[:nnyq]
        
        Sww = np.real(Amw*np.conj(Amw))/(nnyq*df)
        Sww = Sww.squeeze()[:nnyq]
        
        
        Suv = np.real(Amu*np.conj(Amv))/(nnyq*df)
        Suv = Suv.squeeze()[:nnyq]
        
        Suw = np.real(Amu*np.conj(Amw))/(nnyq*df)
        Suw = Suw.squeeze()[:nnyq]
        
        Svw = np.real(Amv*np.conj(Amw))/(nnyq*df)
        Svw = Svw.squeeze()[:nnyq]
        
        offset = np.sum(fm<=0.1)
                    
        uumax = np.argmax(Suu[(fm>0.1) & (fm < 0.7)]) + offset
        
        widthratiolow = 2.333
        widthratiohigh = 1.4
        fmmax = fm[uumax]
        waverange = np.arange(uumax - (fmmax/widthratiolow)//df,uumax + (fmmax/widthratiohigh)//df).astype(int)
        
        interprange = np.arange(1,np.nanargmin(np.abs(fm - 1))).astype(int)
        
        interprangeW = np.arange(1,np.nanargmin(np.abs(fm-1))).astype(int)
        
        interprange = interprange[(interprange>=0) & (interprange<nnyq)]
        waverange = waverange[(waverange>=0) & (waverange<nnyq)]
        interprangeW = interprangeW[(interprangeW >= 0) & (interprangeW < nnyq)]
        
        Suu_turb = Suu[interprange]
        fmuu = fm[interprange]
        Suu_turb = np.delete(Suu_turb,waverange-interprange[0])
        fmuu = np.delete(fmuu,waverange-interprange[0])
        Suu_turb = Suu_turb[fmuu>0]
        fmuu = fmuu[fmuu>0]
        
        Svv_turb = Svv[interprange]
        fmvv = fm[interprange]
        Svv_turb = np.delete(Svv_turb,waverange-interprange[0])
        fmvv = np.delete(fmvv,waverange-interprange[0])
        Svv_turb = Svv_turb[fmvv>0]
        fmvv = fmvv[fmvv>0]
        
        Sww_turb = Sww[interprangeW]
        fmww = fm[interprangeW]
        Sww_turb = np.delete(Sww_turb,waverange-interprangeW[0])
        fmww = np.delete(fmww,waverange-interprangeW[0])
        Sww_turb = Sww_turb[fmww>0]
        fmww = fmww[fmww>0]
        
        
        #Linear interpolation over turbulent spectra
        F = np.log(fmuu)
        S = np.log(Suu_turb)
        Puu = np.polyfit(F,S,deg = 1)
        Puuhat = np.exp(np.polyval(Puu,np.log(fm)))
        
        F = np.log(fmvv)
        S = np.log(Svv_turb)
        Pvv = np.polyfit(F,S,deg = 1)
        Pvvhat = np.exp(np.polyval(Pvv,np.log(fm)))
                                  
        F = np.log(fmww)
        S = np.log(Sww_turb)
        Pww = np.polyfit(F,S,deg = 1)
        Pwwhat = np.exp(np.polyval(Pww,np.log(fm)))
        
        #Wave spectra
        Suu_wave = Suu[waverange] - Puuhat[waverange]
        Svv_wave = Svv[waverange] - Pvvhat[waverange]
        Sww_wave = Sww[waverange] - Pwwhat[waverange]
        
        #Wave Fourier components
        Amu_wave = np.sqrt((Suu_wave+0j)*(df))
        Amv_wave = np.sqrt((Svv_wave+0j))*(df)
        Amww_wave = np.sqrt((Sww_wave+0j)*(df))
        
        #Wave Magnitudes
        Um_wave = np.sqrt(np.real(Amu_wave)**2 + np.imag(Amu_wave)**2)
        Vm_wave = np.sqrt(np.real(Amv_wave)**2 + np.imag(Amv_wave)**2)
        wm_wave = np.sqrt(np.real(Amww_wave)**2 + np.imag(Amww_wave)**2)
        
        #Wave reynolds stress
        uw_wave = np.nansum(Um_wave*wm_wave*np.cos(Wph[waverange]-Uph[waverange]))
        uv_wave =  np.nansum(Um_wave*Vm_wave*np.cos(Vph[waverange]-Uph[waverange]))
        vw_wave = np.nansum(Vm_wave*wm_wave*np.cos(Wph[waverange]-Vph[waverange]))
        
        uu_wave = np.nansum(Suu_wave*df)
        vv_wave = np.nansum(Svv_wave*df)
        ww_wave = np.nansum(Sww_wave*df)
        
                        
        #Full reynolds stresses
        uu = np.nansum(np.real(Suu)*df)
        uv = np.nansum(np.real(Suv)*df)
        uw = np.nansum(np.real(Suw)*df)
        vv = np.nansum(np.real(Svv)*df)
        vw = np.nansum(np.real(Svw)*df)
        ww = np.nansum(np.real(Sww)*df)
        
        #Turbulent reynolds stresses
        
        upup = uu - uu_wave
        vpvp = vv - vv_wave
        wpwp = ww - ww_wave
        upwp = uw - uw_wave
        upvp = uv - uv_wave
        vpwp = vw - vw_wave
        
        #Turbulent reynolds stresses
        waveturb['uw'][ii] = upwp
        waveturb['vw'][ii] = vpwp
        waveturb['uv'][ii] = upvp
        waveturb['uu'][ii] = upup
        waveturb['vv'][ii] = vpvp
        waveturb['ww'][ii] = wpwp
    
        
        #Wave reynolds stresses
        waveturb['uw_wave'][ii] = uw_wave
        waveturb['vw_wave'][ii] = vw_wave
        waveturb['uv_wave'][ii] = uv_wave
        waveturb['uu_wave'][ii] = uu_wave
        waveturb['vv_wave'][ii] = vv_wave
        waveturb['ww_wave'][ii] = ww_wave
                
    return waveturb
    
    
    
    
    
    
    
    