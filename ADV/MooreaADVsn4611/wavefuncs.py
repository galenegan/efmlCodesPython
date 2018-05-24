#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:47:07 2018

@author: gegan
"""

"""Functions for calculating wave spectra and other wave statistics. Based
mostly on code written by Falk Feddersen and Justin Rogers"""

import copy
from mylib import naninterp
import numpy as np
import scipy.signal


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
        Xwind = X[istart:istop]
        lenX = len(Xwind)
        Xwind = Xwind - np.mean(Xwind) #de-mean
        varXwind = np.matmul(Xwind.T,Xwind)/lenX
        Xwind = scipy.signal.detrend(Xwind)
        
        varXwindtot = varXwindtot + varXwind
        Xwind = Xwind*WIN
        tmp = np.matmul(Xwind.T,Xwind)/lenX
        
        if tmp == 0:
            Xwind = Xwind*0
        else:
            Xwind = Xwind*np.sqrt(varXwind/tmp)
        
        A[ii,:] = np.fft.fft(Xwind.T)/np.sqrt(nfft)
        
    
    return A
    
    

def wave_stats_spectra(u,v,p,nfft,doffu,doffp,fs,fc,rho,dirmethod):
    
    U = copy.deepcopy(u)
    V = copy.deepcopy(v)
    P = copy.deepcopy(p)
    
    g = 9.81
    
#    #Make sure everything is column vectors
#    if np.size(U,axis=0) < np.size(U,axis = 1):
#        U = U.T
#        
#    if np.size(V,axis=0) < np.size(V,axis = 1):
#        V = V.T
#    
#    if np.size(P,axis=0) < np.size(P,axis = 1):
#        P = P.T
    
    #Interpolating out nans
    P = naninterp(P)*1e-4 #pressure in dbar
    U = naninterp(U)
    V = naninterp(V)
    
    depth = np.nanmean(P)/(rho*g) #Average water depth
    
    #Making sure average depth is positive
    if depth<0:
        raise ValueError
    
    Amu = calculate_fft2(U,nfft)
    Amv = calculate_fft2(V,nfft)
    Amp = calculate_fft2(P,nfft)
    
    nA,mA = np.shape(Amu)
    
    df = fs/(nfft-1) #frequency resolution
    nnyq = int(nfft/2 + 1)
    num_avg = nA
    
    fm = np.arange(0,nnyq)*df
    
    Suu = np.real(np.nanmean(Amu*np.conj(Amu),axis=0))/(nnyq*df)
    Suu = Suu[0:nnyq]
    
    Svv = np.real(np.nanmean(Amv*np.conj(Amv),axis=0))/(nnyq*df)
    Svv = Svv[0:nnyq]
    
    Suv = np.nanmean(Amu*np.conj(Amv),axis=0)/(nnyq*df)
    Suv = Suv[0:nnyq]
    
    Spp = np.real(np.nanmean(Amp*np.conj(Amp),axis=0))/(nnyq*df)
    Spp = Spp[0:nnyq]
    
    Spu = np.nanmean(Amp*np.conj(Amu),axis=0)/(nnyq*df)
    Spu = Spu[0:nnyq]
    
    Spv = np.nanmean(Amp*np.conj(Amv),axis=0)/(nnyq*df)
    Spv = Spv[0:nnyq]
    
    #Depth correction and spectral weighted averages
    f_ig_low = 1./250
    f_ig_high = 1./33.33
    f_swell_low = 1./25
    f_swell_high = 1./5
    
    i_ig = np.where((fm>f_ig_low) & (fm<f_ig_high))
    i_swell = np.where((fm>f_swell_low) & (fm<f_swell_high))
    i_all = np.where((fm>f_ig_low) & (fm<f_swell_high))
    
    omega = 2*np.pi*fm
    
    k = get_wavenumber(omega,depth)
    
    
    if doffp<0:
        dzp = copy.deepcopy(doffp)
        doffp = 0
        G = np.exp(-np.abs(k)*dzp)
        ind = np.where(fm>0.5)
        G[ind] = 1
    else:
        G  = 1
    
    Spp = Spp*G*G
    Spu = Spu*G
    Spv = Spv*G
    
    correction = np.zeros((nnyq,))
    convert = np.zeros((nnyq,))
    
    ii = np.where(fm<=fc)
    
    correction[ii] = 1e4*np.cosh(k[ii]*depth)/(rho*g*np.cosh(k[ii]*doffp))
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
    
    SppU = np.sqrt(UUpres**2 + VVpres**2)
    SSEU = SppU*(correction**2)
    
    Ztest = Spp[i_swell]/SppU[i_swell]
    ztest_ss = np.nansum((Spp[i_swell]/SppU[i_swell])*SSE[i_swell])/np.nansum(SSE[i_swell])
    
    
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
    Cg = get_cg(k,depth)    
    
    const = g*Cg*((np.cosh(k*depth))**2)/((np.cosh(k*doffp))**2)
    
    
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
    
    Ust = (g*kt*Hrmst**2/(8*omegat*depth))*np.cos(np.radians(dir_calc))
    Vst = (g*kt*Hrmst**2/(8*omegat*depth))*np.sin(np.radians(dir_calc))
    
    Us = np.nansum(Ust)
    Vs = np.nansum(Vst)
    
    #DOF and level of no significant coherence
    merge = 1
    DOF = 2*nA*merge
    SIG = np.sqrt(6/DOF)
    
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
    Wavestats['ztest_ss'] = ztest_ss
    Wavestats['Cpu_ss'] = Cpu_ss
    Wavestats['depth'] = depth
    Wavestats['Us'] = Us
    Wavestats['Vs'] = Vs
    Wavestats['Ust'] = Ust
    Wavestats['Vst'] = Vst
    Wavestats['df'] = df
    Wavestats['num_avg'] = num_avg

    return Wavestats

def benilov(u,v,w,p,nfft,doffp,fs,fc,rho):
    
    U = copy.deepcopy(u)
    V = copy.deepcopy(v)
    W = copy.deepcopy(w)
    P = copy.deepcopy(p)
    
    g = 9.81
    
#    #Make sure everything is column vectors
#    if np.size(U,axis=0) < np.size(U,axis = 1):
#        U = U.T
#        
#    if np.size(V,axis=0) < np.size(V,axis = 1):
#        V = V.T
#    
#    if np.size(P,axis=0) < np.size(P,axis = 1):
#        P = P.T
    
    #Interpolating out nans
    P = naninterp(P)*1e-4 #Convert to dbar
    U = naninterp(U)
    V = naninterp(V)
    W = naninterp(W)
    
    depth = np.nanmean(P)*1e4/(rho*g) + doffp#Average water depth
    
    #Making sure average depth is positive
    if depth<0:
        raise ValueError
    
    Amu = calculate_fft2(U,nfft)
    Amv = calculate_fft2(V,nfft)
    Amw = calculate_fft2(W,nfft)
    Amp = calculate_fft2(P,nfft)
    
    nA,mA = np.shape(Amu)
    
    df = fs/(nfft-1) #frequency resolution
    nnyq = int(nfft/2 + 1)
    num_avg = nA
    
    fm = np.arange(0,nnyq)*df
    
    Suu = np.real(np.nanmean(Amu*np.conj(Amu),axis=0))/(nnyq*df)
    Suu = Suu[0:nnyq]
    
    Svv = np.real(np.nanmean(Amv*np.conj(Amv),axis=0))/(nnyq*df)
    Svv = Svv[0:nnyq]
    
    Sww = np.real(np.nanmean(Amw*np.conj(Amw),axis=0))/(nnyq*df)
    
    Suv = np.nanmean(Amu*np.conj(Amv),axis=0)/(nnyq*df)
    Suv = Suv[0:nnyq]
    
    Suw = np.nanmean(Amu*np.conj(Amw),axis=0)/(nnyq*df)
    Suw = Suw[0:nnyq]
    
    Svw = np.nanmean(Amv*np.conj(Amw),axis=0)/(nnyq*df)
    Svw = Svw[0:nnyq]
    
    Spp = np.real(np.nanmean(Amp*np.conj(Amp),axis=0))/(nnyq*df)
    Spp = Spp[0:nnyq]
    
    Spu = np.nanmean(Amp*np.conj(Amu),axis=0)/(nnyq*df)
    Spu = Spu[0:nnyq]
    
    Spv = np.nanmean(Amp*np.conj(Amv),axis=0)/(nnyq*df)
    Spv = Spv[0:nnyq]
    
    Spw = np.nanmean(Amp*np.conj(Amw),axis=0)/(nnyq*df)
    Spw = Spw[0:nnyq]
    
    #Depth correction and spectral weighted averages
    
    f_swell_low = 1./25
    f_swell_high = 1./5
    i_swell = np.where((fm>f_swell_low) & (fm<f_swell_high))
    
    omega = 2*np.pi*fm

    
    k = get_wavenumber(omega,depth)
    
    correction = np.zeros((nnyq,))
    
    ii = np.where(fm<=fc)
    
    correction[ii] = 1e4*np.cosh(k[ii]*depth)/(rho*g*np.cosh(k[ii]*doffp))
    
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
    B['depth'] = depth
    B['fs'] = fs
    B['num_avg'] = num_avg
    
    return B
    
    
    
    
    
    
    
    