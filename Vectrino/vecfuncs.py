# -*- coding: utf-8 -*-
"""
Created on Sun May 13 16:41:08 2018

@author: gegan
"""

#Packages
import os
parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

import sys
sys.path.append(parentDir + '/General/')

import copy
from mylib import naninterp
import numpy as np
import scipy.optimize
import scipy.io as sio
import scipy.ndimage
import scipy.signal as sig
from autocorr import autocorr


def xyz_enu(vectrino_kk,heading,pitch,roll):
    
    """XYZ velocity data to ENU coordinates"""
    
    CH = np.cos(np.radians(heading))
    SH = np.sin(np.radians(heading))
    CP = np.cos(np.radians(pitch))
    SP = np.sin(np.radians(pitch))
    CR = np.cos(np.radians(roll))
    SR = np.sin(np.radians(roll))
    
    H = np.array(([CH,SH,0],[-SH,CH, 0],[0, 0, 1]))
    P = np.array(([1,0,0],[0, CP, -SP],[0,SP,CP]))
    R = np.array(([CR,0,SR],[0,1,0],[-SR,0,CR]))
    
    m1 = np.matmul(P,R)
    M = np.matmul(H,m1)
    

    for jj in range(np.shape(vectrino_kk['u'])[0]):
        Veltemp1 = np.stack((vectrino_kk['u'][jj,:],vectrino_kk['v'][jj,:],
                             vectrino_kk['w1'][jj,:]))
        Veltemp2 = np.stack((vectrino_kk['u'][jj,:],vectrino_kk['v'][jj,:],
                             vectrino_kk['w2'][jj,:]))
        Velnew1 = np.matmul(M,Veltemp1)
        Velnew2 = np.matmul(M,Veltemp2)
        vectrino_kk['u'][jj,:] = Velnew1[0,:]
        vectrino_kk['v'][jj,:] = Velnew1[1,:]
        vectrino_kk['w1'][jj,:] = Velnew1[2,:]
        vectrino_kk['w2'][jj,:] = Velnew2[2,:]

    return vectrino_kk


def lpf(arr_in,fs,fc):
    
    """Fourier low-pass filter"""
    
    arr = copy.deepcopy(arr_in)
    
    M,N = np.shape(arr)
    
    for ii in range(N):
        nanidx = np.isnan(arr[:,ii])
        if np.sum(nanidx) < np.size(arr[:,ii]) - 1:
            arr[:,ii] = naninterp(arr[:,ii])
        else: 
            arr[:,ii] = np.zeros(np.shape(arr[:,ii]))
    
    #Windowing function (in this case, no window but could add later)
    H = 1
    
    F = (1./N)*np.fft.fftshift(np.fft.fft2(arr*H))
    
    if N%2 == 1:
        s = np.arange((-(N-1)/2)*fs/N,((N)/2)*fs/N,fs/N)
    else:
        s = np.arange((-N/2)*fs/N,((N+1)/2 - 1)*fs/N,fs/N)
    
    S = np.ones((M,np.size(s)))
    for ii in range(M):
        S[ii,:] = S[ii,:]*s
    
    cutidx = np.where(np.abs(S)>fc)
    
    filt = np.ones((M,np.size(s)))
    filt[cutidx] = 0
    
    F_filt = F*filt
    arr_filt = (1./H)*np.real(np.fft.ifft2(np.fft.ifftshift(F_filt*N)))
    
    #arr_filt[nanidx] = np.nan #This puts original nans back in..probably not necessary 
    
    return arr_filt

def pa_theta(u,v):
    
    """Finding angle of maximum variance"""
    
    #Storing as complex variable w = u + iv
    w = u + 1j*v
    w = w[np.isfinite(w)]
    
    #Covariance matrix
    cv = np.cov(np.stack((np.real(w),np.imag(w))))
    
    #Theta is direction of maximum variance
    theta = (180./np.pi)*(0.5*np.arctan2(2.*cv[1,0],(cv[0,0]-cv[1,1])))
    
    return theta

def pa_rotation(u,v,theta):
    
    """Rotating onto principle axes"""
    
    #Storing as complex variable w = u + iv
    w = u + 1j*v
    
    wr = w*np.exp(-1j*theta*np.pi/180)
    vel_maj = np.real(wr)
    vel_min = np.imag(wr)
    
    return vel_maj,vel_min

def calculate_fft(x,nfft):
    
    """Custom fft"""
    
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
        Xwind = scipy.signal.detrend(Xwind)
        
        varXwindtot = varXwindtot + varXwind
        Xwind = Xwind*WIN
        tmp = np.dot(Xwind,Xwind)/lenX
        
        if tmp == 0:
            Xwind = Xwind*0
        else:
            Xwind = Xwind*np.sqrt(varXwind/tmp)
        
        A[ii,:] = np.fft.fft(Xwind.T)/np.sqrt(nfft)
        
    
    return A

def get_turb_waves(vectrino_kk,fs):
    
    
    """Bricker and Monismith (2007) wave-turbulence decomposition"""
    
    waveturb = dict()
 
    u = copy.deepcopy(vectrino_kk['velmaj'])
    v = copy.deepcopy(vectrino_kk['velmin'])
    w1 = copy.deepcopy(vectrino_kk['w1'])
    w2 = copy.deepcopy(vectrino_kk['w2'])
    
    m,n = np.shape(u)
    
    waveturb = dict()  
    
    #Turbulent reynolds stresses
    waveturb['uw1'] = np.empty((m,))*np.NaN
    waveturb['vw1'] = np.empty((m,))*np.NaN
    waveturb['uw2'] = np.empty((m,))*np.NaN
    waveturb['vw2'] = np.empty((m,))*np.NaN
    waveturb['uv'] = np.empty((m,))*np.NaN
    waveturb['uu'] = np.empty((m,))*np.NaN
    waveturb['vv'] = np.empty((m,))*np.NaN
    waveturb['w1w1'] = np.empty((m,))*np.NaN
    waveturb['w2w2'] = np.empty((m,))*np.NaN
    waveturb['w1w2'] = np.empty((m,))*np.NaN
    
    
    #Wave reynolds stresses
    waveturb['uw1_wave'] = np.empty((m,))*np.NaN
    waveturb['vw1_wave'] = np.empty((m,))*np.NaN
    waveturb['uw2_wave'] = np.empty((m,))*np.NaN
    waveturb['vw2_wave'] = np.empty((m,))*np.NaN
    waveturb['uv_wave'] = np.empty((m,))*np.NaN
    waveturb['uu_wave'] = np.empty((m,))*np.NaN
    waveturb['vv_wave'] = np.empty((m,))*np.NaN
    waveturb['w1w1_wave'] = np.empty((m,))*np.NaN
    waveturb['w2w2_wave'] = np.empty((m,))*np.NaN
    waveturb['w1w2_wave'] = np.empty((m,))*np.NaN
    
    for jj in range(vectrino_kk['z'].size):
        
        if np.sum(np.isnan(u[jj,:])) < np.size(u[jj,:]/2):
            
            #Amplitude
            nfft = u[jj,:].size
            Amu = calculate_fft(naninterp(u[jj,:]),nfft)
            Amv = calculate_fft(naninterp(v[jj,:]),nfft)
            Amw1 = calculate_fft(naninterp(w1[jj,:]),nfft)
            Amw2 = calculate_fft(naninterp(w2[jj,:]),nfft)
            
            df = fs/(nfft-1)
            nnyq = int(np.floor(nfft/2 +1))
            fm = np.arange(0,nnyq)*df
   
            #Phase
            Uph = np.arctan2(np.imag(Amu),np.real(Amu)).squeeze()[:nnyq]
            Vph = np.arctan2(np.imag(Amv),np.real(Amv)).squeeze()[:nnyq]
            W1ph = np.arctan2(np.imag(Amw1),np.real(Amw1)).squeeze()[:nnyq]
            W2ph = np.arctan2(np.imag(Amw2),np.real(Amw2)).squeeze()[:nnyq]
            
            #Computing the full spectra
        
            Suu = np.real(Amu*np.conj(Amu))/(nnyq*df)
            Suu = Suu.squeeze()[:nnyq]
            
            Svv = np.real(Amv*np.conj(Amv))/(nnyq*df)
            Svv = Svv.squeeze()[:nnyq]
            
            Sww1 = np.real(Amw1*np.conj(Amw1))/(nnyq*df)
            Sww1 = Sww1.squeeze()[:nnyq]
            
            Sww2 = np.real(Amw2*np.conj(Amw2))/(nnyq*df)
            Sww2 = Sww2.squeeze()[:nnyq]
            
            Suv = np.real(Amu*np.conj(Amv))/(nnyq*df)
            Suv = Suv.squeeze()[:nnyq]
            
            Suw1 = np.real(Amu*np.conj(Amw1))/(nnyq*df)
            Suw1 = Suw1.squeeze()[:nnyq]
            
            Suw2 = np.real(Amu*np.conj(Amw2))/(nnyq*df)
            Suw2 = Suw2.squeeze()[:nnyq]
            
            Svw1 = np.real(Amv*np.conj(Amw1))/(nnyq*df)
            Svw1 = Svw1.squeeze()[:nnyq]
            
            Svw2 = np.real(Amv*np.conj(Amw2))/(nnyq*df)
            Svw2 = Svw2.squeeze()[:nnyq]
            
            Sw1w2 = np.real(Amw1*np.conj(Amw2))/(nnyq*df)
            Sw1w2 = Sw1w2.squeeze()[:nnyq]
            
            #Make sure wave peak is in a reasonable range of frequencies
            offset = np.sum(fm<=0.1)
            uumax = np.argmax(Suu[(fm>0.1) & (fm < 0.7)]) + offset
            
            #These should be adjusted based on width of wave peak in frequency space
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
            
            Sww1_turb = Sww1[interprangeW]
            fmww1 = fm[interprangeW]
            Sww1_turb = np.delete(Sww1_turb,waverange-interprangeW[0])
            fmww1 = np.delete(fmww1,waverange-interprangeW[0])
            Sww1_turb = Sww1_turb[fmww1>0]
            fmww1 = fmww1[fmww1>0]
            
            Sww2_turb = Sww2[interprangeW]
            fmww2 = fm[interprangeW]
            Sww2_turb = np.delete(Sww2_turb,waverange-interprangeW[0])
            fmww2 = np.delete(fmww2,waverange-interprangeW[0])
            Sww2_turb = Sww2_turb[fmww2>0]
            fmww2 = fmww2[fmww2>0]
            
            #Linear interpolation over turbulent spectra
            F = np.log(fmuu)
            S = np.log(Suu_turb)
            Puu = np.polyfit(F,S,deg = 1)
            Puuhat = np.exp(np.polyval(Puu,np.log(fm)))
            
            F = np.log(fmvv)
            S = np.log(Svv_turb)
            Pvv = np.polyfit(F,S,deg = 1)
            Pvvhat = np.exp(np.polyval(Pvv,np.log(fm)))
                                      
            F = np.log(fmww1)
            S = np.log(Sww1_turb)
            Pww1 = np.polyfit(F,S,deg = 1)
            Pww1hat = np.exp(np.polyval(Pww1,np.log(fm)))
            
            F = np.log(fmww2)
            S = np.log(Sww2_turb)
            Pww2 = np.polyfit(F,S,deg = 1)
            Pww2hat = np.exp(np.polyval(Pww2,np.log(fm)))
        
            
            #Wave spectra
            Suu_wave = Suu[waverange] - Puuhat[waverange]
            Svv_wave = Svv[waverange] - Pvvhat[waverange]
            Sww1_wave = Sww1[waverange] - Pww1hat[waverange]
            Sww2_wave = Sww2[waverange] - Pww2hat[waverange]
            
            
#            #Plotting to test the code
#            plt.figure()
#            plt.loglog(fmuu,Suu_turb,'k*')
#            plt.loglog(fm[waverange],Suu[waverange],'r-')
#            plt.loglog(fm,Puuhat,'b-')
#            plt.title('Suu')
#            
#            plt.figure()
#            plt.loglog(fmww1,Sww1_turb,'k*')
#            plt.loglog(fm[waverange],Sww1[waverange],'r-')
#            plt.loglog(fm,Pww1hat,'b-')
#            plt.title('Sww')
        
            #Fourier coefficients 
            Amu_wave = np.sqrt((Suu_wave+0j)*(df))
            Amv_wave = np.sqrt((Svv_wave+0j))*(df)
            Amww1_wave = np.sqrt((Sww1_wave+0j)*(df))
            Amww2_wave = np.sqrt((Sww2_wave+0j)*(df))
            
            #Wave Magnitudes
            Um_wave = np.sqrt(np.real(Amu_wave)**2 + np.imag(Amu_wave)**2)
            Vm_wave = np.sqrt(np.real(Amv_wave)**2 + np.imag(Amv_wave)**2)
            W1m_wave = np.sqrt(np.real(Amww1_wave)**2 + np.imag(Amww1_wave)**2)
            W2m_wave = np.sqrt(np.real(Amww2_wave)**2 + np.imag(Amww2_wave)**2)
            
            #Wave reynolds stress
            uw1_wave = np.nansum(Um_wave*W1m_wave*np.cos(W1ph[waverange]-Uph[waverange]))
            uw2_wave = np.nansum(Um_wave*W2m_wave*np.cos(W2ph[waverange]-Uph[waverange]))
            uv_wave =  np.nansum(Um_wave*Vm_wave*np.cos(Vph[waverange]-Uph[waverange]))
            vw1_wave = np.nansum(Vm_wave*W1m_wave*np.cos(W1ph[waverange]-Vph[waverange]))
            vw2_wave = np.nansum(Vm_wave*W2m_wave*np.cos(W2ph[waverange]-Vph[waverange]))
            w1w2_wave = np.nansum(W1m_wave*W2m_wave*np.cos(W2ph[waverange]- W1ph[waverange]))
            
            uu_wave = np.nansum(Suu_wave*df)
            vv_wave = np.nansum(Svv_wave*df)
            w1w1_wave = np.nansum(Sww1_wave*df)
            w2w2_wave = np.nansum(Sww2_wave*df)
        
            #Full reynolds stresses
            uu = np.nansum(Suu*df)
            uv = np.nansum(Suv*df)
            uw1 = np.nansum(Suw1*df)
            uw2 = np.nansum(Suw2*df)
            vv = np.nansum(Svv*df)
            vw1 = np.nansum(Svw1*df)
            vw2 = np.nansum(Svw2*df)
            w1w1 = np.nansum(Sww1*df)
            w2w2 = np.nansum(Sww2*df)
            w1w2 = np.nansum(Sw1w2*df)
            
            #Turbulent reynolds stresses           
            upup = uu - uu_wave
            vpvp = vv - vv_wave
            w1pw1p = w1w1 - w1w1_wave
            w2pw2p = w2w2 - w2w2_wave
            upw1p = uw1 - uw1_wave
            upw2p = uw2 - uw2_wave
            upvp = uv - uv_wave
            vpw1p = vw1 - vw1_wave
            vpw2p = vw2 - vw2_wave
            w1pw2p = w1w2 - w1w2_wave
            
            #Turbulent reynolds stresses
            waveturb['uw1'][jj] = upw1p
            waveturb['vw1'][jj] = vpw1p
            waveturb['uw2'][jj] = upw2p
            waveturb['vw2'][jj] = vpw2p
            waveturb['uv'][jj] = upvp
            waveturb['uu'][jj] = upup
            waveturb['vv'][jj] = vpvp
            waveturb['w1w1'][jj] = w1pw1p
            waveturb['w2w2'][jj] = w2pw2p
            waveturb['w1w2'][jj] = w1pw2p
            
            #Wave reynolds stresses
            waveturb['uw1_wave'][jj] = uw1_wave
            waveturb['vw1_wave'][jj] = vw1_wave
            waveturb['uw2_wave'][jj] = uw2_wave
            waveturb['vw2_wave'][jj] = vw2_wave
            waveturb['uv_wave'][jj] = uv_wave
            waveturb['uu_wave'][jj] = uu_wave
            waveturb['vv_wave'][jj] = vv_wave
            waveturb['w1w1_wave'][jj] = w1w1_wave
            waveturb['w2w2_wave'][jj] = w2w2_wave
            waveturb['w1w2_wave'][jj] = w1w2_wave
                
    return waveturb

def get_r2(f,xdata,ydata,popt):
    

    residuals = ydata - f(xdata,*popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.nanmean(ydata))**2)
    
    r2 = 1 - (ss_res/ss_tot)
    return r2
    
                

def get_turb(vectrino_kk,fs):
    
    #initializing dict
    turb = dict()  
    
    m,n = np.shape(vectrino_kk['velmaj'])
    
    #Reynolds stress
    #Just using the covariance, will give one value for the entire burst        
    turb['uw1'] = np.empty((m,))
    turb['vw1'] = np.empty((m,))
    turb['uw2'] = np.empty((m,))
    turb['vw2'] = np.empty((m,))
    turb['uv'] = np.empty((m,))
    turb['w1w2'] = np.empty((m,))
    turb['uu'] = np.empty((m,))
    turb['vv'] = np.empty((m,))
    turb['w1w1'] = np.empty((m,))
    turb['w2w2'] = np.empty((m,))
    
    u = copy.deepcopy(vectrino_kk['velmaj'])
    v = copy.deepcopy(vectrino_kk['velmin'])
    w1 = copy.deepcopy(vectrino_kk['w1'])
    w2 = copy.deepcopy(vectrino_kk['w2'])
    
    for jj in range(np.size(vectrino_kk['z'])):
        turb['uw1'][jj] = np.cov(u[jj,:],w1[jj,:])[0][1]
        turb['vw1'][jj] = np.cov(v[jj,:],w1[jj,:])[0][1]
        turb['uw2'][jj] = np.cov(u[jj,:],w2[jj,:])[0][1]
        turb['vw2'][jj] = np.cov(v[jj,:],w2[jj,:])[0][1]
        turb['uv'][jj] = np.cov(u[jj,:],v[jj,:])[0][1]
        turb['w1w2'][jj] = np.cov(w1[jj,:],w2[jj,:])[0][1]
        turb['uu'][jj] = np.cov(u[jj,:],u[jj,:])[0][1]
        turb['vv'][jj] = np.cov(v[jj,:],v[jj,:])[0][1]
        turb['w1w1'][jj] = np.cov(w1[jj,:],w1[jj,:])[0][1]
        turb['w2w2'][jj] = np.cov(w2[jj,:],w2[jj,:])[0][1]
         
    
    #Log law fitting
    
    #General log law which includes an offset, or canopy penetration parameter d
    def loglaw(z,ustar,z0,d):  
        u = (ustar/0.41)*np.log((z-d)/z0)
        return u
        
    velmean = np.nanmean(u, axis = 1)
    H = len(velmean[vectrino_kk['z']>=0])
    
    botind = np.arange(H//4,H) #Adjust this based on data
    ubar = np.abs(velmean[botind])
    z = vectrino_kk['z'][botind] 
        
    try:
        
        #Finding canopy height based on velocity gradient
        #This is just to isolate the section of the velocity profile that we
        #want to fit.
        dubar = np.gradient(ubar,-.001,edge_order = 2)
        
        if (ubar[np.nanargmax(dubar) + 1] > .01):
            hc = z[np.nanargmax(dubar) + 2]
        else:
            hc = z[np.nanargmax(dubar) + 1]
    
        hcidx = np.nanargmin(np.abs(z-hc))
        zfit = z[:hcidx + 1]
        ufit = ubar[:hcidx + 1]
                
        ustarvsl = np.sqrt(1e-6*scipy.stats.linregress(zfit[-3:],ufit[-3:])[0])
        intercept = scipy.stats.linregress(zfit[-3:],ufit[-3:])[1]
    
        Plog,Pcov = scipy.optimize.curve_fit(loglaw,zfit[:-2],ufit[:-2],p0 = (0.01,1e-5,hc),maxfev = 10000)
        
        logfit = (Plog[0]/.41)*np.log((zfit)/Plog[1])
        logerror = np.abs(logfit - ufit)
    
        linfit = (ustarvsl**2)*(zfit)/1e-6 + intercept
        linerror = np.abs(linfit - ufit)
        
        #Starting log fit wherever the linear VSL profile becomes a worse fit
        evec = np.where(logerror<linerror)[0]
        if np.size(evec) == 0:
            logstart = 3
        else:
            logstart = np.where(logerror<linerror)[0][-1]
        
        turb['ustar_vsl'] = ustarvsl
        

        #Fitting for log ustar
        Plog,Pcov = scipy.optimize.curve_fit(loglaw,zfit[:logstart + 1],ufit[:logstart + 1],p0 = (0.01,1e-5,hc/2),
                                              maxfev = 10000, bounds = (1e-7,[.04,1e-2,hc]))
        
        turb['ustarc'] = Plog[0]
        turb['z0c'] = Plog[1]
        turb['dc'] = Plog[2]
        
        turb['r2ustarvsl'] = scipy.stats.linregress(zfit[-3:],ufit[-3:])[2]**2
        turb['r2ustarc'] = get_r2(loglaw,zfit[:logstart + 1],ufit[:logstart + 1],Plog)

    except ValueError:
        turb['ustar_vsl'] = np.NaN
        turb['ustarc'] = np.NaN
        turb['z0c'] = np.NaN
        turb['r2ustarvsl'] = np.NaN
        turb['r2ustarc'] = np.NaN
        turb['dc'] = np.nan
    except TypeError:
        turb['ustar_vsl'] = np.NaN
        turb['ustarc'] = np.NaN
        turb['z0c'] = np.NaN
        turb['r2ustarlog'] = np.NaN
        turb['r2ustarc'] = np.NaN
        turb['dc'] = np.nan
    except RuntimeError:
        turb['ustar_vsl'] = np.NaN
        turb['ustarc'] = np.NaN
        turb['z0c'] = np.NaN
        turb['r2ustarlog'] = np.NaN
        turb['r2ustarc'] = np.NaN
        turb['dc'] = np.nan
    except IndexError:
        turb['ustar_vsl'] = np.NaN
        turb['ustarc'] = np.NaN
        turb['z0c'] = np.NaN
        turb['r2ustarlog'] = np.NaN
        turb['r2ustarc'] = np.NaN
        turb['dc'] = np.nan

    

    return turb


def noise_correction(turb):
    # Implementing the noise correction method for the Reynolds stress tensor of Thomas et al. 
    
    cal = sio.loadmat('/Users/gegan/Documents/Python/Research/General/vectrino_calibration.mat')['M']      
    m,n = np.shape(cal)
    
    RScorr = dict()

    #Setting up output dict
    RScorr['uu'] = np.empty((m,))
    RScorr['uv'] = np.empty((m,))
    RScorr['w1w1'] = np.empty((m,))
    RScorr['w2w2'] = np.empty((m,))
    RScorr['vv'] = np.empty((m,))
    RScorr['uw1'] = np.empty((m,))
    RScorr['uw2'] = np.empty((m,))
    RScorr['vw1'] = np.empty((m,))
    RScorr['vw2'] = np.empty((m,))
    RScorr['w1w2'] = np.empty((m,))
    RScorr['var13'] = np.empty((m,))
    RScorr['var24'] = np.empty((m,))

        
    for i in range(m):
        ai = np.reshape(cal[i,:],(4,4))
        
        num13 = (((ai[3,1]*(ai[3,1] - ai[2,1]) + ai[3,3]*(ai[3,3]-ai[2,3]))*(turb['w1w1'][i]-turb['w1w2'][i]))-
                 ((ai[2,1]*(ai[2,1] - ai[3,1]) + ai[2,3]*(ai[2,3]-ai[3,3]))*(turb['w2w2'][i] - turb['w1w2'][i])))
        
        den13 = (((ai[2,0]*(ai[2,0]-ai[3,0]) + ai[2,2]*(ai[2,2] - ai[3,2]))*(ai[3,1]*(ai[3,1]-ai[2,1]) + ai[3,3]*(ai[3,3] - ai[2,3]))) -
        ((ai[3,0]*(ai[3,0]-ai[2,0]) + ai[3,2]*(ai[3,2] - ai[2,2]))*(ai[2,1]*(ai[2,1]-ai[3,1]) + ai[2,3]*(ai[2,3] - ai[3,3]))))
        
        var13 = num13/den13
        
        num24 = (((ai[2,0]*(ai[2,0] - ai[3,0]) + ai[2,2]*(ai[2,2]-ai[3,2]))*(turb['w2w2'][i]-turb['w1w2'][i]))-
                 ((ai[3,0]*(ai[3,0] - ai[2,0]) + ai[3,2]*(ai[3,2]-ai[2,2]))*(turb['w1w1'][i] - turb['w1w2'][i])))
        
        den24 = den13
        
        var24 = num24/den24
                   
        #Weighting factors
        auu = ai[0,:]*ai[0,:]
        auv = ai[0,:]*ai[1,:]
        auw1 = ai[0,:]*ai[2,:]
        auw2 = ai[0,:]*ai[3,:]
        avv = ai[1,:]*ai[1,:]
        avw1 = ai[1,:]*ai[2,:]
        avw2 = ai[1,:]*ai[3,:]
        aw1w1 = ai[2,:]*ai[2,:]
        aw1w2 = ai[2,:]*ai[3,:]
        aw2w2 = ai[3,:]*ai[3,:]
        
        RScorr['uu'][i] = turb['uu'][i] -  np.max([0,np.sum((auu[[0,2]]*var13 + auu[[1,3]]*var24))])
        RScorr['vv'][i] = turb['vv'][i] -  np.max([0,np.sum((avv[[0,2]]*var13 + avv[[1,3]]*var24))])
        RScorr['w1w1'][i]= turb['w1w1'][i] -  np.max([0,np.sum((aw1w1[[0,2]]*var13 + aw1w1[[1,3]]*var24))])
        RScorr['w2w2'][i] = turb['w2w2'][i] -  np.max([0,np.sum((aw2w2[[0,2]]*var13 + aw2w2[[1,3]]*var24))])
        RScorr['uv'][i]  = turb['uv'][i] -  np.max([0,np.sum((auv[[0,2]]*var13 + auv[[1,3]]*var24))])
        RScorr['uw1'][i] = turb['uw1'][i] -  np.max([0,np.sum((auw1[[0,2]]*var13 + auw1[[1,3]]*var24))])
        RScorr['uw2'][i] = turb['uw2'][i] -  np.max([0,np.sum((auw2[[0,2]]*var13 + auw2[[1,3]]*var24))])
        RScorr['vw1'][i]= turb['vw1'][i] -  np.max([0,np.sum((avw1[[0,2]]*var13 + avw1[[1,3]]*var24))])
        RScorr['vw2'][i] = turb['vw2'][i] - np.max([0,np.sum((avw2[[0,2]]*var13 + avw2[[1,3]]*var24))])
        RScorr['w1w2'][i] = turb['w1w2'][i] - np.max([0,np.sum((aw1w2[[0,2]]*var13 + aw1w2[[1,3]]*var24))])
        RScorr['var13'][i] = var13
        RScorr['var24'][i] = var24

        
    return RScorr

def get_dissipation(vectrino,fs,method):   

    """Estimates dissipation based on one of various methods. Takes as input a python dictionary
    that contains 2D arrays of major, minor, and vertical components of velocity, along with the sampling
    frequency and a string to indicate method choice. This calculates dissipation over a profile because it
    was used for a Vectrino Profiler, but should be easily adapted to a point ADV"""

    if method == 'Fedd07':
        
        """Feddersen et al 2007"""
        def calcFeddA6(sig1,sig2,sig3,u1,u2,omega):
            arrlen = 120
            rho = np.logspace(-2,4,arrlen)
            theta = np.linspace(-np.pi,np.pi,arrlen/10)
            phi = np.linspace(-np.pi/2,np.pi/2,arrlen/10)
                    
            #This way is pretty slow but I think it works
            I3 = np.zeros((len(theta),len(phi),len(rho)))
            for jj in range(len(phi)):
                for kk in range(len(rho)):
                    gamma = np.sqrt((np.cos(phi[jj])**2)*(((np.cos(theta)**2)/sig1**2) + ((np.sin(theta)**2)/sig2**2))
                        + ((np.sin(phi[jj])**2)/(sig3**2)) )
                    ksquare = (gamma**2)*(rho[kk]**2)
                    k3 = rho[kk]*np.sin(phi[jj])/sig3
                    
                    I3[:,jj,kk] = (gamma**(-11./3))*(1-(k3*k3/ksquare))*np.exp((-(rho[kk]*np.cos(phi[jj])*(
                        (np.cos(theta)*u1/sig1) + (np.sin(theta)*u2/sig2))-omega)**2)/(2*rho[kk]**2))*(
                        np.cos(phi[jj])/(sig1*sig2*sig3))
            I2 = np.trapz(I3,theta,axis = 0)
            
            I1 = np.trapz(I2,phi,axis = 0)

            M33 = np.trapz(rho**(-8./3)*I1,rho)
            return M33
        
        m,n = np.shape(vectrino['velmaj'])
        
        eps = np.zeros((m,))
        omega_range = [2*np.pi*4,2*np.pi*6] #Inertial subrange 
        alpha = 1.5
            
        for ii in range(m):
            u = vectrino['velmaj'][ii,:]
            v = vectrino['velmin'][ii,:]
            w = vectrino['w1'][ii,:]
            
            if np.sum(np.isnan(u)) < len(u)/2:
            
                fw,Pw = sig.welch(w,fs = fs, window = 'hamming', nperseg = len(w)//20,
                                      detrend = 'linear')

                omega = 2*np.pi*fw
                
                inds = (omega > omega_range[0]) & (omega < omega_range[1])
                omega = omega[inds]
                Pw = (Pw[inds])/(2*np.pi)
              
                sig1 = np.std(u)
                sig2 = np.std(v)
                sig3 = np.std(w)
                
                u1 = np.abs(np.nanmean(u))
                u2 = np.abs(np.nanmean(v))
                
                
                M33 = np.zeros_like(omega)
                for jj in range(len(omega)): 
                    M33[jj] = calcFeddA6(sig1,sig2,sig3,u1,u2,omega[jj])
                
                epsomega = ((Pw*2*(2*np.pi)**(3/2))/(alpha*M33))**(3/2)
                
                eps[ii] = np.nanmean(epsomega)
        
        return eps
    
    elif method == 'TE01':
        
        """Trowbridge & Elgar 2001, much of this code taken from Dolfyn package
        reworked for Vectrino Profiler"""
        
        
        def calcA13(sigV,theta):
        #Integral from appendix
           x = np.arange(-20,20,1e-2)
           out = np.empty_like(sigV.flatten())
           for i, (b,t) in enumerate(zip(sigV.flatten(),theta.flatten())):
               out[i] = np.trapz(np.cbrt(x ** 2 - 2 / b * np.cos(t) * x + 
                  b ** (-2)) *np.exp(-0.5 * x ** 2), x)
        
           return out.reshape(sigV.shape)*(2*np.pi)**(-0.5) * sigV**(2/3)
  
        def up_angle(u,v):
            Uh = naninterp(u) + 1j*naninterp(v)
            dt = sig.detrend(Uh)
            fx = dt.imag <= 0
            dt[fx] = dt[fx]*np.exp(1j * np.pi)
            return np.angle(np.mean(dt,-1,dtype = np.complex128))
        
        def U_angle(u,v):
            n = np.nanmean(v)
            e = np.nanmean(u)
            return np.arctan2(n,e)
        
        m,n = np.shape(vectrino['velmaj'])
        
        eps = np.zeros((m,))
        omega_range = [2*np.pi*4,2*np.pi*6] #Customize this based on instrument and spectra
   
        for ii in range(m):
            u = vectrino['velmaj'][ii,:]
            v = vectrino['velmin'][ii,:]
            w = vectrino['w1'][ii,:]
            
            
            if np.sum(np.isnan(u)) < len(u)/2:
                V = np.sqrt(np.nanmean(u**2 + v**2))
                sigma = np.std(np.sqrt(u**2+v**2))
                
                thetaup = up_angle(u,v)
                thetaU = U_angle(u,v)
                theta = thetaU - thetaup
                
                alpha = 1.5
                intgrl = calcA13(sigma/V,theta)
                
                fu,Pu = sig.welch(u,fs = fs, window = 'hamming', nperseg = len(u)//20,
                                  detrend = 'linear')
                fv,Pv = sig.welch(v,fs = fs, window = 'hamming', nperseg = len(v)//20,
                                  detrend = 'linear')
                fw,Pw = sig.welch(w,fs = fs, window = 'hamming', nperseg = len(w)//20,
                                  detrend = 'linear')
                
                noiserange = (fu>=20) & (fu<=30) #Customize this based on instrument and spectra
                noiselevel = np.nanmean(Pu[noiserange] + Pv[noiserange])
                
                omega = 2*np.pi*fu
                inds = (omega > omega_range[0]) & (omega < omega_range[1])
                omega = omega[inds]
                Pu = Pu[inds]/(2*np.pi)
                Pv = Pv[inds]/(2*np.pi)
                Pw = Pw[inds]/(2*np.pi)
                
                uv = (np.mean((Pu + Pv - noiselevel)*(omega)**(5/3))/
                      (21/55*alpha*intgrl))**(3/2)/V
                      
                #Adding w component
                uv += (np.mean((Pw)*(omega)**(5/3))/
                      (12/55*alpha*intgrl))**(3/2)/V
                
               # Averaging
                uv *= 0.5
                
                eps[ii] = uv
            else:
                eps[ii] = np.NaN
        return eps
    
    elif method == 'full':
        
        """Calculating based on velocity gradients w/ frozen turbulence assumption.
        Probably a bad idea, wouldn't use"""
    
        u = vectrino['velmaj']
        v = vectrino['velmin']
        w = vectrino['w1']
        
        #Calculating fluctuating velocities
        m,n = np.shape(u)
        up = np.zeros((m,n))
        vp = np.zeros((m,n))
        wp = np.zeros((m,n))
        
        
        for ii in range(m):
            if np.sum(np.isnan(u[ii,:])) < len(u[ii,:])/2:
                fu,Pu = sig.welch(u[ii,:],fs = fs, window = 'hamming', nperseg = n//50, detrend = 'linear')
                fv,Pv = sig.welch(v[ii,:],fs = fs, window = 'hamming', nperseg = n//50, detrend = 'linear')
                fw,Pw = sig.welch(w[ii,:],fs = fs, window = 'hamming', nperseg = n//50, detrend = 'linear')
                
                fumax = fu[np.argmax(Pu)]
                fvmax = fv[np.argmax(Pv)]
                fwmax = fw[np.argmax(Pv)]
                
                try:
                    bu,au = sig.butter(2,fumax/(fs/2))
                    bv,av = sig.butter(2,fvmax/(fs/2))
                    bw,aw = sig.butter(2,fwmax/(fs/2))
                except ValueError:
                    bu,au = sig.butter(2,.35/(fs/2))
                    bv,av = sig.butter(2,.35/(fs/2))
                    bw,aw = sig.butter(2,.35/(fs/2))

                
                ufilt = sig.filtfilt(bu,au,u[ii,:])
                vfilt = sig.filtfilt(bv,av,v[ii,:])
                wfilt = sig.filtfilt(bw,aw,w[ii,:])
                
                up[ii,:] = u[ii,:] - ufilt
                vp[ii,:] = v[ii,:] - vfilt
                wp[ii,:] = w[ii,:] - wfilt
        
        ubar = np.nanmean(u, axis = 1)
        vbar = np.nanmean(v, axis = 1)
        dudz = np.gradient(up,np.diff(vectrino['z'])[0],edge_order = 2 , axis = 0)
        dvdz = np.gradient(vp,np.diff(vectrino['z'])[0],edge_order = 2 , axis = 0)
        dwdz = np.gradient(wp,np.diff(vectrino['z'])[0],edge_order = 2 , axis = 0)
        
        dudt = np.gradient(up,(1./fs),edge_order = 2, axis = 1)
        dvdt = np.gradient(vp,(1./fs),edge_order = 2, axis = 1)
        dwdt = np.gradient(wp,(1./fs),edge_order = 2, axis = 1)
        
        dudx = np.empty_like(dudz)
        dudy = np.empty_like(dudz)
        dvdx = np.empty_like(dvdz)
        dvdy = np.empty_like(dvdz)
        dwdx = np.empty_like(dwdz)
        dwdy = np.empty_like(dwdz)
        
        for ii in range(m):
            dudx[ii,:] = dudt[ii,:]/ubar[ii]
            dudy[ii,:] = dudt[ii,:]/vbar[ii]
            dvdx[ii,:] = dvdt[ii,:]/ubar[ii]
            dvdy[ii,:] = dvdt[ii,:]/vbar[ii]
            dwdx[ii,:] = dwdt[ii,:]/ubar[ii]
            dwdy[ii,:] = dwdt[ii,:]/vbar[ii]
        
        eps = np.zeros((m,))
        
        for ii in range(m):
            S11 = dudx[ii,:]
            S12 = 0.5*(dudy[ii,:] + dvdx[ii,:])
            S13 = 0.5*(dudz[ii,:] + dwdx[ii,:])
            S21 = copy.deepcopy(S12)
            S22 = dvdy[ii,:]
            S23 = 0.5*(dvdz[ii,:] + dwdy[ii,:])
            S31 = copy.deepcopy(S13)
            S32 = copy.deepcopy(S23)
            S33 = dwdz[ii,:]
            
            eps[ii] = 2*1e-6*np.nanmean(S11**2 + S12**2 + S13**2 + S21**2 + 
               S22**2 + S23**2 + S31**2 + S32**2 + S33**2)
        
        return eps
    
    elif method == 'scaling':
        
        """scaling estimate based on integral length scales"""
        
        u = vectrino['velmaj']
        
        Tint, Lint, uprime = IntScales(u,fs)
        
        eps = uprime**3/Lint
        
        return eps
        
        

def IntScales(u,fs):
    
    """Integral length scales for estimating dissipation"""
    
    m,n = np.shape(u)
    
    ubar = np.nanmean(u,axis = 1)
    up = np.zeros((m,n)) 
  
    for ii in range(m):
        if np.sum(np.isnan(u[ii,:])) < len(u[ii,:])/2:
            fu,Pu = sig.welch(u[ii,:],fs = fs, window = 'hamming', nperseg = n//50, detrend = 'linear')
            fumax = fu[np.argmax(Pu)] 
            try:
                bu,au = sig.butter(2,fumax/(fs/2))
            except ValueError:
                bu,au = sig.butter(2,.35/(fs/2))
            ufilt = sig.filtfilt(bu,au,u[ii,:])
            up[ii,:] = u[ii,:] - ufilt
        
    lags = np.arange(0,n//4)
    Tint = np.zeros((m,))
    Lint = np.zeros((m,))
    for ii in range(m):
        uptemp = up[ii,:n//4]
        if np.sum(np.isnan(uptemp)) < len(uptemp)/2:
            uptemp = sig.detrend(uptemp)
            r = autocorr(uptemp,lags)
            lnum = np.argmax(r<0)
            Tint[ii] = (1/fs)*lnum
            Lint[ii] = Tint[ii]*ubar[ii]
    
    uprime = np.sqrt(np.nanvar(up, axis = 1))

    
    return Tint, Lint, uprime
    
            
def despikeGN(velarray,fs):
    
    """Goring & Nikora ADV despiking algorithm"""
    
    def ellipsoid_3d_outlier(xi,yi,zi,theta):
        n = len(xi)
        lam = np.sqrt(2*np.log(n))
        
        xp = []
        yp = []
        zp = []
        ip = []
        
        if theta == 0:
            X = xi
            Y = yi
            Z = zi
        else:
            R = np.array([[np.cos(theta),0,np.sin(theta)],[0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            X = xi*R[0,0] + yi*R[0,1] + zi*R[0,2]
            Y = xi*R[1,0] + yi*R[1,1] + zi*R[1,2]
            Z = xi*R[2,0] + yi*R[2,1] + zi*R[2,2]
        
        a = lam*np.nanstd(X)
        b = lam*np.nanstd(Y)
        c = lam*np.nanstd(Z)
        
        m = 0
        for i in range(n):
            x1 = X[i]
            y1 = Y[i]
            z1 = Z[i]
            
            x2 = a*b*c*x1/np.sqrt((a*c*y1)**2 + (b**2) + ((c**2)*(x1**2)+ (a**2)*(z1**2)))
            y2 = a*b*c*y1/np.sqrt((a*c*y1)**2 + (b**2) + ((c**2)*(x1**2)+ (a**2)*(z1**2)))
            zt = (c**2)*(1-(x2/a)**2 - (y2/b)**2)
            
            if z1<0:
                z2 = -np.sqrt(zt)
            elif z1>0:
                z2 = np.sqrt(zt)
            else:
                z2 = 0
            
            dis = (x2**2 + y2**2 + z2**2) - (x1**2 + y1**2 * z1**2)
            
            if dis<0:
                ip = np.concatenate((ip,[i]))
                xp = np.concatenate((xp,[xi[i]]))
                yp = np.concatenate((yp,[yi[i]]))
                zp = np.concatenate((zp,[zi[i]]))
                m += 1
        
        return xp,yp,zp,[int(i) for i in ip]
      
        
    m,n = np.shape(velarray)
    velout = np.empty_like(velarray)
    indout = []
    for ii in range(m):
        u = velarray[ii,:].squeeze()
        
        n_iter = 20
        n_out = 999
        u_mean = 0
        
        n_loop = 1
        while (n_out != 0) & (n_loop <= n_iter):
            u_mean = u_mean + np.nanmean(u)
            u  = u - np.nanmean(u)
            
            ut = np.gradient(u,(1./fs),edge_order = 2)
            utt = np.gradient(ut,(1./fs),edge_order = 2)
            
            if (n_loop == 1):
                theta = np.arctan2(np.sum(u*utt),np.sum(u**2))
            
            xp,yp,zp,ip = ellipsoid_3d_outlier(u,ut,utt,theta)
            
            indout = np.concatenate((indout,ip))
            n_nan_1 = np.sum(np.isnan(u))
            u[ip] = np.NaN
            n_nan_2 = np.sum(np.isnan(u))
            n_out = n_nan_2 - n_nan_1
            n_loop += 1
        
        go = u + u_mean
        uo = naninterp(go)
        velout[ii,:] = uo
    return indout, velout
         