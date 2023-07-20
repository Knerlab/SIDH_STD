#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:33:12 2022

@author: shaohli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 01:41:12 2022
ZH fixed to 180, study f2
@author: shaohli
"""


#!/apps/eb/Python/3.8.2-GCCcore-8.3.0/bin/python3
# -*- coding: utf-8 -*-


import os, time
import numpy as np
import tifffile as tf
import numpy.random as rd
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import pandas as pd
from scipy.optimize import fmin
import functools
import itertools as IT
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


pi = np.pi
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
cm = ndimage.measurements.center_of_mass

class hologram_simulation(object):
    
    def __init__(self):
        self.wl = 0.00067  #Wavelenth of the light
        self.na = 1.42    # Numerical aperture of objective
        self.dx = 0.016 #
        self.nx = 256
        self.ny = 256     
        self.dp = 1 / (self.nx * 2 * self.dx)
        self.k = 2 * np.pi / self.wl
        
        self.f_o = 3       # Focal length of objective (mm)
        self.d_slm = 3    # Distance between objective and SLM 
        self.z_s = 3.000     # Distance between sample and objective (mm)
        self.f_slm1 = 300
        
        self.f_TL = 180.         #focal length of tube lens
        self.f_2 = 120.          #focal length of second lens
        self.z_h = 150.          #distance between Interferometer and camera
        self.d1 = 183.           #distance between objective and tube lens
        self.d2 = self.f_TL + self.f_2          #distance between tube lens and second lens
        self.f_3 = 120.          #focal length of third lens
        self.f_4 = 100.          #focal length of fourth lens
        self.d3 = self.f_2 + self.f_3         #distance between second lens and third lens
        self.d4 = self.f_4 + self.f_3          #distance between third lens and fourth lens
        self.d5 = self.f_4          #distance between fourth lens and interferometer
        
        
        self.xr = np.arange(-self.nx, self.nx)
        self.yr = np.arange(-self.ny, self.ny)
        self.xv, self.yv = np.meshgrid(self.xr, self.yr, indexing='ij', sparse=True)
        self.xv1 = self.dx * self.xv
        self.yv1 = self.dx * self.yv
        self.xv2 = self.dx * self.xv
        self.yv2 = self.dx * self.yv
        
        self.dz = 0.0001                        # z stepsize 500nm
   
        self.radius = 50
        self.msk = (self.xv ** 2 + self.yv ** 2) <= self.radius**2
        self.msk = self.msk * 1           # remove Nan
        
        self.Nph = 6000
        self.bg = 1
        
        self.path = time.strftime("%Y%m%d_%H%M%S" + 'bg' + str(self.bg)) 

        
    def __del__(self):
        pass
    
    def run(self):
        sx = 0
        sy = 0
        self.generate_holoImgs(sx,sy,verbose=True)
        self.finch_recon(verbose=True)
        
    
    # def recon_dist_calc(self, z_s, z_h, f_slm2):
    #     f_o = self.f_o
    #     f_slm1 = self.f_slm1
    #     d = self.d_slm
    #     if z_s == f_o:
    #         # recon_dist = np.abs(( (z_h - f_slm1) * (z_h - f_slm2) )/(f_slm1 - f_slm2))
    #         recon_dist = ( (z_h - f_slm1) * (z_h - f_slm2) )/(f_slm1 - f_slm2)
    #     else:
    #         z_d = ( z_s * (f_o - d) + f_o * d )/(f_o - z_s)
    #         z_f1 = z_h * z_d - f_slm1 * (z_d + z_h)
    #         z_f2 = z_h * z_d - f_slm2 * (z_d + z_h)
    #         # recon_dist = np.abs( (z_f1 * z_f2) / (z_d * z_d *(f_slm1 - f_slm2)))
    #         recon_dist =  (z_f1 * z_f2) / (z_d * z_d *(f_slm1 - f_slm2))
    #     return recon_dist
    
    def recon_dist_calc_realsetting(self, z_s, z_h, f_slm2):
        f_o = self.f_o
        f_slm1 = self.f_slm1
        # d_slm = self.d_slm
        d1 = self.d1
        d2 = self.d2
        d3 = self.d3
        d4 = self.d4
        d5 = self.d5
        f_TL = self.f_TL
        f_2 = self.f_2
        f_3 = self.f_3
        f_4 = self.f_4
        if z_s == f_o:
            # recon_dist = np.abs(( (z_h - f_slm1) * (z_h - f_slm2) )/(f_slm1 - f_slm2))
            recon_dist = ( (z_h - f_slm1) * (z_h - f_slm2) )/(f_slm1 - f_slm2)
        else:
            f_e = (z_s*f_o)/(f_o-z_s)
            z_d = f_e +d1
            f_bar1 = (f_TL*z_d)/(f_TL-z_d)
            z_d1 = f_bar1+d2
            f_bar2 = (f_2*z_d1)/(f_2-z_d1)
            z_d2 = f_bar2+d3
            f_bar3 = (f_3*z_d2)/(f_3-z_d2)
            z_d3 = f_bar3+d4
            f_bar4 = (f_4*z_d3)/(f_4-z_d3)
            z_d4 = f_bar4+d5
            
            z_f1 = z_h * z_d4 - f_slm1 * (z_d4 + z_h)
            z_f2 = z_h * z_d4 - f_slm2 * (z_d4 + z_h)
            # recon_dist = np.abs( (z_f1 * z_f2) / (z_d * z_d *(f_slm1 - f_slm2)))
            recon_dist =  (z_f1 * z_f2) / (z_d4 * z_d4 *(f_slm1 - f_slm2))
      
            # ###############Mt###########
            # f_e = (z_s*f_o)/(f_o-z_s)
            # magt = (z_h*f_e*f_bar1*f_bar2*f_bar3*f_bar4)/(z_s*z_d*z_d1*z_d2*z_d3*z_d4)    #Mt
            # ######resample###########
            # trans_zero = z_h/f_o
            # z_h_perfect, R_prefect = self.Get_ZH_at_perfect_overlap(z_s, f_slm2)
            # if z_h >= z_h_perfect:
            #     trans_mag = (f_e * z_h)/(z_s*(f_e + f_slm2))
            # else:
            #     trans_mag = (f_e * z_h)/(z_s*(f_e + f_slm1))
            # mag_resample = (trans_mag/trans_zero)
            # #######################
            
        return recon_dist
    
    # def Transverse_Magnification(self, z_s, z_h):
    #     if z_s == self.f_o:
    #         trans_mag = z_h/self.f_o
    #     else:
    #         f_e = (z_s*self.f_o)/(self.f_o-z_s)
    #         trans_mag = (f_e * z_h)/(z_s*(f_e + self.d_slm))
    #     return np.abs(trans_mag)
    
    def Transverse_Magnification_realsetting(self, z_s, z_h):
        f_o = self.f_o
        # d_slm = self.d_slm
        d1 = self.d1
        d2 = self.d2
        d3 = self.d3
        d4 = self.d4
        d5 = self.d5
        f_TL = self.f_TL
        f_2 = self.f_2
        f_3 = self.f_3
        f_4 = self.f_4
        if z_s == self.f_o:
            trans_mag = z_h/self.f_o
        else:
            f_e = (z_s*f_o)/(f_o-z_s)
            z_d = ( z_s * (f_o - d1) + f_o * d1)/(f_o - z_s)
            f_bar1 = (f_TL*z_d)/(f_TL-z_d)
            z_d1 = (f_bar1+d2)
            f_bar2 = (f_2*z_d1)/(f_2-z_d1)
            z_d2 = f_bar2+d3
            f_bar3 = (f_3*z_d2)/(f_3-z_d2)
            z_d3 = f_bar3+d4
            f_bar4 = (f_4*z_d3)/(f_4-z_d3)
            z_d4 = f_bar4+d5
            trans_mag = (z_h*f_e*f_bar1*f_bar2*f_bar3*f_bar4)/(z_s*z_d*z_d1*z_d2*z_d3*z_d4)    #Mt
        return np.abs(trans_mag)
    
    
    # def Hologram_radius(self, z_s, z_h, f_slm, d_slm, f_o , na):
    #     M1=np.array([[1,z_h],[0,1]])
    #     M2=np.array([[1,0],[(-1/f_slm),1]])
    #     M3=np.array([[1,d_slm],[0,1]])
    #     M4=np.array([[1,0],[(-1/f_o),1]])
    #     M5=np.array([[1,z_s],[0,1]])
    #     M = M1.dot(M2).dot(M3).dot(M4).dot(M5)
    #     Holo_R = M[0,1] * na
    #     return Holo_R
    
    def Hologram_radius_realsetting(self, z_s, z_h, f_slm, f_o , na):
        M1=np.array([[1,z_h],[0,1]])
        M2=np.array([[1,0],[(-1/f_slm),1]])
        M3=np.array([[1,self.d5],[0,1]])
        M4=np.array([[1,0],[(-1/self.f_4),1]])
        M5=np.array([[1,self.d4],[0,1]])
        M6=np.array([[1,0],[(-1/self.f_3),1]])
        M7=np.array([[1,self.d3],[0,1]])
        M8=np.array([[1,0],[(-1/self.f_2),1]])
        M9=np.array([[1,self.d2],[0,1]])
        M10=np.array([[1,0],[(-1/self.f_TL),1]])
        M11=np.array([[1,self.d1],[0,1]])
        M12=np.array([[1,0],[(-1/f_o),1]])
        M13=np.array([[1,z_s],[0,1]])
        M = M1.dot(M2).dot(M3).dot(M4).dot(M5).dot(M6).dot(M7).dot(M8).dot(M9).dot(M10).dot(M11).dot(M12).dot(M13)
        Holo_R_real = M[0,1] * na
        return Holo_R_real
    
    
    # def Get_holo_radius(self, z_s, z_h, f_slm2):
    #     Holo_R_stack1 = self.Hologram_radius(z_s, z_h, self.f_slm1, self.d_slm, self.f_o, self.na)
    #     Holo_R_stack2 = self.Hologram_radius(z_s, z_h, f_slm2, self.d_slm, self.f_o, self.na)
    #     Holo_R_stack = min(np.abs(Holo_R_stack1), np.abs(Holo_R_stack2))
    #     return Holo_R_stack
    
    def Get_holo_radius_realsetting(self, z_s, z_h, f_slm2):
        Holo_R_stack1 = self.Hologram_radius_realsetting(z_s, z_h, self.f_slm1, self.f_o, self.na)
        Holo_R_stack2 = self.Hologram_radius_realsetting(z_s, z_h, f_slm2, self.f_o, self.na)
        Holo_R_stack_real = min(np.abs(Holo_R_stack1), np.abs(Holo_R_stack2))
        return Holo_R_stack_real
    
    
    # def Get_ZH_at_perfect_overlap(self, z_s, f_slm2):
    #     if self.f_slm1 > f_slm2:
    #         z_h_range = np.arange(f_slm2,self.f_slm1,1)
    #     elif self.f_slm1 < f_slm2:
    #         z_h_range = np.arange(self.f_slm1,f_slm2,1)
    #     Rh_list = np.ones((len(z_h_range)))
    #     for i in range(len(z_h_range)):
    #         z_h = z_h_range[i]
    #         Holo_R_stack1 = self.Hologram_radius(z_s, z_h, self.f_slm1, self.d_slm, self.f_o, self.na)
    #         Holo_R_stack2 = self.Hologram_radius(z_s, z_h, f_slm2, self.d_slm, self.f_o, self.na)
    #         Rh_list[i] = min(np.abs(Holo_R_stack1), np.abs(Holo_R_stack2))
    #     index = np.where(Rh_list == max(Rh_list))
    #     z_h_perfect = z_h_range[index]
    #     print('Perfect ZH:',z_h_perfect)
    #     print('Hologram Radius at Perfect ZH:', max(Rh_list))
    #     return z_h_perfect, max(Rh_list)
    
    def Get_ZH_at_perfect_overlap_realsetting(self, z_s, f_slm2):
        if self.f_slm1 > f_slm2:
            z_h_range = np.arange(f_slm2,self.f_slm1,1)
        elif self.f_slm1 < f_slm2:
            z_h_range = np.arange(self.f_slm1,f_slm2,1)
        Rh_list = np.ones((len(z_h_range)))
        for i in range(len(z_h_range)):
            z_h = z_h_range[i]
            Holo_R_stack1 = self.Hologram_radius_realsetting(z_s, z_h, self.f_slm1,  self.f_o, self.na)
            Holo_R_stack2 = self.Hologram_radius_realsetting(z_s, z_h, f_slm2, self.f_o, self.na)
            Rh_list[i] = min(np.abs(Holo_R_stack1), np.abs(Holo_R_stack2))
        index = np.where(Rh_list == max(Rh_list))
        z_h_perfect = z_h_range[index]
        if len(z_h_perfect) == 2:
            z_h_perfect =  z_h_perfect[0]
        else:
            z_h_perfect =  z_h_perfect
        # print('Perfect real ZH:',z_h_perfect)
        # print('Hologram Radius at Perfect real ZH:', max(Rh_list))
        return z_h_perfect, max(Rh_list)
    
    def I_function(self, sx ,sy, x, y, wl, recon_dist, theta, Mt):
        I = 2 + np.exp(((1j * np.pi )/ (wl * recon_dist) ) * 
                        ( x**2 + y**2 - 2*sx*x*Mt - 2*sy*y*Mt + 
                         + (Mt*sx)**2 + (Mt*sy)**2 ) + (1j * theta)) + np.exp((((-1j * np.pi) / (wl * recon_dist) ) *  ( x**2 + y**2 - 2*sx*x*Mt - 2*sy*y*Mt + (Mt*sx)**2 + (Mt*sy)**2 )  + (-1j * theta)))
        return I.real
    
    
    def recon(self, recon_dist, imgstack, xv, yv, msk):
        theta1 = 0 * np.pi / 3                               #Theta timed 2 cause there are two exp(i*theta)
        theta2 = 4 * np.pi / 3
        theta3 = 8 * np.pi / 3
        final_intensity = (imgstack[0] * (np.exp(-1j*theta3)-np.exp(-1j*theta2)) +
                           imgstack[1] * (np.exp(-1j*theta1)-np.exp(-1j*theta3)) +
                           imgstack[2] * (np.exp(-1j*theta2)-np.exp(-1j*theta1)))
        recon_temp = np.exp((1j*pi/self.wl/recon_dist)*((xv)**2+(yv)**2))*msk
        g = ifftshift(ifft2(fft2(final_intensity)*fft2(recon_temp)))
        return  final_intensity, np.abs(g)
    
    
    def generate_holoImgs(self, z_s, z_h, recon_dist, f_slm2, sx, sy, verbose=False):
        self.imgstack = np.zeros((3, self.nx*2, self.ny*2))
        self.Holo_R = self.Get_holo_radius_realsetting(z_s, z_h, f_slm2)
        self.radius = self.Holo_R / self.dx
        self.msk = (self.xv ** 2 + self.yv ** 2) <= self.radius**2
        # Mt = self.Transverse_Magnification(z_s, z_h)
        Mt = self.Transverse_Magnification_realsetting(z_s, z_h)
        for i in range(3):
            h = np.zeros((self.nx*2, self.ny*2))
            bg_metrix = rd.poisson ( self.bg * np.ones((512,512)) )
            theta = i * 2 * np.pi / 3
            h = self.I_function(sx, sy, self.xv2, self.yv2, self.wl, recon_dist, theta, Mt)
            h = h * self.msk
            h = h / h.sum()                         #normalization
            h = h * self.Nph
            h = rd.poisson(h)+bg_metrix
            self.imgstack[i] = h
        if verbose:
            tf.imshow(np.abs(self.imgstack[0]))
            tf.imshow(np.abs(self.imgstack[1]))
            tf.imshow(np.abs(self.imgstack[2]))
            
            
    def finch_recon(self,recon_dist, verbose=False):
        self.final_intensity, self.holo_recon = self.recon(recon_dist, self.imgstack, self.xv1, self.yv1, self.msk)
        if verbose:
            tf.imshow(np.abs(self.final_intensity))
            tf.imshow(np.angle(self.final_intensity))
            tf.imshow(np.abs(self.holo_recon))
            tf.imwrite('2Dreconstack.tif', np.abs(self.holo_recon))
        
            
    def finch_recon3D(self, z_s, z_h, f_slm2, sx, sy):                                                         # 3D reconstruction of FINCH
        recon_dist = self.recon_dist_calc_realsetting(z_s, z_h, f_slm2)
        self.generate_holoImgs(z_s, z_h, recon_dist, f_slm2, sx, sy, False)
        #z_depth(-5um to 5um, step size:dz)
        self.z_depth = np.arange((z_s-0.005), (z_s+0.005), self.dz)        #recon range(z_depth(-5um to 5um, step size:100 nm))
        self.z_depth_n = len(self.z_depth)                       #recon range slides

        zr_depth = np.zeros(self.z_depth_n)
        for i in range(self.z_depth_n):
            zr_depth[i] = self.recon_dist_calc_realsetting(self.z_depth[i], z_h, f_slm2)                  #z_depth : 20um
        self.Recon_3d_stack = np.zeros((len(zr_depth), self.nx*2, self.ny*2))
        #self.intensity_stack = np.zeros((len(zr_depth), self.nx*2, self.ny*2))
        for m in range(len(zr_depth)):
            self.finch_recon(zr_depth[m], False)
            #self.intensity_stack[m,:,:] = self.final_intensity 
            self.Recon_3d_stack[m,:,:] = self.holo_recon   
        #tf.imwrite('Holoimages.tif', np.abs(self.imgstack))
        #tf.imsave('FinalIntensity.tif', np.abs(self.intensity_stack))
        #tf.imwrite('3Dreconstack.tif', np.abs(self.Recon_3d_stack))
        #tf.imshow(np.abs(self.Recon_3d_stack[25,:,:]))
        return zr_depth                          #return z_r of axial range
    
    def get_STD(self, z_s, iteration, f_slm2, sx, sy):       #Calculate the STD of each Z_h(different radius)
        locs_xy = np.zeros((2,iteration))
        FWHM_xy = np.zeros((1,iteration))
        Recon_std_xy = np.zeros((2,1))
        FWHM_mean_xy = np.zeros((1,1))
        Locs_xy = np.zeros((2,1))
        
        
        locs_zy = np.zeros((2,iteration))
        FWHM_zy = np.zeros((2,iteration))
        Recon_std_zy = np.zeros((2,1))
        FWHM_mean_zy = np.zeros((2,1))
        Locs_zy = np.zeros((2,1))
        
        self.z_h_perfect, self.R_prefect = self.Get_ZH_at_perfect_overlap_realsetting(z_s, f_slm2)
        Mt = self.Transverse_Magnification_realsetting(z_s,self.z_h_perfect)
        
        for i in range(iteration):
            #STD calculate for xy plane
            recon_dist = self.recon_dist_calc_realsetting(z_s, self.z_h_perfect, f_slm2)
            self.generate_holoImgs(z_s, self.z_h_perfect, recon_dist, f_slm2, sx, sy, False)
            self.finch_recon(recon_dist, False)
            Recon_xy = self.holo_recon
            "p=(amplitude, x_center, y_center, sigma_xy, background)"
            sec = 6              #zise of sub_XY 
            init_params_xy = (1.e4, sec, sec, 2.,  0.)    
            Sub_Recon_xy = np.abs(Recon_xy)[self.nx-sec:self.nx+sec, self.ny-sec:self.ny+sec]
            out_xy = self.GaussianFit2D(Sub_Recon_xy , init_params_xy)
            locs_xy[0,i] = out_xy[1]
            locs_xy[1,i] = out_xy[2]
            FWHM_xy[0,i] = 2.35*out_xy[3]
            # if i == 1:
            #     self.gauss2Dplot(out_xy,Sub_Recon_xy)
            
            self.holo_recon = []
            
            #STD calculate for zy plane
            self.zr_depth = self.finch_recon3D(z_s, self.z_h_perfect, f_slm2, sx, sy)          #self.Recon_3d_stack
            secz = 6              #zise of sub_Z
            nz, nx, ny = self.Recon_3d_stack.shape
            nz = int(nz/2)
            Recon_sub = np.abs(self.Recon_3d_stack)[nz-secz:nz+secz,self.nx-sec:self.nx+sec,self.nx-sec:self.nx+sec]
            Recon_zy = self.zslides_trans(self.Recon_3d_stack)
            "p=(amplitude, z_center, y_center, sigma_z, sigma_y, background)"
            init_params_zy = (1.e4, secz, sec, 4., 2., 0.)    
            Sub_Recon_zy = np.abs(Recon_zy)[nz-secz:nz+secz,self.ny-sec:self.ny+sec]
            out_zy = self.GaussianFit2DZ(Sub_Recon_zy, init_params_zy)    #3d guassian fit
            #out = self.GaussianFit3D(np.abs(self.Recon_3d_stack)[nz-16:nz+16, self.nx-16:self.nx+16, self.ny-16:self.ny+16], init_params)    #2d guassian fit
            # if i == 1:
            #     self.gauss2DplotZ(out_zy,Sub_Recon_zy)
                
            locs_zy[0,i] = out_zy[1]     #z
            locs_zy[1,i] = out_zy[2]     #y
            FWHM_zy[0,i] = 2.35*out_zy[3]  #FWHM for z
            FWHM_zy[1,i] = 2.35*out_zy[4]  #FWHM for y
            
            # if i == 1:
            #     "f_p=(amplitude, z_center, x_center, y_center, sigma_xy, (sigma_z), background)"
            #     f_p = (out_xy[0],out_zy[1],out_xy[1],out_xy[2],out_xy[3],out_zy[3],out_xy[4])
            #     self.save_fit_data(f_p, Recon_sub)
            #out_imgs[i,:,:] = abs(self.holo_recon)
            
        Locs_xy[0] = np.mean(locs_xy[0,:])   #x
        Locs_xy[1] = np.mean(locs_xy[1,:])   #y
        Recon_std_xy[0] = np.std(locs_xy[0,:]) *  self.dx / Mt 
        Recon_std_xy[1] = np.std(locs_xy[1,:]) *  self.dx / Mt
        FWHM_mean_xy[0] = np.mean(FWHM_xy[0,:]) *  self.dx / Mt 
        
        Locs_zy[0] = np.mean(locs_zy[0,:])   #z
        Locs_zy[1] = np.mean(locs_zy[1,:])   #y
        Recon_std_zy[0] = np.std(locs_zy[0,:]) *  self.dz 
        Recon_std_zy[1] = np.std(locs_zy[1,:]) *  self.dx / Mt
        FWHM_mean_zy[0] = np.mean(FWHM_zy[0,:]) *  self.dz 
        FWHM_mean_zy[1] = np.mean(FWHM_zy[1,:]) *  self.dx / Mt
        # print('FWHM_xy(nm)')
        # print(FWHM_mean_xy)
        # print('FWHM_zy(nm)')
        # print( FWHM_mean_zy)
        
        
        # print('iteration')
        # print(i)
        #tf.imsave('test.tif', out_imgs, photometric='minisblack')
        return Locs_xy, Locs_zy, Recon_std_xy, Recon_std_zy, FWHM_mean_xy, FWHM_mean_zy

    def Get_STD_vs_R_p(self, z_s, iteration, f_slm2, sx, sy): #Calculate the STD of the hologram with different Z_h(different radius)
        try:
            os.mkdir(self.path)
        except:
            print('Directory already exists')
        Recon_std_XY = np.zeros((2,len(f_slm2)))
        Recon_std_ZY = np.zeros((2,len(f_slm2)))
        
        Locs_mean_XY = np.zeros((2,len(f_slm2)))
        Locs_mean_ZY = np.zeros((2,len(f_slm2)))
        
        FWHM_XY = np.zeros((1,len(f_slm2)))
        FWHM_ZY = np.zeros((2,len(f_slm2)))
        
        Holo_R_stack = np.zeros((len(f_slm2)))
        Holo_R_perfect_stack = np.zeros((len(f_slm2)))
        for i in range(len(f_slm2)):
            self.fnd = 'f_slm2_%.2f' %f_slm2[i]
            
            Locs_xy, Locs_zy, Recon_std_xy, Recon_std_zy, FWHM_mean_xy, FWHM_mean_zy = self.get_STD(z_s, iteration, f_slm2[i], sx, sy)
            Holo_R_stack[i] = self.Holo_R       #hologram radius
            Holo_R_perfect_stack[i] = self.R_prefect
            
            
            Recon_std_XY[0,i] = Recon_std_xy[0]*1e6               #x
            Recon_std_XY[1,i] = Recon_std_xy[1]*1e6               #y
            
            
            Recon_std_ZY[0,i] = Recon_std_zy[0]*1e6               #z
            Recon_std_ZY[1,i] = Recon_std_zy[1]*1e6               #y from zslides
            
            Locs_mean_XY[0,i] = Locs_xy[0]            # Mean of Locs position of x for n iteration
            Locs_mean_XY[1,i] = Locs_xy[1]            # Mean of Locs position of y for n iteration
           
            Locs_mean_ZY[0,i] = Locs_zy[0]            #z
            Locs_mean_ZY[1,i] = Locs_zy[1]            #z
            
            FWHM_XY[0,i] = FWHM_mean_xy[0]*1e6           #xy
            
            FWHM_ZY[0,i] = FWHM_mean_zy[0]*1e6           #z
            FWHM_ZY[1,i] = FWHM_mean_zy[1]*1e6           #y
            
            # print('std_xy(nm)')
            # print(Recon_std_XY)
            # print('std_zy(nm)')
            # print(Recon_std_ZY)
            # print('locs_xy')
            # print(Locs_mean_XY)
            # print('locs_zy')
            # print(Locs_mean_ZY)
            # print('FWHM_xy(nm)')
            # print(FWHM_XY)
            # print('FWHM_zy(nm)')
            # print(FWHM_ZY)
            # tf.imwrite(self.path + '/' + self.fnd + '_Holoimages.tif', np.abs(self.imgstack))
            # tf.imwrite(path + '/' + fnd + '_reconimage.tif', np.abs(self.holo_recon))
            # tf.imwrite(self.path + '/' + self.fnd + '_3dreconstack.tif', np.abs(self.Recon_3d_stack))
        save_excel = np.zeros((10,len(f_slm2)))
        save_excel[0,:] = f_slm2
        save_excel[1,:] = Holo_R_perfect_stack
        
        save_excel[2,:] = Recon_std_XY[0,:]
        save_excel[3,:] = Recon_std_XY[1,:]
        save_excel[4,:] = Recon_std_ZY[0,:]
        
        save_excel[5,:] = Locs_mean_XY[0,:]
        save_excel[6,:] = Locs_mean_XY[1,:]
        save_excel[7,:] = Locs_mean_ZY[0,:]
        
        save_excel[8,:] = FWHM_XY[0,:]
        save_excel[9,:] = FWHM_ZY[0,:]
        
        df = pd.DataFrame(save_excel)
        df.index = ['f_slm2', 'R_Perfect', 'x_STD', 'y_STD','z_STD','x_Locs','y_Locs','z_Locs','xy_FWHM','z_FWHM']
        df.to_excel(self.path + '/' + str(self.bg) +'bg' + str(self.z_s) + 'z_s' + 'STD.xlsx', index=True)
        

        fig = plt.figure(figsize=(10,5))
        plt.plot(f_slm2,Holo_R_perfect_stack)
        plt.xlabel("f_slm2(mm)")
        plt.ylabel("R_Perfect(nm)")
        plt.show()
        fig.savefig(self.path + '/' + 'R_STD_Z.jpg',bbox_inches='tight',dpi=150)
        
        fig = plt.figure(figsize=(10,5))
        plt.plot(Holo_R_perfect_stack,Locs_mean_ZY[0,:])
        plt.xlabel("R_Perfect(mm)")
        plt.ylabel("z_locs")
        plt.show()
        fig.savefig(self.path + '/' + 'locs_Z.jpg',bbox_inches='tight',dpi=150)
        
        fig = plt.figure(figsize=(10,5))
        plt.plot(Holo_R_perfect_stack,Locs_mean_XY[0,:])
        plt.plot(Holo_R_perfect_stack,Locs_mean_XY[1,:])
        plt.legend(['x','y'])
        plt.xlabel("R_Perfect(mm)")
        plt.ylabel("xy_locs")
        plt.show()
        fig.savefig(self.path + '/' + 'locs_XY.jpg',bbox_inches='tight',dpi=150)
        
        fig = plt.figure(figsize=(10,5))
        plt.plot(Holo_R_perfect_stack,Recon_std_ZY[0,:])
        plt.xlabel("R_Perfect(mm)")
        plt.ylabel("z_STD(nm)")
        plt.show()
        fig.savefig(self.path + '/' + 'GaussianSTD_3D_Z.jpg',bbox_inches='tight',dpi=150)
        
        fig = plt.figure(figsize=(10,5))
        plt.plot(Holo_R_perfect_stack,Recon_std_XY[0,:])
        plt.plot(Holo_R_perfect_stack,Recon_std_XY[1,:])
        plt.legend(['x','y'])
        plt.xlabel("R_Perfect(mm)")
        plt.ylabel("xy_STD(nm)")
        plt.show()
        fig.savefig(self.path + '/' + 'GaussianSTD_3D_XY.jpg',bbox_inches='tight',dpi=150)
        
        fig = plt.figure(figsize=(10,5))
        plt.plot(Holo_R_perfect_stack,FWHM_ZY[0,:])          #z
        plt.xlabel("R_Perfect(mm)")
        plt.ylabel("FWHM_z(nm)")
        plt.show()
        fig.savefig(self.path + '/' + 'FWHM_z.jpg',bbox_inches='tight',dpi=150)
        
        fig = plt.figure(figsize=(10,5))
        plt.plot(Holo_R_perfect_stack,FWHM_XY[0,:])          #xy
        plt.xlabel("R_Perfect(mm)")
        plt.ylabel("FWHM_xy(nm)")
        plt.show()
        fig.savefig(self.path + '/' + 'FWHM_xy.jpg',bbox_inches='tight',dpi=150)


    def rms(self, arr):
        n = arr.shape
        square = (arr**2).sum()
        mean = square / n
        root = math.sqrt(mean)
        return root

    def gauss2Dplot(self, p, img):
        nx, ny = img.shape
        x = np.arange(nx)
        y = np.arange(ny)
        [xx, yy] = np.meshgrid(x,y,indexing = 'ij')
        imgfit = p[0]*np.exp(-0.5*((xx-p[1])**2 + (yy-p[2])**2)/p[3]**2)+p[4]
        
        fig = plt.figure(figsize=(7,7))
        #plt.imshow(imgfit)
        fig.savefig(self.path + '/' + self.fnd + 'Fit_ZY.jpg',bbox_inches='tight',dpi=150)
        fig = plt.figure(figsize=(7,7))
        #plt.imshow(img)
        fig.savefig(self.path + '/' + self.fnd + 'Ori_ZY.jpg',bbox_inches='tight',dpi=150)
        return imgfit
    
    def GaussianFit2D(self, img, init_params):
        out, fopt, iter1, iter2, warnflag = fmin(self._gausserr2D, init_params, args=(img,), full_output=True)
        # print('out_xy')
        # print(out)
        return out
       
    def _gausserr2D(self, p, img):
        '''
        (p, zshape)
        p=(amplitude, x_center, y_center, sigma, background)
        '''
        nx, ny = img.shape
        x = np.arange(nx)
        y = np.arange(ny)
        [xx, yy] = np.meshgrid(x,y, indexing = 'ij')
        imgfit = p[0]*np.exp(-0.5*(((xx-p[1])**2 + (yy-p[2])**2)/p[3]**2))+p[4]
        err = ((img-imgfit)**2).sum()/(img**2).sum()
        return err
    
       
    def _gausserr2DZ(self, p, img):
        '''
        (p, zshape)
        p=(amplitude, z_center, y_center, sigmaz, sigmay, background)
        '''
        nz, ny = img.shape
        z = np.arange(nz)
        y = np.arange(ny)
        [zz, yy] = np.meshgrid(z,y, indexing = 'ij')
        imgfit = p[0]*np.exp(-0.5*((((zz-p[1])**2)/p[3]**2) + (((yy-p[2])**2)/p[4]**2)))+p[5]
        err = ((img-imgfit)**2).sum()/(img**2).sum()
        return err
    
    def GaussianFit2DZ(self, img, init_params):
        out, fopt, iter1, iter2, warnflag = fmin(self._gausserr2DZ, init_params, args=(img,), full_output=True)
        # print('out_zy')
        # print(out)
        return out
    
    def gauss2DplotZ(self, p, img):
        '''
        (p, zshape)
        p=(amplitude, z_center, y_center, sigmaz, sigmay, background)'''
        nz, ny = img.shape
        z = np.arange(nz)
        y = np.arange(ny)
        [zz, yy] = np.meshgrid(z,y,indexing = 'ij')
        imgfit = p[0]*np.exp(-0.5*((((zz-p[1])**2)/p[3]**2) + (((yy-p[2])**2)/p[4]**2)))+p[5]
        fig = plt.figure(figsize=(7,7))
        #plt.imshow(imgfit)
        fig.savefig(self.path + '/' + self.fnd + 'Fit_ZY.jpg',bbox_inches='tight',dpi=150)
        fig = plt.figure(figsize=(7,7))
        #plt.imshow(img)
        fig.savefig(self.path + '/' + self.fnd + 'Ori_ZY.jpg',bbox_inches='tight',dpi=150)
        return imgfit
    
    
    def zslides_trans(self, img):
        nz, nx, ny = img.shape
        z_y = np.zeros((nz,nx))
        for i in range(nz):
            for j in range(ny):
                z_y[i,j] = max(img[i,:,j])
        # plt.imshow(z_y)
        return z_y
    
    def save_fit_data(self, p, img):
        '''
        (p, zshape)
        p=(amplitude, z_center, x_center, y_center, sigma_xy, sigma_z, background)
        '''
        # print('f_p')
        tf.imwrite(self.path + '/' + self.fnd + 'ori_holoimages.tif', np.abs(img))
        nz, nx, ny = img.shape
        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)
        [zz, xx, yy] = np.meshgrid(z, x, y, indexing = 'ij')
        imgfit = p[0]*np.exp(-0.5*((((((xx-p[2])**2)/p[4]**2) + (((yy-p[3])**2)/p[4]**2) + (((zz-p[1])**2)/p[5]**2) ))))+p[6]
        tf.imwrite(self.path + '/' + self.fnd + 'fit_holoimages.tif', np.abs(imgfit))
        
        ''''plot fit'''
        # #plot fit data
        # plt.figure(figsize = (8,8))
        # ax = plt.axes(projection ="3d")
        # [z1, x1, y1] = np.meshgrid(z, x, y, indexing = 'ij')
        # for i in range(nz):
        #     for j in range(nx):
        #         for k in range(ny):
        #             if imgfit[i,j,k] < 639:
        #                imgfit[i,j,k] = np.nan
        # imgfit2 = np.zeros((nx,ny,nz))
        # for i in range(nz):
        #     imgfit2[:,:,i] = imgfit[i,:,:] 
        # [x1, y1, z1] = np.meshgrid(x, y, z, indexing = 'ij')
        # ax.scatter3D(x1, y1, z1, c=imgfit2, marker='.')
        # plt.show()
        
        # #plot original data
        # [z2, x2, y2] = np.meshgrid(z, x, y, indexing = 'ij')
        # for i in range(nz):
        #     for j in range(nx):
        #         for k in range(ny):
        #             if img[i,j,k] < 639:
        #               img[i,j,k] = np.nan
        # img2 = np.zeros((nx,ny,nz))
        # for i in range(nz):
        #     img2[:,:,i] = img[i,:,:] 
        # [x2, y2, z2] = np.meshgrid(x, y, z, indexing = 'ij')
        # plt.figure(figsize = (8,8))
        # ax = plt.axes(projection ="3d")
        # ax.scatter3D(x2, y2, z2, c=img2, marker='.')
        # plt.show()

        





if __name__ == '__main__':
    sx = 0
    sy = 0
    iteration = 100
    t = hologram_simulation()
    z_s = 3.000
    
    f_slm2 = np.arange(380, 6000, 100) 
    t.Get_STD_vs_R_p(z_s,iteration,f_slm2,sx,sy)

        

