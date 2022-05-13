#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cpnest.model
import numpy as np
import BANG.data as d
import math
from numba import cuda
import time
from BANG.model_creation import model

def pp(l,r):
    return -np.sum(np.log10(r-l))

class GalaxyModel(model,cpnest.model.Model):

    def __init__(self,
                device,
                gpu_device,
                fast_math,
                include,
                bulge_type,
                halo_type,
                N,
                J,
                K,
                x_g,
                y_g,
                ibrightness_data,
                v_data,
                sigma_data,
                ML_data,
                ibrightness_error,
                v_error,
                sigma_error,
                ML_error,
                kin_psf,
                pho_psf,
                indexing,
                goodpix,
                variable_quantities,    # this is a dictionary whose keys are the names and whose values are the bounds
                fixed_quantities,       # this is a dictionary whose keys are the names and whose values are the fixed values
                rho_crit):             
        ''' 
        A class for representing all models
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------                                           
            Attributes:
                    device : str
                        Type of device (gpu/cpu) 
                    gpu_device : int
                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
                    fast_math: Bool
                        Whether to use fastmath operation of not
                    include: 1D arr Bool
                        include[0] if True BH is present if false without BH
                        include[1] if True Bulge is present if false without Bulge
                        include[2] if True First disc is present if false without First disc
                        include[3] if True Second disc is present if false without Second disc
                        include[4] if True Halo is present if false without Halo
                    bulge_type: str
                        Whether to use Herquinst/Jaffe/Dehnen (Dehnen is recommended since it includes Herquinst and Jaffe by default)
                    halo_type: str
                        Herquinst of NFW
                    N: int
                        Number of pixels on x-direction
                    J: int
                        Number of pixels on y-direction
                    K: int
                        Number of pixels considered for psf convolution (computed from in data.py)
                    x_g: 1D arr float32
                        x positions of the grid in kpc (computed from in data.py)
                    y_g: 1D arr float32
                        y positions of the grid in kpc (computed from in data.py)
                    ibrightness_data: 1D arr float32
                        brightness data in log10 (use np.nan as masking)
                    v_data: 1D arr float32
                        los velocity data (use np.nan as masking)
                    sigma_data: 1D arr float32
                        los velocity dispersion data (use np.nan as masking)
                    ML_data: 1D arr float32
                        M/L data (use np.nan as masking)
                    ibrightness_error: 1D arr float32
                        brightness data error in log10 (use np.nan as masking)
                    v_error: 1D arr float32
                        los velocity data error (use np.nan as masking)
                    sigma_error: 1D arr float32
                        los velocity dispersion data error (use np.nan as masking)
                    ML_error: 1D arr float32
                        M/L data error (use np.nan as masking)
                    kin_psf: 1D arr float32
                        weights of the kinematic psf (computed from in data.py)
                    pho_psf: 1D arr float32
                        weights of the photometric psf (computed from in data.py)
                    indexing: 1D arr int16
                        indexes useful for the psf convolution
                    goodpix: 1D arr float32
                        if goodpix[j] == 0. then pixel "j" is not used in the analysis
                    rho_crit: float32
                        cosmological critical density (3H^2(z)/8piG)
                variable_quantities: dict
                        dictionary the keys are the names of the variable and the values are their bounds
                fixed_quantities: dict       
                        dictionary the keys are the names and the values are the fixed parameter that do not
                        vary during the fit.
            Methods:
                    log_prior(x):
                        return the logarithm of the prior
                    log_likelihood(x):
                        return the log likelihood
        '''
        super(GalaxyModel,self).__init__(device,
                                         gpu_device,
                                         fast_math,
                                         include,
                                         bulge_type,
                                         halo_type,
                                         N,
                                         J,
                                         K,
                                         x_g,
                                         y_g,
                                         ibrightness_data,
                                         v_data,
                                         sigma_data,
                                         ML_data,
                                         ibrightness_error,
                                         v_error,
                                         sigma_error,
                                         ML_error,
                                         kin_psf,
                                         pho_psf,
                                         indexing,
                                         goodpix,
                                         rho_crit)
        

        self.fixed_quantities    = fixed_quantities
        self.variable_quantities = variable_quantities
        self.bounds      = np.array(list(self.variable_quantities.values()))
        self.names       = list(self.variable_quantities.keys())
        self.names_fixed = list(fixed_quantities.keys())
        self.value       = list(fixed_quantities.values())

        # need to copy the dictionary before popping inclination
        # after popping copy_varible will no more contain the inclinatio key
        self.copy_variable = self.variable_quantities.copy()
        if 'inclination' in self.variable_quantities:
            self.incl_bounds   = self.copy_variable.pop('inclination')
        
        # if inclination is in the variable parameters then my_bounds and my_names do 
        # not include inclination, otherwise my_bounds and my_names are equal to bounds and names
        self.my_bounds   = np.array(list(self.copy_variable.values())) 
        self.my_names    = np.array(list(self.copy_variable.keys()))
        self.rho_crit = rho_crit

        self.my_model = model(device,
                              gpu_device,
                              fast_math,
                              include,
                              bulge_type,
                              halo_type,
                              N,
                              J,
                              K,
                              x_g,
                              y_g,
                              ibrightness_data,
                              v_data,
                              sigma_data,
                              ML_data,
                              ibrightness_error,
                              v_error,
                              sigma_error,
                              ML_error,
                              kin_psf,
                              pho_psf,
                              indexing,
                              goodpix,
                              rho_crit)

    def log_prior(self,x):
        # | command is for merging two dictionaries
        self.all_params = {**x , **self.fixed_quantities}
        for i in range(len(self.names)):
            if x[self.names[i]]>self.bounds[i,1] or x[self.names[i]]<self.bounds[i,0]:
                return -np.inf
        if 'log10_bulge_radius' in self.all_params  and 'log10_disc2_radius' in self.all_params: 
            if 10**self.all_params['log10_bulge_radius'] > 10**self.all_params['log10_disc2_radius']: 
                return -np.inf  
        
        if 'log10_disc1_radius' in self.all_params and 'log10_disc2_radius' in self.all_params:
            if 10**self.all_params['log10_disc1_radius'] > 10**self.all_params['log10_disc2_radius']: 
                return -np.inf   

        if "concentration" in self.all_params:
            if 'log10_disc2_mass' in self.all_params: 
                if 'log10_bulge_mass' in self.all_params:
                    M_halo = 10**self.all_params["log10_halo_fraction"]*(\
                             10**self.all_params["log10_disc2_mass"]+\
                             10**self.all_params["log10_disc1_mass"]+\
                             10**self.all_params["log10_bulge_mass"]) 
                else:
                    M_halo = 10**self.all_params["log10_halo_fraction"]*(\
                             10**self.all_params["log10_disc2_mass"]+\
                             10**self.all_params["log10_disc1_mass"])

                a = ((3.*M_halo/(800.*np.pi*self.rho_crit))**(1./3.))/self.all_params["concentration"]
                if a < 10**self.all_params["log10_disc2_radius"]:
                    return -np.inf 
            
            elif 'log10_disc1_mass' in self.all_params: 
                M_halo = 10**self.all_params["log10_halo_fraction"]*(\
                         10**self.all_params["log10_disc1_mass"]+\
                         10**self.all_params["log10_bulge_mass"]) 
                a = ((3.*M_halo/(800.*np.pi*self.rho_crit))**(1./3.))/self.all_params["concentration"]
                if a < 10**self.all_params["log10_disc1_radius"]:
                    return -np.inf 
                    
        elif "log10_halo_radius" in self.all_params:
            if 'log10_disc2_mass' in self.all_params: 
                if 10**self.all_params["log10_halo_radius"] < 10**self.all_params["log10_disc2_radius"]:
                    return -np.inf 
            elif 'log10_disc1_mass' in self.all_params: 
                if 10**self.all_params["log10_halo_radius"] < 10**self.all_params["log10_disc1_radius"]:
                    return -np.inf 
        
        #if 'log10_disc2_radius' and 'log10_halo_radius' in self.all_params:
        #    if 10*10**self.all_params['log10_disc2_radius'] > 10**self.all_params['log10_halo_radius']: 
        #        return -np.inf  
        
        #if 'ML_b' in self.all_params and 'ML_d1' in self.all_params:
        #    if self.all_params['ML_d1'] > self.all_params['ML_b']: 
        #        return -np.inf  
        #if 'ML_b' in self.all_params and 'ML_d2' in self.all_params:
        #    if self.all_params['ML_d2'] > self.all_params['ML_b']: 
        #        return -np.inf 
        #if 'ML_d1' in self.all_params and 'ML_d2' in self.all_params:
        #    if self.all_params['ML_d2'] > self.all_params['ML_d1']: 
        #        return -np.inf 

        p1 = pp(self.my_bounds[:,0],self.my_bounds[:,1])
        if 'inclination' in self.variable_quantities:
            incl_prior = np.log(np.sin(x['inclination'])/(np.cos(self.incl_bounds[0])-np.cos(self.incl_bounds[1])))
        else:
            incl_prior = 0.
        return incl_prior+p1
    
    def log_likelihood(self, x):
        logL = self.my_model.likelihood(self.all_params)
        return logL

if __name__=="__main__":

    import data as d
    import matplotlib.pyplot as plt 
    from astropy.cosmology import FlatLambdaCDM



    # Declare options (you will specify then in the config.yaml file)
    fast_math  = True
    bulge_type = 'dehnen'
    halo_type  ='NFW'
    include    = [False,True,True,True,True]
    gpu_device = 0
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    H_z   = cosmo.H(0.03).value # this is in km/Mpc/s
    H_z   = H_z*1e5/(1e6*3.086e18)         # this is in 1/s
    G     = 4.299e-6                       # kcp*km^2/M_sun/s^2
    G_1   = G * (1e5*1e-3*3.24078e-19)**2  # kcp^3/M_sun/s^2
    rho_crit = 3*H_z*H_z/(8*np.pi*G_1)


    # Create mock grid
    N,J = 65,65
    x = np.linspace(-15,15,N)
    y = np.linspace(-15,15,N)
    x,y = np.meshgrid((x,y),indexing = 'ij')
    x,y = x.ravel(),y.ravel()
   
    # Create grid for psf convolution 
    data_obj = d.data(x.ravel(),y.ravel())
    x_g,y_g,kin_psf,K,indexing = data_obj.refined_grid_indexing(sigma_psf=1.374,N_px=12)
    _,_,pho_psf,_,_ = data_obj.refined_grid_indexing(sigma_psf=1.374,N_px=12)
    x_g = np.round(x_g,decimals=5)
    y_g = np.round(y_g,decimals=5)

    # Create fake data    
    ibrightness_data = np.zeros_like(x)
    v_data = np.zeros_like(x)
    sigma_data = np.zeros_like(x)
    ML_data = np.zeros_like(x)
    ibrightness_error = np.zeros_like(x)
    v_error = np.zeros_like(x)
    sigma_error = np.zeros_like(x)
    ML_error = np.zeros_like(x)
    goodpix = np.ones_like(x)

    variable_quantities = {'log10_bulge_mass':[8.5,12.],
                           'log10_bulge_radius':[-2,1.5],
                           'gamma':[0,2.40],
                           "ML_b":[0.1,20],
                           "log10_disc1_mass":[8.5,11.5],
                           "log10_disc1_radius":[-2,1.5],
                           "ML_d1":[0.1,20],
                           "k1":[0.01,1],
                           "log10_disc2_mass":[8.5,11.5],
                           "log10_disc2_radius":[-2,1.5],
                           "ML_d2":[0.1,20],
                           "k2":[0.01,1],
                            'log10_halo_fraction' :[1.,2.7] ,
                            'concentration':[5,15],
                           'x0': [-1.,1.],
                           'y0':[-1.,1.],
                           'orientation':[-3.141592653589793,3.141592653589793],
                           'inclination':[0.1,1.3]
                           }

    fixed_quantities = {'sys_rho':-10}

    M = GalaxyModel(device,
                    gpu_device,
                    fast_math,
                    include,
                    bulge_type,
                    halo_type,
                    N,
                    J,
                    K,
                    x_g,
                    y_g,
                    ibrightness_data,
                    v_data,
                    sigma_data,
                    ML_data,
                    ibrightness_error,
                    v_error,
                    sigma_error,
                    ML_error,
                    kin_psf,
                    pho_psf,
                    indexing,
                    goodpix,
                    variable_quantities,    # this is a dictionary whose keys are the names and whose values are the bounds
                    fixed_quantities,       # this is a dictionary whose keys are the names and whose values are the fixed values
                    rho_crit)

    #import multiprocessing as mp
    #mp.set_start_method('spawn')
    work=cpnest.CPNest(M,
                       verbose  = 2,
                       poolsize = 100,
                       nthreads = 1 ,
                       nlive    = 1000,
                       maxmcmc  = 100,
                       output   = './run1',
                       nhamiltonian = 0,
                       nslice   = 0,
                       resume   = 1)
    work.run()

    work.get_posterior_samples()
    work.get_nested_samples()
