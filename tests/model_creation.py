#THE STRUCTURE CAN be changed, all model on gpu can be easily put togheter.
# idem for CPU

import math
from numba import cuda
import numpy as np   
#import torch   


'''
Models class, it contains all the possible model combinations i.e. :
        - GPU with fastmath:
                - Bulge+Black Hole+Halo ("model_gpu_B_BH")                   UNSTABLE with new convolution
                - Bulge+Disc+Black Hole+Halo ("model_gpu_BD_BH")             UNSTABLE with new convolution
                - Bulge+Disc+Halo ("model_gpu_BD")                           STABLE
                - Disc+Disc+Halo ("model_gpu_DD")                            STABLE
                - Bulge+Disc+Disc+Black Hole+Halo ("model_gpu_BH")           UNSTABLE with new convolution
                - Bulge+Disc+Disc+Halo ("model_gpu")                         STABLE
        - GPU without fastmath:
                - Bulge+Black Hole+Halo ("model_gpu_B_BH")                   UNSTABLE with new convolution
                - Bulge+Disc+Black Hole+Halo ("model_gpu_BD_BH")             UNSTABLE with new convolution
                - Bulge+Disc+Halo ("model_gpu_BD")                           STABLE 
                - Disc+Disc+Halo ("model_gpu_DD")                            STABLE 
                - Bulge+Disc+Disc+Black Hole+Halo ("model_gpu_BH")           UNSTABLE with new convolution
                - Bulge+Disc+Disc+Halo ("model_gpu")                         STABLE 
        - CPU:
                - Bulge+Black Hole+Halo ("model_cpu_B_BH")                   UNSTABLE with new convolution
                - Bulge+Disc+Black Hole+Halo ("model_cpu_BD_BH")             UNSTABLE with new convolution
                - Bulge+Disc+Halo ("model_cpu_BD")                           UNSTABLE with new convolution
                - Bulge+Disc+Disc+Black Hole+Halo ("model_cpu_BH")           UNSTABLE with new convolution
                - Bulge+Disc+Disc+Halo ("model_cpu")                         UNSTABLE with new convolution
        - GPU + Super Resolution:
                - Bulge+Disc+Disc+Halo ("model_superresolution_gpu")         UNSTABLE with new convolution
        - CPU + Super Resolution:  
                - Bulge+Disc+Disc+Halo ("model_superresolution_cpu")         UNSTABLE with new convolution     
'''




class model_gpu_B_BH():
    def __init__(self,
                 fast_math,
                 bulge_type,
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
                 psf):

    # PROBABLY IT CAN BE UNIFIED WITH model_gpu
        ''' 
        Model with only bulge and BH, halo can be defined but is useless
        '''
        
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data                  = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error                 = ML_error.copy(order='C')
        self.psf                      = psf.copy(order='C')
        self.bulge_type               = bulge_type
        self.fast_math                = fast_math

        self.counter = 0


    def _function_definitionBBH(self):
        if self.fast_math == True:
            if self.bulge_type=='hernquist':
                from utils_numba_1D_32bit import herquinst_rho
                from utils_numba_1D_32bit_no_proxy import herquinst_sigma_BH
                from utils_numba_1D_32bit_no_proxy import all_Xdot
                from utils_numba_1D_32bit_no_proxy import Y1
                from utils_numba_1D_32bit_no_proxy import Y2
                self._bulge_rho   = herquinst_rho 
                self._bulge_sigma = herquinst_sigma_BH 
                self._all_Xdot    = all_Xdot
                self._Y1          = Y1
                self._Y2          = Y2
                print('Warning bulge dispersion has not yet be implemented in fast math mode')
            elif self.bulge_type=='jaffe':
                print('Error Jaffe with BH has not be implemented')
                exit()
            else:
                print('Error')
                exit()

            from utils_numba_1D_32bit import avg_rho_v_only_bulge
            from utils_numba_1D_32bit import avg_LM_only_bulge
            from utils_numba_1D_32bit import sum_likelihood
            from utils_numba_1D_32bit import coordinate_transformation
            from utils_numba_1D_32bit import Xfunction
        else:
            if self.bulge_type=='hernquist':
                from utils_numba_1D_32bit_no_proxy import herquinst_rho
                from utils_numba_1D_32bit_no_proxy import herquinst_sigma_BH
                from utils_numba_1D_32bit_no_proxy import all_Xdot
                from utils_numba_1D_32bit_no_proxy import Y1
                from utils_numba_1D_32bit_no_proxy import Y2
                self._bulge_rho   = herquinst_rho 
                self._bulge_sigma = herquinst_sigma_BH 
                self._all_Xdot    = all_Xdot
                self._Y1          = Y1
                self._Y2          = Y2
            elif self.bulge_type=='jaffe':
                print('Error Jaffe with BH has not be implemented')
                exit()
            else:
                print('Error')
                exit()

            from utils_numba_1D_32bit_no_proxy import avg_rho_v_only_bulge
            from utils_numba_1D_32bit_no_proxy import avg_LM_only_bulge
            from utils_numba_1D_32bit_no_proxy import sum_likelihood
            from utils_numba_1D_32bit_no_proxy import coordinate_transformation
            from utils_numba_1D_32bit_no_proxy import Xfunction

       
        self._coordinate_transformation = coordinate_transformation
        self._Xfunction                 = Xfunction
        self._avg_rho_v                 = avg_rho_v_only_bulge
        self._avg_LM                    = avg_LM_only_bulge
        self._sum_likelihood            = sum_likelihood

    def _init_grid(self):

        Nthreads_1 = 64
        Nthreads_3 = 64
        Nblocks_1 = math.ceil(self.N*self.J*self.K/Nthreads_1)
        Nblocks_3 = math.ceil(self.N*self.J/Nthreads_3)

        self.threadsperblock_1 = (Nthreads_1)
        self.threadsperblock_3 = (Nthreads_3)
        self.blockpergrid_1 = (Nblocks_1)
        self.blockpergrid_3 = (Nblocks_3)

    def _load_to_deviceBBH(self):
        self.x_device = cuda.to_device(np.float32(self.x_g))
        self.y_device = cuda.to_device(np.float32(self.y_g))
        
        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)


        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
    
    def _init_data(self):
        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
 


    def likelihood(self,
                   all_params,
                   *args):
        ''' 
        '''
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Rb             = all_params['log10_bulge_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        sys_rho        = all_params['sys_rho']

        if self.counter==0:
            self._function_definitionBBH()
            self._init_grid()
            self._load_to_deviceBBH()
            self._init_data()
            self.counter +=1
        else: 
            pass

        Mbh          = 10**Mbh
        Mb           = 10**Mb
        Rb           = 10**Rb    
        sys_rho      = 10**sys_rho

        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)


        self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device)
        self._all_Xdot[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device,
                                                                   self.X1_dot_device,
                                                                   self.X2_dot_device,
                                                                   self.X3_dot_device)


        self._Y1[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.Y_1_device)

        self._Y2[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.X2_dot_device,
                                                             self.X3_dot_device,
                                                             self.Y_2_device)

        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                                   Rb,
                                                                   self.r_proj_device,
                                                                   self.all_rhoB_device,
                                                                   self.Xs_device)
        # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mbh,
                                                                      Mb,
                                                                      Rb,
                                                                      self.r_proj_device,
                                                                      self.Y_1_device,
                                                                      self.Y_2_device,
                                                                      self.all_rhoB_device,
                                                                      self.all_sigma2_device)


        #  RHO, V E SIGMA with psf
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_sigma2_device,
                                                                   self.psf_device,
                                                                   1.0/ML_b,
                                                                   self.rho_avg_device,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device)


        self._avg_LM[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                self.psf_device,
                                                                1.0/ML_b,
                                                                self.LM_avg_device)


        self._sum_likelihood[self.blockpergrid_3,self.threadsperblock_3](self.rho_avg_device,
                                                                        self.v_avg_device,
                                                                        self.sigma_avg_device,
                                                                        self.LM_avg_device,
                                                                        self.ibrightness_data_device,
                                                                        self.v_data_device,
                                                                        self.sigma_data_device,
                                                                        self.ML_data_device,
                                                                        self.ibrightness_error_device,
                                                                        self.v_error_device,
                                                                        self.sigma_error_device,
                                                                        self.ML_error_device,
                                                                        sys_rho,
                                                                        self.lk_device)

        lk = self.lk_device.copy_to_host()


        return lk[0]



    def model(self,
              all_params,
              *args):
        ''' 
        Returns:
        '''
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Rb             = all_params['log10_bulge_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        sys_rho        = all_params['sys_rho']

        if self.counter==0:
            self._function_definitionBBH()
            self._init_grid()
            self._load_to_deviceBBH()
            self._init_data()
            self.counter +=1
        else: 
            pass

        Mbh          = 10**Mbh
        Mb           = 10**Mb
        Rb           = 10**Rb    
        sys_rho      = 10**sys_rho

        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)


        self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device)
        self._all_Xdot[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device,
                                                                   self.X1_dot_device,
                                                                   self.X2_dot_device,
                                                                   self.X3_dot_device)


        self._Y1[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.Y_1_device)

        self._Y2[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.X2_dot_device,
                                                             self.X3_dot_device,
                                                             self.Y_2_device)

        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                                   Rb,
                                                                   self.r_proj_device,
                                                                   self.all_rhoB_device,
                                                                   self.Xs_device)
        # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mbh,
                                                                      Mb,
                                                                      Rb,
                                                                      self.r_proj_device,
                                                                      self.Y_1_device,
                                                                      self.Y_2_device,
                                                                      self.all_rhoB_device,
                                                                      self.all_sigma2_device)


        #  RHO, V E SIGMA with psf
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_sigma2_device,
                                                                   self.psf_device,
                                                                   1.0/ML_b,
                                                                   self.rho_avg_device,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device)


        self._avg_LM[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                self.psf_device,
                                                                1.0/ML_b,
                                                                self.LM_avg_device)

        rho   = self.rho_avg_device.copy_to_host()
        v     = self.v_avg_device.copy_to_host()
        sigma = self.sigma_avg_device.copy_to_host()
        LM    = self.LM_avg_device.copy_to_host()

        return rho,v,sigma,LM


class model_gpu_BD_BH():
    def __init__(self,
                 fast_math,
                 bulge_type,
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
                 psf):

    # PROBABLY IT CAN BE UNIFIED WITH model_gpu
        ''' 
        '''
        
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data                  = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error                 = ML_error.copy(order='C')
        self.psf                      = psf.copy(order='C')
        self.bulge_type               = bulge_type
        self.fast_math                = fast_math

        self.counter = 0


    def _function_definitionBDBH(self):
        if self.fast_math == True:
            if self.bulge_type=='hernquist':
                from utils_numba_1D_32bit import herquinst_rho
                from utils_numba_1D_32bit_no_proxy import herquinst_sigma_BH
                from utils_numba_1D_32bit import v_H
                from utils_numba_1D_32bit_no_proxy import all_Xdot
                from utils_numba_1D_32bit_no_proxy import Y1
                from utils_numba_1D_32bit_no_proxy import Y2
                from utils_numba_1D_32bit_no_proxy import v_tot_BD_BH
                self._bulge_rho   = herquinst_rho 
                self._bulge_sigma = herquinst_sigma_BH 
                self._bulge_vel   = v_H 
                self._all_Xdot    = all_Xdot
                self._Y1          = Y1
                self._Y2          = Y2
                print('Warning bulge dispersion has not yet be implemented in fast math mode')
            elif self.bulge_type=='jaffe':
                print('Error Jaffe with BH has not be implemented')
                exit()
            else:
                print('Error')
                exit()

            from utils_numba_1D_32bit import rho_D
            from utils_numba_1D_32bit import v_D
            from utils_numba_1D_32bit import v_H
            from utils_numba_1D_32bit import avg_rho_v_2_vis_components
            from utils_numba_1D_32bit import avg_LM_2_vis_components
            from utils_numba_1D_32bit import sum_likelihood
            from utils_numba_1D_32bit import coordinate_transformation
            from utils_numba_1D_32bit import Xfunction
            from utils_numba_1D_32bit_no_proxy import v_BH
        else:
            if self.bulge_type=='hernquist':
                from utils_numba_1D_32bit_no_proxy import herquinst_rho
                from utils_numba_1D_32bit_no_proxy import herquinst_sigma_BH
                from utils_numba_1D_32bit_no_proxy import v_H
                from utils_numba_1D_32bit_no_proxy import all_Xdot
                from utils_numba_1D_32bit_no_proxy import Y1
                from utils_numba_1D_32bit_no_proxy import Y2
                self._bulge_rho   = herquinst_rho 
                self._bulge_sigma = herquinst_sigma_BH 
                self._bulge_vel   = v_H 
                self._all_Xdot    = all_Xdot
                self._Y1          = Y1
                self._Y2          = Y2
            elif self.bulge_type=='jaffe':
                print('Error Jaffe with BH has not be implemented')
                exit()
            else:
                print('Error')
                exit()

            from utils_numba_1D_32bit_no_proxy import rho_D
            from utils_numba_1D_32bit_no_proxy import v_D
            from utils_numba_1D_32bit_no_proxy import v_H
            from utils_numba_1D_32bit_no_proxy import v_tot_BD_BH
            from utils_numba_1D_32bit_no_proxy import avg_rho_v_2_vis_components
            from utils_numba_1D_32bit_no_proxy import avg_LM_2_vis_components
            from utils_numba_1D_32bit_no_proxy import sum_likelihood
            from utils_numba_1D_32bit_no_proxy import coordinate_transformation
            from utils_numba_1D_32bit_no_proxy import Xfunction
            from utils_numba_1D_32bit_no_proxy import v_BH

       
        self._coordinate_transformation = coordinate_transformation
        self._Xfunction                 = Xfunction
        self._rho_D                     = rho_D
        self._v_D                       = v_D
        self._halo_vel                  = v_H
        self._v_tot                     = v_tot_BD_BH
        self._avg_rho_v                 = avg_rho_v_2_vis_components
        self._avg_LM                    = avg_LM_2_vis_components
        self._sum_likelihood            = sum_likelihood
        self._v_BH                      = v_BH

    def _init_grid(self):

        Nthreads_1 = 64
        Nthreads_3 = 64
        Nblocks_1 = math.ceil(self.N*self.J*self.K/Nthreads_1)
        Nblocks_3 = math.ceil(self.N*self.J/Nthreads_3)

        self.threadsperblock_1 = (Nthreads_1)
        self.threadsperblock_3 = (Nthreads_3)
        self.blockpergrid_1 = (Nblocks_1)
        self.blockpergrid_3 = (Nblocks_3)

    def _load_to_deviceBDBH(self):
        self.x_device = cuda.to_device(np.float32(self.x_g))
        self.y_device = cuda.to_device(np.float32(self.y_g))
        
        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_rhoD1_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_v_device         = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_v2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.v_exp1_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.v_bulge_device       = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.v_halo_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.v_BH_device          = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)


        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
    
    def _init_data(self):
        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
 


    def likelihood(self,
                   all_params,
                   *args):
        ''' 
        '''
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Mh             = all_params['log10_halo_mass']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rh             = all_params['log10_halo_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        k1             = all_params['k1']
        sys_rho        = all_params['sys_rho']

        if self.counter==0:
            self._function_definitionBDBH()
            self._init_grid()
            self._load_to_deviceBDBH()
            self._init_data()
            self.counter +=1
        else: 
            pass

        Mbh          = 10**Mbh
        Mb           = 10**Mb
        Md1          = 10**Md1    
        Mh           = 10**Mh    
        Rb           = 10**Rb    
        Rd1          = 10**Rd1    
        Rh           = 10**Rh
        sys_rho      = 10**sys_rho

        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)


        self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device)
        self._all_Xdot[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device,
                                                                   self.X1_dot_device,
                                                                   self.X2_dot_device,
                                                                   self.X3_dot_device)


        self._Y1[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.Y_1_device)

        self._Y2[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.X2_dot_device,
                                                             self.X3_dot_device,
                                                             self.Y_2_device)

        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                                   Rb,
                                                                   self.r_proj_device,
                                                                   self.all_rhoB_device,
                                                                   self.Xs_device)
        # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mbh,
                                                                      Mb,
                                                                      Rb,
                                                                      self.r_proj_device,
                                                                      self.Y_1_device,
                                                                      self.Y_2_device,
                                                                      self.all_rhoB_device,
                                                                      self.all_sigma2_device)

        ## DISC SURFACE DENSITY
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                               Rd1,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD1_device)

        ## DISC VELOCITY 
        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](Mh,
                                                             Rh,
                                                             self.r_true_device,
                                                             self.v_halo_device)
        self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                            Rb,
                                                            self.r_true_device,
                                                            self.v_bulge_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                             Rd1,
                                                             self.r_true_device,
                                                             self.v_exp1_device)
        self._v_BH[self.blockpergrid_1,self.threadsperblock_1](Mbh,
                                                               self.r_true_device,
                                                               self.v_BH_device)
        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
                                                                self.v_bulge_device,
                                                                self.v_halo_device,
                                                                self.v_BH_device,
                                                                incl,
                                                                self.phi_device,
                                                                self.all_v_device,
                                                                self.all_v2_device)

        #  RHO, V E SIGMA with psf
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.psf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   k1,
                                                                   self.rho_avg_device,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device)


        self._avg_LM[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                self.all_rhoD1_device,
                                                                self.psf_device,
                                                                1.0/ML_b,
                                                                1.0/ML_d1,
                                                                self.LM_avg_device)


        self._sum_likelihood[self.blockpergrid_3,self.threadsperblock_3](self.rho_avg_device,
                                                                        self.v_avg_device,
                                                                        self.sigma_avg_device,
                                                                        self.LM_avg_device,
                                                                        self.ibrightness_data_device,
                                                                        self.v_data_device,
                                                                        self.sigma_data_device,
                                                                        self.ML_data_device,
                                                                        self.ibrightness_error_device,
                                                                        self.v_error_device,
                                                                        self.sigma_error_device,
                                                                        self.ML_error_device,
                                                                        sys_rho,
                                                                        self.lk_device)

        lk = self.lk_device.copy_to_host()


        return lk[0]



    def model(self,
              all_params,
              *args):
        ''' 
        '''
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Mh             = all_params['log10_halo_mass']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rh             = all_params['log10_halo_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        k1             = all_params['k1']
        if self.counter==0:
            self._function_definitionBDBH()
            self._init_grid()
            self._load_to_deviceBDBH()
            self._init_data()
            self.counter +=1
        else: 
            pass

        Mbh          = 10**Mbh
        Mb           = 10**Mb
        Md1          = 10**Md1    
        Mh           = 10**Mh    
        Rb           = 10**Rb    
        Rd1          = 10**Rd1    
        Rh           = 10**Rh
        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)


        self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device)
        self._all_Xdot[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device,
                                                                   self.X1_dot_device,
                                                                   self.X2_dot_device,
                                                                   self.X3_dot_device)
        self._Y1[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.Y_1_device)
        self._Y2[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.X2_dot_device,
                                                             self.X3_dot_device,
                                                             self.Y_2_device)
        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                                    Rb,
                                                                    self.r_proj_device,
                                                                    self.all_rhoB_device,
                                                                    self.Xs_device)
        # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mbh,
                                                                      Mb,
                                                                      Rb,
                                                                      self.r_proj_device,
                                                                      self.Y_1_device,
                                                                      self.Y_2_device,
                                                                      self.all_rhoB_device,
                                                                      self.all_sigma2_device)

        ## DISC SURFACE DENSITY
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                               Rd1,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD1_device)

        ## DISC VELOCITY 
        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](Mh,
                                                             Rh,
                                                             self.r_true_device,
                                                             self.v_halo_device)
        self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                            Rb,
                                                            self.r_true_device,
                                                            self.v_bulge_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                             Rd1,
                                                             self.r_true_device,
                                                             self.v_exp1_device)
        self._v_BH[self.blockpergrid_1,self.threadsperblock_1](Mbh,
                                                               self.r_true_device,
                                                               self.v_BH_device)
        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
                                                                self.v_bulge_device,
                                                                self.v_halo_device,
                                                                self.v_BH_device,
                                                                incl,
                                                                self.phi_device,
                                                                self.all_v_device,
                                                                self.all_v2_device)

        #  RHO, V E SIGMA with psf
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.psf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   k1,
                                                                   self.rho_avg_device,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device)


        self._avg_LM[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                self.all_rhoD1_device,
                                                                self.psf_device,
                                                                1.0/ML_b,
                                                                1.0/ML_d1,
                                                                self.LM_avg_device)

        rho   = self.rho_avg_device.copy_to_host()
        v     = self.v_avg_device.copy_to_host()
        sigma = self.sigma_avg_device.copy_to_host()
        LM    = self.LM_avg_device.copy_to_host()

        return rho,v,sigma,LM


class model_gpu_BD():
    def __init__(self,
                 gpu_device,
                 fast_math,
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
                 rho_crit):
        ''' 
        A class for representing a "Bulge + Disc + Halo" model on GPU
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------                                           
            Attributes:
                    gpu_device : int
                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
                    fast_math: Bool
                        Whether to use fastmath operation of not
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
            Methods:
                    _function_definition():
                        Define all functions depending on the condition above (bulge_type, halo_type ...)
                    _init_grid():
                        Compute # blocks, # theads for the Gpu
                    _load_to_device():
                        Load all data on GPU
                    _init_data():
                        Allocate memory for GPU operations
                    _load_device_model():
                        Allocate additional memory in case of model (not likelihood) call
                    _load_tabulated():
                        Load Dehnen interpolated quantities
                    likelihood(all_params,
                               *args,
                               **kwargs):
                        Compute likelihood of the model
                    model(all_params,
                               *args,
                               **kwargs):
                        Compute model all outputs
                    
        '''
        
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data                  = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error                 = ML_error.copy(order='C')
        self.kin_psf                  = kin_psf.copy(order='C')
        self.pho_psf                  = pho_psf.copy(order='C')
        self.bulge_type               = bulge_type
        self.halo_type                = halo_type
        self.fast_math                = fast_math
        self.indexing                 = indexing
        self.counter = 0
        self.size = self.x_g.shape[0]
        self.goodpix = goodpix

        self.rho_crit = rho_crit
        self.new_N,self.new_J = int(self.size**0.5),int(self.size**0.5)

        self.dx = np.abs(self.x_g[1]-self.x_g[0])

        self.central_integration_radius = self.dx/2**0.5
        flat_index = np.argmin(np.sqrt(self.x_g*self.x_g+self.y_g*self.y_g))
        self.central_i = int(flat_index/self.new_J)
        self.central_j = flat_index - self.central_i*self.new_J
        
        self.gpu_device = gpu_device

    def _function_definitionBD(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Define all functions depending on the condition above (bulge_type, halo_type ...)
            
            Parameters:

            Returns:

        '''
        if self.fast_math == True:
            if self.bulge_type=='hernquist':
                from utils_numba_1D_32bit import herquinst_rho
                from utils_numba_1D_32bit import herquinst_sigma
                from utils_numba_1D_32bit import v_H
                self._bulge_rho   = herquinst_rho 
                self._bulge_sigma = herquinst_sigma 
                self._bulge_vel   = v_H 
            elif self.bulge_type=='jaffe':
                from utils_numba_1D_32bit import jaffe_rho        
                from utils_numba_1D_32bit import jaffe_sigma  
                from utils_numba_1D_32bit import v_J
                self._bulge_rho   = jaffe_rho
                self._bulge_sigma = jaffe_sigma
                self._bulge_vel   = v_J
                print('Warning bulge dispersion has not yet be implemented in fast math mode')
            elif self.bulge_type=='dehnen':
                from utils_numba_1D_32bit import dehnen        
                #from utils_numba_1D_32bit import jaffe_sigma  
                from utils_numba_1D_32bit import v_dehnen
                self._bulge = dehnen
                self._bulge_vel = v_dehnen
            else:
                print('Error')
                exit()
            
            if self.halo_type=='hernquist':
                from utils_numba_1D_32bit import v_Halo
                self._halo_vel = v_Halo
                self.halo_radius_name = 'log10_halo_radius'
                self.halo_mass_name   = 'log10_halo_fraction'
                
            elif self.halo_type=='NFW':
                from utils_numba_1D_32bit import v_NFW
                self._halo_vel = v_NFW
                self.halo_radius_name = 'concentration'
                self.halo_mass_name   = 'log10_halo_fraction'

            from utils_numba_1D_32bit import rho_D
            from utils_numba_1D_32bit import v_D
            from utils_numba_1D_32bit import v_H
            from utils_numba_1D_32bit import avg_rho_v_2_vis_components
            from utils_numba_1D_32bit import avg_sigma_2_vis_components
            from utils_numba_1D_32bit import sum_likelihood
            from utils_numba_1D_32bit import coordinate_transformation
            from utils_numba_1D_32bit import Xfunction
            from utils_numba_1D_32bit_no_proxy import v_tot_BD

        else:
            if self.bulge_type=='hernquist':
                from utils_numba_1D_32bit_no_proxy import herquinst_rho
                from utils_numba_1D_32bit_no_proxy import herquinst_sigma
                from utils_numba_1D_32bit_no_proxy import v_H
                self._bulge_rho   = herquinst_rho 
                self._bulge_sigma = herquinst_sigma_BH 
                self._bulge_vel   = v_H 
            elif self.bulge_type=='jaffe':
                from utils_numba_1D_32bit_no_proxy import jaffe_rho        
                from utils_numba_1D_32bit_no_proxy import jaffe_sigma  
                from utils_numba_1D_32bit_no_proxy import v_J
                self._bulge_rho   = jaffe_rho
                self._bulge_sigma = jaffe_sigma
                self._bulge_vel   = v_J
            elif self.bulge_type=='dehnen':
                from utils_numba_1D_32bit import dehnen        
                from utils_numba_1D_32bit import v_dehnen
                self._bulge = dehnen
                self._bulge_vel = v_dehnen
            else:
                print('Error')
                exit()
            
            if self.halo_type=='hernquist':
                from utils_numba_1D_32bit import v_Halo
                self._halo_vel = v_Halo
                self.halo_radius_name = 'log10_halo_radius'
                self.halo_mass_name   = 'log10_halo_fraction'
                
            elif self.halo_type=='NFW':
                from utils_numba_1D_32bit import v_NFW
                self._halo_vel = v_NFW
                self.halo_radius_name = 'concentration'
                self.halo_mass_name   = 'log10_halo_fraction'
            
            from utils_numba_1D_32bit_no_proxy import rho_D
            from utils_numba_1D_32bit_no_proxy import v_D
            from utils_numba_1D_32bit_no_proxy import v_H
            from utils_numba_1D_32bit_no_proxy import v_tot_BD
            from utils_numba_1D_32bit_no_proxy import avg_rho_v_2_vis_components
            from utils_numba_1D_32bit_no_proxy import avg_LM_2_vis_components
            from utils_numba_1D_32bit_no_proxy import sum_likelihood
            from utils_numba_1D_32bit_no_proxy import coordinate_transformation
            from utils_numba_1D_32bit_no_proxy import Xfunction

       
        self._coordinate_transformation = coordinate_transformation
        self._Xfunction                 = Xfunction
        self._rho_D                     = rho_D
        self._v_D                       = v_D
        self._v_tot                     = v_tot_BD
        self._avg_rho_v                 = avg_rho_v_2_vis_components
        self._avg_sigma                 = avg_sigma_2_vis_components
        #self._avg_LM                    = avg_LM_2_vis_components
        self._sum_likelihood            = sum_likelihood

    def _init_grid(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Compute # blocks, # theads for the Gpu
        
            Parameters:

            Returns:
            
        '''
        Nthreads_1 = 64
        Nthreads_3 = 64
        Nblocks_1 = math.ceil(self.size/Nthreads_1)
        Nblocks_3 = math.ceil(self.N*self.J/Nthreads_3)

        self.threadsperblock_1 = (Nthreads_1)
        self.threadsperblock_3 = (Nthreads_3)
        self.blockpergrid_1 = (Nblocks_1)
        self.blockpergrid_3 = (Nblocks_3)

    def _load_to_deviceBD(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
         Load all data on GPU
            
            Parameters:

            Returns:
            
        '''
        self.x_device = cuda.to_device(np.float32(self.x_g))
        self.y_device = cuda.to_device(np.float32(self.y_g))
        
        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
        self.all_rhoB_device      = cuda.device_array((self.size),dtype=np.float32)
        self.all_sigma2_device    = cuda.device_array((self.size),dtype=np.float32)
        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
        self.v_bulge_device       = cuda.device_array((self.size),dtype=np.float32)
        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)


        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
    
    def _init_dataBD(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Allocate memory for GPU operations
            
            Parameters:

            Returns:
            
        '''
        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))

    def _load_device_modelBD(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Allocate additional memory in case of model (not likelihood) call

            Parameters:

            Returns:
            
        '''
        self.LM_avg_B_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_B_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_B_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_B_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
    
    def _load_tabulated(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Load Dehnen interpolated quantities

            Parameters:

            Returns:
            
        '''
        path = './dehnen/'
        s_grid        = np.load(path+'s_grid.npy')
        gamma_grid    = np.load(path+'gamma_grid.npy') 
        rho_grid      = np.load(path+'dehnen_brightness.npy')
        sigma_grid    = np.load(path+'dehnen_sigma2.npy')
        central_rho   = np.load(path+'dehnen_luminosity.npy')
        central_sigma = np.load(path+'dehnen_avg_sigma2.npy')

        self.s_grid_device        = cuda.to_device(np.float32(s_grid))
        self.gamma_grid_device    = cuda.to_device(np.float32(gamma_grid))
        self.rho_grid_device      = cuda.to_device(np.float32(rho_grid))
        self.sigma_grid_device    = cuda.to_device(np.float32(sigma_grid))
        self.central_rho_device   = cuda.to_device(np.float32(central_rho))
        self.central_sigma_device = cuda.to_device(np.float32(central_sigma))

    def likelihood(self,
                   all_params,
                   *args):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Compute likelihood of the model.
            
            Parameters:
                all_params: dict
                    Dictionary containing all the model parameters

            Returns:
                lk: float32
                    Likelihood of the model.

        '''
        if self.counter==0:
            cuda.select_device(self.gpu_device)
            self._init_grid()
            self._load_to_deviceBD()
            self._init_dataBD()
            self._function_definitionBD()
            self._load_tabulated()
            self.counter +=1
        else: 
            pass
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        k1             = all_params['k1']
        sys_rho        = all_params['sys_rho']
        f              = all_params['log10_halo_fraction']
        c              = all_params[self.halo_radius_name]
        gamma          = all_params['gamma']

        Mb           = 10**Mb
        Md1          = 10**Md1    
        Rb           = 10**Rb    
        Rd1          = 10**Rd1    
        sys_rho      = 10**sys_rho
        f            = 10**f 
        
        if self.halo_type=='NFW':
            pass
        else:
            c = 10**c
        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)

        self.central_index = np.argmin(self.r_proj_device.copy_to_host())
        #self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
        #                                                           self.r_proj_device,
        #                                                           self.Xs_device,
        #                                                           self.central_integration_radius,
        #                                                           self.central_index)

        ### BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        #self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
        #                                                           Rb,
        #                                                           self.r_proj_device,
        #                                                           self.all_rhoB_device,
        #                                                           self.Xs_device,
        #                                                           self.central_integration_radius,
        #                                                           self.central_index)
        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        #self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mb,
        #                                                             Rb,
        #                                                             self.r_proj_device,
        #                                                             self.all_sigma2_device,
        #                                                             self.Xs_device,
        #                                                             self.central_integration_radius,
        #                                                             self.central_index)
        self._bulge[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                                Rb,
                                                                gamma,
                                                                self.r_proj_device,
                                                                self.s_grid_device,
                                                                self.gamma_grid_device,
                                                                self.rho_grid_device,
                                                                self.sigma_grid_device,
                                                                self.all_rhoB_device,
                                                                self.all_sigma2_device,
                                                                self.central_integration_radius,
                                                                self.central_index,
                                                                self.central_rho_device,
                                                                self.central_sigma_device)
        ## DISC SURFACE DENSITY
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                               Rd1,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD1_device)

        ## DISC VELOCITY 
        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](f*(Md1+Mb),
                                                             c,
                                                             self.rho_crit,
                                                             self.r_true_device,
                                                             self.v_halo_device)
        #self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
        #                                                    Rb,
        #                                                    self.r_true_device,
        #                                                    self.v_bulge_device)
        self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                            Rb,
                                                            gamma,
                                                            self.r_true_device,
                                                            self.v_bulge_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                             Rd1,
                                                             self.r_true_device,
                                                             self.v_exp1_device)

        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
                                                                self.v_bulge_device,
                                                                self.v_halo_device,
                                                                incl,
                                                                self.phi_device,
                                                                self.all_v_device,
                                                                self.all_v2_device)

        #  RHO, V E SIGMA with psf
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.kinpsf_device,
                                                                   self.phopsf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   k1,
                                                                   self.rho_avg_device,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device,
                                                                   self.LM_avg_device,
                                                                   self.indexing_device,
                                                                   self.goodpix_device)


        self._avg_sigma[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.kinpsf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   k1,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device,
                                                                   self.indexing_device,
                                                                   self.goodpix_device)

        self._sum_likelihood[self.blockpergrid_3,self.threadsperblock_3](self.rho_avg_device,
                                                                        self.v_avg_device,
                                                                        self.sigma_avg_device,
                                                                        self.LM_avg_device,
                                                                        self.ibrightness_data_device,
                                                                        self.v_data_device,
                                                                        self.sigma_data_device,
                                                                        self.ML_data_device,
                                                                        self.ibrightness_error_device,
                                                                        self.v_error_device,
                                                                        self.sigma_error_device,
                                                                        self.ML_error_device,
                                                                        sys_rho,
                                                                        self.goodpix_device,
                                                                        self.lk_device)

        lk = self.lk_device.copy_to_host()

        return lk[0]


    def model(self,
              all_params,
              *args):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Compute likelihood of the model.
            
            Parameters:
                all_params: dict
                    Dictionary containing all the model parameters

            Returns:
                rho: 1D arr float32
                    Total surface brighness
                rho_B: 1D arr float32
                    Bulge surface brightness
                rho_D1: 1D arr float32
                    Disc1 surface brightness
                v: 1D arr float32
                    Mean los velocity
                v_B: 1D arr float32
                    Bulge los velocity
                v_D1: 1D arr float32
                    Disc1 los velocity
                sigma: 1D arr float32
                    Mean los velocity dispersion
                sigma_B: 1D arr float32
                    Bulge los velocity dispersion
                sigma_D1: 1D arr float32
                    Disc1 los velocity dispersion
                LM: 1D arr float32
                    Mean L/M ratio
                LM_B: 1D arr float32
                    Bulge L/M ratio
                LM_D1: 1D arr float32
                    Disc1 L/M ratio


        '''
        if self.counter==0:
            cuda.select_device(self.gpu_device)
            self._init_grid()
            self._load_to_deviceBD()
            self._init_dataBD()
            self._function_definitionBD()
            self._load_tabulated()
            from utils_numba_1D_32bit import avg_rho_v_2_vis_components_all_output
            from utils_numba_1D_32bit import avg_sigma_2_vis_components_all_output
            self._avg_sigma                 = avg_sigma_2_vis_components_all_output
            self._avg_rho_v                 = avg_rho_v_2_vis_components_all_output
            self._load_device_model()
            self.counter +=1
        else: 
            cuda.select_device(self.gpu_device)
            self._init_grid()
            self._load_to_deviceBD()
            self._init_dataBD()
            self._function_definitionBD()
            self._load_tabulated()
            from utils_numba_1D_32bit import avg_rho_v_2_vis_components_all_output
            from utils_numba_1D_32bit import avg_sigma_2_vis_components_all_output
            self._avg_sigma                 = avg_sigma_2_vis_components_all_output
            self._avg_rho_v                 = avg_rho_v_2_vis_components_all_output
            self._load_device_model()
            self.counter +=1


        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        k1             = all_params['k1']
        sys_rho        = all_params['sys_rho']
        f              = all_params['log10_halo_fraction']
        c              = all_params[self.halo_radius_name]
        gamma          = all_params['gamma']

        Mb           = 10**Mb
        Md1          = 10**Md1    
        Rb           = 10**Rb    
        Rd1          = 10**Rd1    
        sys_rho      = 10**sys_rho
        f            = 10**f 
        
        if self.halo_type=='NFW':
            pass
        else:
            c = 10**c
        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)

        self.central_index = np.argmin(self.r_proj_device.copy_to_host())

        self._bulge[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                                Rb,
                                                                gamma,
                                                                self.r_proj_device,
                                                                self.s_grid_device,
                                                                self.gamma_grid_device,
                                                                self.rho_grid_device,
                                                                self.sigma_grid_device,
                                                                self.all_rhoB_device,
                                                                self.all_sigma2_device,
                                                                self.central_integration_radius,
                                                                self.central_index,
                                                                self.central_rho_device,
                                                                self.central_sigma_device)
        ## DISC SURFACE DENSITY
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                               Rd1,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD1_device)

        ## DISC VELOCITY 
        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](f*(Md1+Mb),
                                                             c,
                                                             self.rho_crit,
                                                             self.r_true_device,
                                                             self.v_halo_device)
        #self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
        #                                                    Rb,
        #                                                    self.r_true_device,
        #                                                    self.v_bulge_device)
        self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                            Rb,
                                                            gamma,
                                                            self.r_true_device,
                                                            self.v_bulge_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                             Rd1,
                                                             self.r_true_device,
                                                             self.v_exp1_device)
        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
                                                                self.v_bulge_device,
                                                                self.v_halo_device,
                                                                incl,
                                                                self.phi_device,
                                                                self.all_v_device,
                                                                self.all_v2_device)

        #  RHO, V E SIGMA with psf
        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.kinpsf_device,
                                                                   self.phopsf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   k1,
                                                                   self.rho_avg_device,
                                                                   self.rho_avg_B_device,
                                                                   self.rho_avg_D1_device,
                                                                   self.v_avg_device,
                                                                   self.v_avg_B_device,
                                                                   self.v_avg_D1_device,
                                                                   self.LM_avg_device,
                                                                   self.LM_avg_B_device,
                                                                   self.LM_avg_D1_device,
                                                                   self.indexing_device,
                                                                   model_goodpix)


        self._avg_sigma[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.kinpsf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   k1,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device,
                                                                   self.sigma_avg_B_device,
                                                                   self.sigma_avg_D1_device,
                                                                   self.indexing_device,
                                                                   model_goodpix)


        rho      = self.rho_avg_device.copy_to_host()
        rho_B    = self.rho_avg_B_device.copy_to_host()
        rho_D1   = self.rho_avg_D1_device.copy_to_host()
        v        = self.v_avg_device.copy_to_host()
        v_B      = self.v_avg_B_device.copy_to_host()
        v_D1     = self.v_avg_D1_device.copy_to_host()
        sigma    = self.sigma_avg_device.copy_to_host()
        sigma_B  = self.sigma_avg_B_device.copy_to_host()
        sigma_D1 = self.sigma_avg_D1_device.copy_to_host()
        LM       = self.LM_avg_device.copy_to_host()
        LM_B     = self.LM_avg_B_device.copy_to_host()
        LM_D1    = self.LM_avg_D1_device.copy_to_host()
        zero = np.zeros_like(rho)

        return rho,rho_B,rho_D1,zero,v,v_B,v_D1,zero,sigma,sigma_B,sigma_D1,zero,LM,LM_B,LM_D1,zero
        

class model_gpu_DD():
    def __init__(self,
                 gpu_device,
                 fast_math,
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
                 rho_crit):


        ''' 
        A class for representing a "Disc + Disc + Halo" model on GPU
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------                                           
            Attributes:
                    gpu_device : int
                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
                    fast_math: Bool
                        Whether to use fastmath operation of not
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
            Methods:
                    _function_definition():
                        Define all functions depending on the condition above (bulge_type, halo_type ...)
                    _init_grid():
                        Compute # blocks, # theads for the Gpu
                    _load_to_device():
                        Load all data on GPU
                    _init_data():
                        Allocate memory for GPU operations
                    _load_device_model():
                        Allocate additional memory in case of model (not likelihood) call
                    _load_tabulated():
                        Load Dehnen interpolated quantities
                    likelihood(all_params,
                               *args,
                               **kwargs):
                        Compute likelihood of the model
                    model(all_params,
                               *args,
                               **kwargs):
                        Compute model all outputs

        '''
        
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data                  = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error                 = ML_error.copy(order='C')
        self.kin_psf                  = kin_psf.copy(order='C')
        self.pho_psf                  = pho_psf.copy(order='C')
        self.bulge_type               = bulge_type
        self.halo_type                = halo_type
        self.fast_math                = fast_math
        self.indexing                 = indexing
        self.counter = 0
        self.size = self.x_g.shape[0]
        self.goodpix = goodpix

        self.rho_crit = rho_crit
        self.new_N,self.new_J = int(self.size**0.5),int(self.size**0.5)

        self.dx = np.abs(self.x_g[1]-self.x_g[0])

        self.central_integration_radius = self.dx/2**0.5
        flat_index = np.argmin(np.sqrt(self.x_g*self.x_g+self.y_g*self.y_g))
        self.central_i = int(flat_index/self.new_J)
        self.central_j = flat_index - self.central_i*self.new_J
        self.gpu_device = gpu_device

    def _function_definitionDD(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Define all functions depending on the condition above (bulge_type, halo_type ...)
            
            Parameters:

            Returns:

        '''
        if self.fast_math == True:
            if self.halo_type=='hernquist':
                from utils_numba_1D_32bit import v_Halo
                self._halo_vel = v_Halo
                self.halo_radius_name = 'log10_halo_radius'
                self.halo_mass_name   = 'log10_halo_fraction'
                
            elif self.halo_type=='NFW':
                from utils_numba_1D_32bit import v_NFW
                self._halo_vel = v_NFW
                self.halo_radius_name = 'concentration'
                self.halo_mass_name   = 'log10_halo_fraction'

            from utils_numba_1D_32bit import rho_D
            from utils_numba_1D_32bit import v_D
            from utils_numba_1D_32bit import avg_rho_v_2_vis_components_DD
            from utils_numba_1D_32bit import avg_sigma_2_vis_components_DD
            from utils_numba_1D_32bit import sum_likelihood
            from utils_numba_1D_32bit import coordinate_transformation
            from utils_numba_1D_32bit_no_proxy import v_tot_BD

        else:
            if self.halo_type=='hernquist':
                from utils_numba_1D_32bit import v_Halo
                self._halo_vel = v_Halo
                self.halo_radius_name = 'log10_halo_radius'
                self.halo_mass_name   = 'log10_halo_fraction'
                
            elif self.halo_type=='NFW':
                from utils_numba_1D_32bit import v_NFW
                self._halo_vel = v_NFW
                self.halo_radius_name = 'concentration'
                self.halo_mass_name   = 'log10_halo_fraction'
            
            from utils_numba_1D_32bit_no_proxy import rho_D
            from utils_numba_1D_32bit_no_proxy import v_D
            from utils_numba_1D_32bit_no_proxy import v_tot_BD
            from utils_numba_1D_32bit_no_proxy import avg_rho_v_2_vis_components
            from utils_numba_1D_32bit_no_proxy import avg_LM_2_vis_components
            from utils_numba_1D_32bit_no_proxy import sum_likelihood
            from utils_numba_1D_32bit_no_proxy import coordinate_transformation
            from utils_numba_1D_32bit_no_proxy import Xfunction

       
        self._coordinate_transformation = coordinate_transformation
        self._rho_D                     = rho_D
        self._v_D                       = v_D
        self._v_tot                     = v_tot_BD
        self._avg_rho_v                 = avg_rho_v_2_vis_components_DD
        self._avg_sigma                 = avg_sigma_2_vis_components_DD
        #self._avg_LM                    = avg_LM_2_vis_components
        self._sum_likelihood            = sum_likelihood

    def _init_grid(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Compute # blocks, # theads for the Gpu
        
            Parameters:

            Returns:
            
        '''
        Nthreads_1 = 64
        Nthreads_3 = 64
        Nblocks_1 = math.ceil(self.size/Nthreads_1)
        Nblocks_3 = math.ceil(self.N*self.J/Nthreads_3)

        self.threadsperblock_1 = (Nthreads_1)
        self.threadsperblock_3 = (Nthreads_3)
        self.blockpergrid_1 = (Nblocks_1)
        self.blockpergrid_3 = (Nblocks_3)

    def _load_to_deviceDD(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
         Load all data on GPU
            
            Parameters:

            Returns:
            
        '''
        self.x_device = cuda.to_device(np.float32(self.x_g))
        self.y_device = cuda.to_device(np.float32(self.y_g))
        
        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
        self.all_rhoD2_device     = cuda.device_array((self.size),dtype=np.float32)
        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
        self.v_exp2_device        = cuda.device_array((self.size),dtype=np.float32)
        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)


        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
    
    def _init_data(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Allocate memory for GPU operations
            
            Parameters:

            Returns:
            
        '''
        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))

    def _load_device_modelDD(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Allocate additional memory in case of model (not likelihood) call

            Parameters:

            Returns:
            
        '''
        self.LM_avg_D2_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_D2_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_D2_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_D2_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
    

    def likelihood(self,
                   all_params,
                   *args):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Compute likelihood of the model.
            
            Parameters:
                all_params: dict
                    Dictionary containing all the model parameters

            Returns:
                lk: float32
                    Likelihood of the model.

        '''
        if self.counter==0:
            cuda.select_device(self.gpu_device)
            self._init_grid()
            self._load_to_deviceDD()
            self._init_data()
            self._function_definitionDD()
            self.counter +=1
        else: 
            pass
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']
        sys_rho        = all_params['sys_rho']
        f              = all_params['log10_halo_fraction']
        c              = all_params[self.halo_radius_name]

        Md2          = 10**Md2    
        Md1          = 10**Md1    
        Rd1          = 10**Rd1    
        Rd2          = 10**Rd2    
        sys_rho      = 10**sys_rho
        f            = 10**f 
        
        if self.halo_type=='NFW':
            pass
        else:
            c = 10**c
        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)

        self.central_index = np.argmin(self.r_proj_device.copy_to_host())
        ## DISC SURFACE DENSITY
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                               Rd1,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD1_device)
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                                Rd2,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD2_device)

        ## DISC VELOCITY 
        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](f*(Md1+Md2),
                                                             c,
                                                             self.rho_crit,
                                                             self.r_true_device,
                                                             self.v_halo_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                             Rd1,
                                                             self.r_true_device,
                                                             self.v_exp1_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                             Rd2,
                                                             self.r_true_device,
                                                             self.v_exp2_device)

        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
                                                                self.v_exp2_device,
                                                                self.v_halo_device,
                                                                incl,
                                                                self.phi_device,
                                                                self.all_v_device,
                                                                self.all_v2_device)

        #  RHO, V E SIGMA with psf
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoD1_device,
                                                                   self.all_rhoD2_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.kinpsf_device,
                                                                   self.phopsf_device,
                                                                   1.0/ML_d1,
                                                                   1.0/ML_d2,
                                                                   k1,
                                                                   k2,
                                                                   self.rho_avg_device,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device,
                                                                   self.LM_avg_device,
                                                                   self.indexing_device,
                                                                   self.goodpix_device)


        self._avg_sigma[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoD1_device,
                                                                   self.all_rhoD2_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.kinpsf_device,
                                                                   1.0/ML_d1,
                                                                   1.0/ML_d2,
                                                                   k1,
                                                                   k2,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device,
                                                                   self.indexing_device,
                                                                   self.goodpix_device)

        self._sum_likelihood[self.blockpergrid_3,self.threadsperblock_3](self.rho_avg_device,
                                                                        self.v_avg_device,
                                                                        self.sigma_avg_device,
                                                                        self.LM_avg_device,
                                                                        self.ibrightness_data_device,
                                                                        self.v_data_device,
                                                                        self.sigma_data_device,
                                                                        self.ML_data_device,
                                                                        self.ibrightness_error_device,
                                                                        self.v_error_device,
                                                                        self.sigma_error_device,
                                                                        self.ML_error_device,
                                                                        sys_rho,
                                                                        self.goodpix_device,
                                                                        self.lk_device)

        lk = self.lk_device.copy_to_host()

        return lk[0]



    def model(self,
              all_params,
              *args):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Compute likelihood of the model.
            
            Parameters:
                all_params: dict
                    Dictionary containing all the model parameters

            Returns:
                rho: 1D arr float32
                    Total surface brighness
                rho_D1: 1D arr float32
                    Disc1 surface brightness
                rho_D2: 1D arr float32
                    Disc2 surface brightness
                v: 1D arr float32
                    Mean los velocity
                v_D1: 1D arr float32
                    Disc1 los velocity
                v_D2: 1D arr float32
                    Disc2 los velocity
                sigma: 1D arr float32
                    Mean los velocity dispersion
                sigma_D1: 1D arr float32
                    Disc1 los velocity dispersion
                sigma_D2: 1D arr float32
                    Disc2 los velocity dispersion
                LM: 1D arr float32
                    Mean L/M ratio
                LM_D1: 1D arr float32
                    Disc1 L/M ratio
                LM_D2: 1D arr float32
                    Disc2 L/M ratio


        '''
        if self.counter==0:
            cuda.select_device(self.gpu_device)
            self._init_grid()
            self._load_to_deviceDD()
            self._init_data()
            self._function_definitionDD()
            from utils_numba_1D_32bit import avg_rho_v_2_vis_components_DD_all_output
            from utils_numba_1D_32bit import avg_sigma_2_vis_components_DD_all_output
            self._avg_sigma                 = avg_sigma_2_vis_components_DD_all_output
            self._avg_rho_v                 = avg_rho_v_2_vis_components_DD_all_output
            self._load_device_model()
            self.counter +=1
        else: 
            cuda.select_device(self.gpu_device)
            self._init_grid()
            self._load_to_deviceDD()
            self._init_data()
            self._function_definitionDD()
            from utils_numba_1D_32bit import avg_rho_v_2_vis_components_DD_all_output
            from utils_numba_1D_32bit import avg_sigma_2_vis_components_DD_all_output
            self._avg_sigma                 = avg_sigma_2_vis_components_DD_all_output
            self._avg_rho_v                 = avg_rho_v_2_vis_components_DD_all_output
            self._load_device_model()
            self.counter +=1
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']
        sys_rho        = all_params['sys_rho']
        f              = all_params['log10_halo_fraction']
        c              = all_params[self.halo_radius_name]

        Md2          = 10**Md2    
        Md1          = 10**Md1    
        Rd1          = 10**Rd1    
        Rd2          = 10**Rd2    
        sys_rho      = 10**sys_rho
        f            = 10**f 
        
        if self.halo_type=='NFW':
            pass
        else:
            c = 10**c
        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)

        self.central_index = np.argmin(self.r_proj_device.copy_to_host())
        ## DISC SURFACE DENSITY
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                               Rd1,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD1_device)
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                                Rd2,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD2_device)

        ## DISC VELOCITY 
        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](f*(Md1+Md2),
                                                             c,
                                                             self.rho_crit,
                                                             self.r_true_device,
                                                             self.v_halo_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                             Rd1,
                                                             self.r_true_device,
                                                             self.v_exp1_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                             Rd2,
                                                             self.r_true_device,
                                                             self.v_exp2_device)

        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
                                                                self.v_exp2_device,
                                                                self.v_halo_device,
                                                                incl,
                                                                self.phi_device,
                                                                self.all_v_device,
                                                                self.all_v2_device)

        #  RHO, V E SIGMA with psf
        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
        #  RHO, V E SIGMA with psf
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoD1_device,
                                                                   self.all_rhoD2_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.kinpsf_device,
                                                                   self.phopsf_device,
                                                                   1.0/ML_d1,
                                                                   1.0/ML_d2,
                                                                   k1,
                                                                   k2,
                                                                   self.rho_avg_device,
                                                                   self.rho_avg_D1_device,
                                                                   self.rho_avg_D2_device,
                                                                   self.v_avg_device,
                                                                   self.v_avg_D1_device,
                                                                   self.v_avg_D2_device,
                                                                   self.LM_avg_device,
                                                                   self.LM_avg_D1_device,
                                                                   self.LM_avg_D2_device,
                                                                   self.indexing_device,
                                                                   model_goodpix)


        self._avg_sigma[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoD1_device,
                                                                   self.all_rhoD2_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.kinpsf_device,
                                                                   1.0/ML_d1,
                                                                   1.0/ML_d2,
                                                                   k1,
                                                                   k2,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device,
                                                                   self.sigma_avg_D1_device,
                                                                   self.sigma_avg_D2_device,
                                                                   self.indexing_device,
                                                                   model_goodpix)





        rho      = self.rho_avg_device.copy_to_host()
        rho_D2   = self.rho_avg_D2_device.copy_to_host()
        rho_D1   = self.rho_avg_D1_device.copy_to_host()
        v        = self.v_avg_device.copy_to_host()
        v_D2     = self.v_avg_D2_device.copy_to_host()
        v_D1     = self.v_avg_D1_device.copy_to_host()
        sigma    = self.sigma_avg_device.copy_to_host()
        sigma_D2 = self.sigma_avg_D2_device.copy_to_host()
        sigma_D1 = self.sigma_avg_D1_device.copy_to_host()
        LM       = self.LM_avg_device.copy_to_host()
        LM_D2    = self.LM_avg_D2_device.copy_to_host()
        LM_D1    = self.LM_avg_D1_device.copy_to_host()
        zero = np.zeros_like(rho)

        return rho,zero,rho_D1,rho_D2,v,zero,v_D1,v_D2,sigma,zero,sigma_D1,sigma_D2,LM,zero,LM_D1,LM_D2
        

class model_gpu_BH():
    # data can be putted as arg since they are useless in
    # case of only model evaluation
    def __init__(self,
                 fast_math,
                 bulge_type,
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
                 psf):

        ''' 
        '''
        
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data                  = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error                 = ML_error.copy(order='C')
        self.psf                      = psf.copy(order='C')
        self.bulge_type               = bulge_type
        self.fast_math                = fast_math

        self.counter = 0


    def _function_definitionBH(self):
        if self.fast_math == True:
            if self.bulge_type=='hernquist':
                from utils_numba_1D_32bit import herquinst_rho
                from utils_numba_1D_32bit_no_proxy import herquinst_sigma_BH
                from utils_numba_1D_32bit import v_H
                from utils_numba_1D_32bit_no_proxy import all_Xdot
                from utils_numba_1D_32bit_no_proxy import Y1
                from utils_numba_1D_32bit_no_proxy import Y2
                from utils_numba_1D_32bit_no_proxy import v_tot_BH
                self._bulge_rho   = herquinst_rho 
                self._bulge_sigma = herquinst_sigma_BH 
                self._bulge_vel   = v_H 
                self._all_Xdot    = all_Xdot
                self._Y1          = Y1
                self._Y2          = Y2
                print('Warning bulge dispersion has not yet be implemented in fast math mode')
            elif self.bulge_type=='jaffe':
                print('Error Jaffe with BH has not be implemented')
                exit()
            else:
                print('Error')
                exit()

            from utils_numba_1D_32bit import rho_D
            from utils_numba_1D_32bit import v_D
            from utils_numba_1D_32bit import v_H
            from utils_numba_1D_32bit import avg_rho_v
            from utils_numba_1D_32bit import avg_LM
            from utils_numba_1D_32bit import sum_likelihood
            from utils_numba_1D_32bit import coordinate_transformation
            from utils_numba_1D_32bit import Xfunction
            from utils_numba_1D_32bit_no_proxy import v_BH
        else:
            if self.bulge_type=='hernquist':
                from utils_numba_1D_32bit_no_proxy import herquinst_rho
                from utils_numba_1D_32bit_no_proxy import herquinst_sigma_BH
                from utils_numba_1D_32bit_no_proxy import v_H
                from utils_numba_1D_32bit_no_proxy import all_Xdot
                from utils_numba_1D_32bit_no_proxy import Y1
                from utils_numba_1D_32bit_no_proxy import Y2
                self._bulge_rho   = herquinst_rho 
                self._bulge_sigma = herquinst_sigma_BH 
                self._bulge_vel   = v_H 
                self._all_Xdot    = all_Xdot
                self._Y1          = Y1
                self._Y2          = Y2
            elif self.bulge_type=='jaffe':
                print('Error Jaffe with BH has not be implemented')
                exit()
            else:
                print('Error')
                exit()

            from utils_numba_1D_32bit_no_proxy import rho_D
            from utils_numba_1D_32bit_no_proxy import v_D
            from utils_numba_1D_32bit_no_proxy import v_H
            from utils_numba_1D_32bit_no_proxy import v_tot_BH
            from utils_numba_1D_32bit_no_proxy import avg_rho_v
            from utils_numba_1D_32bit_no_proxy import avg_LM
            from utils_numba_1D_32bit_no_proxy import sum_likelihood
            from utils_numba_1D_32bit_no_proxy import coordinate_transformation
            from utils_numba_1D_32bit_no_proxy import Xfunction
            from utils_numba_1D_32bit_no_proxy import v_BH

       
        self._coordinate_transformation = coordinate_transformation
        self._Xfunction                 = Xfunction
        self._rho_D                     = rho_D
        self._v_D                       = v_D
        self._halo_vel                  = v_H
        self._v_tot                     = v_tot_BH
        self._avg_rho_v                 = avg_rho_v
        self._avg_LM                    = avg_LM
        self._sum_likelihood            = sum_likelihood
        self._v_BH                      = v_BH

    def _init_grid(self):

        Nthreads_1 = 64
        Nthreads_3 = 64
        Nblocks_1 = math.ceil(self.N*self.J*self.K/Nthreads_1)
        Nblocks_3 = math.ceil(self.N*self.J/Nthreads_3)

        self.threadsperblock_1 = (Nthreads_1)
        self.threadsperblock_3 = (Nthreads_3)
        self.blockpergrid_1 = (Nblocks_1)
        self.blockpergrid_3 = (Nblocks_3)

    def _load_to_deviceBH(self):
        self.x_device = cuda.to_device(np.float32(self.x_g))
        self.y_device = cuda.to_device(np.float32(self.y_g))
        
        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_rhoD1_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_rhoD2_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_v_device         = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.all_v2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.v_exp1_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.v_exp2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.v_bulge_device       = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.v_halo_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.v_BH_device          = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)


        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
    
    def _init_data(self):
        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
 


    def likelihood(self,
                   all_params,
                   *args):
        ''' 
        '''
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        Mh             = all_params['log10_halo_mass']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        Rh             = all_params['log10_halo_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']
        sys_rho        = all_params['sys_rho']

        if self.counter==0:
            self._function_definitionBH()
            self._init_grid()
            self._load_to_deviceBH()
            self._init_data()
            self.counter +=1
        else: 
            pass

        Mbh          = 10**Mbh
        Mb           = 10**Mb
        Md1          = 10**Md1    
        Md2          = 10**Md2    
        Mh           = 10**Mh    
        Rb           = 10**Rb    
        Rd1          = 10**Rd1    
        Rd2          = 10**Rd2    
        Rh           = 10**Rh
        sys_rho      = 10**sys_rho

        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)


        self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device)
        self._all_Xdot[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device,
                                                                   self.X1_dot_device,
                                                                   self.X2_dot_device,
                                                                   self.X3_dot_device)


        self._Y1[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.Y_1_device)

        self._Y2[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.X2_dot_device,
                                                             self.X3_dot_device,
                                                             self.Y_2_device)

        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                                   Rb,
                                                                   self.r_proj_device,
                                                                   self.all_rhoB_device,
                                                                   self.Xs_device)
        # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mbh,
                                                                      Mb,
                                                                      Rb,
                                                                      self.r_proj_device,
                                                                      self.Y_1_device,
                                                                      self.Y_2_device,
                                                                      self.all_rhoB_device,
                                                                      self.all_sigma2_device)

        ## DISC SURFACE DENSITY
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                               Rd1,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD1_device)

        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                               Rd2,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD2_device)

        ## DISC VELOCITY 
        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](Mh,
                                                             Rh,
                                                             self.r_true_device,
                                                             self.v_halo_device)
        self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                            Rb,
                                                            self.r_true_device,
                                                            self.v_bulge_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                             Rd1,
                                                             self.r_true_device,
                                                             self.v_exp1_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                             Rd2,
                                                             self.r_true_device,
                                                             self.v_exp2_device)
        self._v_BH[self.blockpergrid_1,self.threadsperblock_1](Mbh,
                                                               self.r_true_device,
                                                               self.v_BH_device)
        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
                                                                self.v_exp2_device,
                                                                self.v_bulge_device,
                                                                self.v_halo_device,
                                                                self.v_BH_device,
                                                                incl,
                                                                self.phi_device,
                                                                self.all_v_device,
                                                                self.all_v2_device)

        #  RHO, V E SIGMA with psf
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_rhoD2_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.psf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   1.0/ML_d2,
                                                                   k1,
                                                                   k2,
                                                                   self.rho_avg_device,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device)


        self._avg_LM[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                self.all_rhoD1_device,
                                                                self.all_rhoD2_device,
                                                                self.psf_device,
                                                                1.0/ML_b,
                                                                1.0/ML_d1,
                                                                1.0/ML_d2,
                                                                self.LM_avg_device)


        self._sum_likelihood[self.blockpergrid_3,self.threadsperblock_3](self.rho_avg_device,
                                                                        self.v_avg_device,
                                                                        self.sigma_avg_device,
                                                                        self.LM_avg_device,
                                                                        self.ibrightness_data_device,
                                                                        self.v_data_device,
                                                                        self.sigma_data_device,
                                                                        self.ML_data_device,
                                                                        self.ibrightness_error_device,
                                                                        self.v_error_device,
                                                                        self.sigma_error_device,
                                                                        self.ML_error_device,
                                                                        sys_rho,
                                                                        self.lk_device)

        lk = self.lk_device.copy_to_host()


        return lk[0]



    def model(self,
              all_params,
              *args):
        ''' 
        '''
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        Mh             = all_params['log10_halo_mass']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        Rh             = all_params['log10_halo_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']
        if self.counter==0:
            self._function_definitionBH()
            self._init_grid()
            self._load_to_deviceBH()
            self._init_data()
            self.counter +=1
        else: 
            pass

        Mbh          = 10**Mbh
        Mb           = 10**Mb
        Md1          = 10**Md1    
        Md2          = 10**Md2    
        Mh           = 10**Mh    
        Rb           = 10**Rb    
        Rd1          = 10**Rd1    
        Rd2          = 10**Rd2    
        Rh           = 10**Rh
        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)


        self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device)
        self._all_Xdot[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                                   self.r_proj_device,
                                                                   self.Xs_device,
                                                                   self.X1_dot_device,
                                                                   self.X2_dot_device,
                                                                   self.X3_dot_device)
        self._Y1[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.Y_1_device)
        self._Y2[self.blockpergrid_1,self.threadsperblock_1](Rb,
                                                             self.r_proj_device,
                                                             self.Xs_device,
                                                             self.X1_dot_device,
                                                             self.X2_dot_device,
                                                             self.X3_dot_device,
                                                             self.Y_2_device)
        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                                    Rb,
                                                                    self.r_proj_device,
                                                                    self.all_rhoB_device,
                                                                    self.Xs_device)
        # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mbh,
                                                                      Mb,
                                                                      Rb,
                                                                      self.r_proj_device,
                                                                      self.Y_1_device,
                                                                      self.Y_2_device,
                                                                      self.all_rhoB_device,
                                                                      self.all_sigma2_device)

        ## DISC SURFACE DENSITY
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                               Rd1,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD1_device)

        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                               Rd2,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD2_device)

        ## DISC VELOCITY 
        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](Mh,
                                                             Rh,
                                                             self.r_true_device,
                                                             self.v_halo_device)
        self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                            Rb,
                                                            self.r_true_device,
                                                            self.v_bulge_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                             Rd1,
                                                             self.r_true_device,
                                                             self.v_exp1_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                             Rd2,
                                                             self.r_true_device,
                                                             self.v_exp2_device)
        self._v_BH[self.blockpergrid_1,self.threadsperblock_1](Mbh,
                                                               self.r_true_device,
                                                               self.v_BH_device)
        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
                                                                self.v_exp2_device,
                                                                self.v_bulge_device,
                                                                self.v_halo_device,
                                                                self.v_BH_device,
                                                                incl,
                                                                self.phi_device,
                                                                self.all_v_device,
                                                                self.all_v2_device)

        #  RHO, V E SIGMA with psf
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_rhoD2_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.psf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   1.0/ML_d2,
                                                                   k1,
                                                                   k2,
                                                                   self.rho_avg_device,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device)


        self._avg_LM[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                self.all_rhoD1_device,
                                                                self.all_rhoD2_device,
                                                                self.psf_device,
                                                                1.0/ML_b,
                                                                1.0/ML_d1,
                                                                1.0/ML_d2,
                                                                self.LM_avg_device)

        rho   = self.rho_avg_device.copy_to_host()
        v     = self.v_avg_device.copy_to_host()
        sigma = self.sigma_avg_device.copy_to_host()
        LM    = self.LM_avg_device.copy_to_host()

        return rho,v,sigma,LM


class model_gpu():
    def __init__(self,
                 gpu_device,
                 fast_math,
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
                 rho_crit):
        ''' 
        A class for representing a "Bulge + Disc + Disc + Halo" model on GPU
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------                                           
            Attributes:
                    gpu_device : int
                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
                    fast_math: Bool
                        Whether to use fastmath operation of not
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
            Methods:
                    _function_definition():
                        Define all functions depending on the condition above (bulge_type, halo_type ...)
                    _init_grid():
                        Compute # blocks, # theads for the Gpu
                    _load_to_device():
                        Load all data on GPU
                    _init_data():
                        Allocate memory for GPU operations
                    _load_device_model():
                        Allocate additional memory in case of model (not likelihood) call
                    _load_tabulated():
                        Load Dehnen interpolated quantities
                    likelihood(all_params,
                               *args,
                               **kwargs):
                        Compute likelihood of the model
                    model(all_params,
                               *args,
                               **kwargs):
                        Compute model all outputs

        '''
        
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data                  = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error                 = ML_error.copy(order='C')
        self.kin_psf                  = kin_psf.copy(order='C')
        self.pho_psf                  = pho_psf.copy(order='C')
        self.bulge_type               = bulge_type
        self.halo_type                = halo_type
        self.fast_math                = fast_math
        self.indexing                 = indexing
        self.counter = 0
        self.size = self.x_g.shape[0]
        self.goodpix = goodpix

        self.rho_crit = rho_crit
        self.new_N,self.new_J = int(self.size**0.5),int(self.size**0.5)

        self.dx = np.abs(self.x_g[1]-self.x_g[0])

        self.central_integration_radius = self.dx/2**0.5
        flat_index = np.argmin(np.sqrt(self.x_g*self.x_g+self.y_g*self.y_g))
        self.central_i = int(flat_index/self.new_J)
        self.central_j = flat_index - self.central_i*self.new_J

        self.gpu_device = gpu_device

    def _function_definition(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Define all functions depending on the condition above (bulge_type, halo_type ...)
            
            Parameters:

            Returns:

        '''
        if self.fast_math == True:
            if self.bulge_type=='hernquist':
                from utils_numba_1D_32bit import herquinst_rho
                from utils_numba_1D_32bit import herquinst_sigma
                from utils_numba_1D_32bit import v_H
                self._bulge_rho   = herquinst_rho 
                self._bulge_sigma = herquinst_sigma 
                self._bulge_vel   = v_H 
            elif self.bulge_type=='jaffe':
                from utils_numba_1D_32bit import jaffe_rho        
                from utils_numba_1D_32bit import jaffe_sigma  
                from utils_numba_1D_32bit import v_J
                self._bulge_rho   = jaffe_rho
                self._bulge_sigma = jaffe_sigma
                self._bulge_vel   = v_J
            elif self.bulge_type=='dehnen':
                from utils_numba_1D_32bit import dehnen        
                #from utils_numba_1D_32bit import jaffe_sigma  
                from utils_numba_1D_32bit import v_dehnen
                self._bulge = dehnen
                self._bulge_vel = v_dehnen
                
            else:
                print('Error')
                exit()
            
            if self.halo_type=='hernquist':
                from utils_numba_1D_32bit import v_Halo
                self._halo_vel = v_Halo
                self.halo_radius_name = 'log10_halo_radius'
                self.halo_mass_name   = 'log10_halo_fraction'
                
            elif self.halo_type=='NFW':
                from utils_numba_1D_32bit import v_NFW
                self._halo_vel = v_NFW
                self.halo_radius_name = 'concentration'
                self.halo_mass_name   = 'log10_halo_fraction'

            from utils_numba_1D_32bit import rho_D
            from utils_numba_1D_32bit import v_D
            from utils_numba_1D_32bit import v_tot
            from utils_numba_1D_32bit import avg_rho_v
            from utils_numba_1D_32bit import sum_likelihood
            from utils_numba_1D_32bit import coordinate_transformation
            from utils_numba_1D_32bit import Xfunction
            from utils_numba_1D_32bit import avg_sigma
        else:
            if self.bulge_type=='hernquist':
                from utils_numba_1D_32bit_no_proxy import herquinst_rho
                from utils_numba_1D_32bit_no_proxy import herquinst_sigma
                from utils_numba_1D_32bit_no_proxy import v_H
                self._bulge_rho   = herquinst_rho 
                self._bulge_sigma = herquinst_sigma 
                self._bulge_vel   = v_H 
            elif self.bulge_type=='jaffe':
                from utils_numba_1D_32bit_no_proxy import jaffe_rho        
                from utils_numba_1D_32bit_no_proxy import jaffe_sigma  
                from utils_numba_1D_32bit_no_proxy import v_J
                self._bulge_rho   = jaffe_rho
                self._bulge_sigma = jaffe_sigma
                self._bulge_vel   = v_J
            elif self.bulge_type=='dehnen':
                from utils_numba_1D_32bit import dehnen        
                from utils_numba_1D_32bit import v_dehnen
                self._bulge = dehnen
                self._bulge_vel = v_dehnen
            else:
                print('Error')
                exit()

            if self.halo_type=='hernquist':
                from utils_numba_1D_32bit import v_Halo
                self._halo_vel = v_Halo
                self.halo_radius_name = 'log10_halo_radius'
                self.halo_mass_name   = 'log10_halo_fraction'
                
            elif self.halo_type=='NFW':
                from utils_numba_1D_32bit import v_NFW
                self._halo_vel = v_NFW
                self.halo_radius_name = 'concentration'
                self.halo_mass_name   = 'log10_halo_fraction'
            from utils_numba_1D_32bit_no_proxy import rho_D
            from utils_numba_1D_32bit_no_proxy import v_D
            from utils_numba_1D_32bit_no_proxy import v_H
            from utils_numba_1D_32bit_no_proxy import v_tot
            from utils_numba_1D_32bit_no_proxy import avg_rho_v
            from utils_numba_1D_32bit_no_proxy import sum_likelihood
            from utils_numba_1D_32bit_no_proxy import coordinate_transformation
            from utils_numba_1D_32bit_no_proxy import Xfunction

        self._avg_sigma                 = avg_sigma
        self._coordinate_transformation = coordinate_transformation
        self._Xfunction                 = Xfunction
        self._rho_D                     = rho_D
        self._v_D                       = v_D
        self._v_tot                     = v_tot
        self._avg_rho_v                 = avg_rho_v
        self._sum_likelihood            = sum_likelihood

    def _init_grid(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Compute # blocks, # theads for the Gpu
        
            Parameters:

            Returns:
            
        '''
        Nthreads_1 = 64
        Nthreads_3 = 64
        #Nblocks_1 = math.ceil(self.N*self.J*self.K/Nthreads_1)
        Nblocks_1 = math.ceil(self.size/Nthreads_1)
        Nblocks_3 = math.ceil(self.N*self.J/Nthreads_3)

        self.threadsperblock_1 = (Nthreads_1)
        self.threadsperblock_3 = (Nthreads_3)
        self.blockpergrid_1 = (Nblocks_1)
        self.blockpergrid_3 = (Nblocks_3)

    def _load_to_device(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
         Load all data on GPU
            
            Parameters:

            Returns:
            
        '''
        self.x_device = cuda.to_device(np.float32(self.x_g))
        self.y_device = cuda.to_device(np.float32(self.y_g))
        
        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
        self.all_rhoB_device      = cuda.device_array((self.size),dtype=np.float32)
        self.all_sigma2_device    = cuda.device_array((self.size),dtype=np.float32)
        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
        self.all_rhoD2_device     = cuda.device_array((self.size),dtype=np.float32)
        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
        self.v_exp2_device        = cuda.device_array((self.size),dtype=np.float32)
        self.v_bulge_device       = cuda.device_array((self.size),dtype=np.float32)
        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)

        self.vD1_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.vD2_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
    
    def _init_data(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Allocate memory for GPU operations
            
            Parameters:

            Returns:
            
        '''
        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))

    def _load_device_model(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Allocate additional memory in case of model (not likelihood) call

            Parameters:

            Returns:
            
        '''
        self.LM_avg_B_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.LM_avg_D2_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_B_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.rho_avg_D2_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_B_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.v_avg_D2_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_B_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
        self.sigma_avg_D2_device         = cuda.device_array((self.N*self.J),dtype=np.float32)

    def _load_tabulated(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Load Dehnen interpolated quantities

            Parameters:

            Returns:
            
        '''
        path = './dehnen/'
        s_grid        = np.load(path+'s_grid.npy')
        gamma_grid    = np.load(path+'gamma_grid.npy') 
        rho_grid      = np.load(path+'dehnen_brightness.npy')
        sigma_grid    = np.load(path+'dehnen_sigma2.npy')
        central_rho   = np.load(path+'dehnen_luminosity.npy')
        central_sigma = np.load(path+'dehnen_avg_sigma2.npy')

        self.s_grid_device        = cuda.to_device(np.float32(s_grid))
        self.gamma_grid_device    = cuda.to_device(np.float32(gamma_grid))
        self.rho_grid_device      = cuda.to_device(np.float32(rho_grid))
        self.sigma_grid_device    = cuda.to_device(np.float32(sigma_grid))
        self.central_rho_device   = cuda.to_device(np.float32(central_rho))
        self.central_sigma_device = cuda.to_device(np.float32(central_sigma))

    def likelihood(self,
                   all_params,
                   *args,
                   **kwargs):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Compute likelihood of the model.
            
            Parameters:
                all_params: dict
                    Dictionary containing all the model parameters

            Returns:
                lk: float32
                    Likelihood of the model.

        '''
        if self.counter==0:
            cuda.select_device(self.gpu_device)
            self._init_grid()
            self._load_to_device()
            self._init_data()
            self._function_definition()
            self._load_tabulated()
            self.counter +=1
        else: 
            pass
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        f              = all_params['log10_halo_fraction']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        c              = all_params[self.halo_radius_name]
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']
        gamma          = all_params['gamma']
        sys_rho        = all_params['sys_rho']
        #i0,j0 = np.round(x0/self.dx),np.round(y0/self.dx)
        #self.central_index = (self.central_i+j0)*self.new_J+self.central_j+i0
        #self.central_index = int(self.central_index)
        
        #itmp = self.central_i+j0
        #jtmp = self.central_j+i0
        
        Mb           = 10**Mb
        Md1          = 10**Md1    
        Md2          = 10**Md2   
        f            = 10**f 
        Rb           = 10**Rb    
        Rd1          = 10**Rd1    
        Rd2          = 10**Rd2    
        sys_rho      = 10**sys_rho

        if self.halo_type=='NFW':
            pass
        else:
            c = 10**c


        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                    self.y_device,
                                                                                    x0,
                                                                                    y0,
                                                                                    theta,
                                                                                    incl,
                                                                                    self.r_proj_device,
                                                                                    self.r_true_device,
                                                                                    self.phi_device)

        self.central_index = np.argmin(self.r_proj_device.copy_to_host())
        #self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
        #                                                           self.r_proj_device,
        #                                                           self.Xs_device,
        #                                                           self.central_integration_radius,
        #                                                           self.central_index)
        ### BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        #self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
        #                                                           Rb,
        #                                                           self.r_proj_device,
        #                                                           self.all_rhoB_device,
        #                                                           self.Xs_device,
        #                                                           self.central_integration_radius,
        #                                                           self.central_index)
        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        #self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mb,
        #                                                             Rb,
        #                                                             self.r_proj_device,
        #                                                             self.all_sigma2_device,
        #                                                             self.Xs_device,
        #                                                             self.central_integration_radius,
        #                                                             self.central_index)

        self._bulge[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                                Rb,
                                                                gamma,
                                                                self.r_proj_device,
                                                                self.s_grid_device,
                                                                self.gamma_grid_device,
                                                                self.rho_grid_device,
                                                                self.sigma_grid_device,
                                                                self.all_rhoB_device,
                                                                self.all_sigma2_device,
                                                                self.central_integration_radius,
                                                                self.central_index,
                                                                self.central_rho_device,
                                                                self.central_sigma_device)

        ## DISC SURFACE DENSITY
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                               Rd1,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD1_device)

        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                               Rd2,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD2_device)

        ## DISC VELOCITY 
        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](f*(Md1+Md2+Mb),
                                                             c,
                                                             self.rho_crit,
                                                             self.r_true_device,
                                                             self.v_halo_device)
        #self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
        #                                                    Rb,
        #                                                    self.r_true_device,
        #                                                    self.v_bulge_device)
        self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                            Rb,
                                                            gamma,
                                                            self.r_true_device,
                                                            self.v_bulge_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                             Rd1,
                                                             self.r_true_device,
                                                             self.v_exp1_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                             Rd2,
                                                             self.r_true_device,
                                                             self.v_exp2_device)
        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
                                                               self.v_exp2_device,
                                                               self.v_bulge_device,
                                                               self.v_halo_device,
                                                               incl,
                                                               self.phi_device,
                                                               self.all_v_device,
                                                               self.all_v2_device)

        #  RHO, V E SIGMA with psf
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_rhoD2_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.kinpsf_device,
                                                                   self.phopsf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   1.0/ML_d2,
                                                                   k1,
                                                                   k2,
                                                                   self.rho_avg_device,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device,
                                                                   self.LM_avg_device,
                                                                   self.indexing_device,
                                                                   self.goodpix_device)

        self._avg_sigma[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_rhoD2_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.kinpsf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   1.0/ML_d2,
                                                                   k1,
                                                                   k2,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device,
                                                                   self.indexing_device,
                                                                   self.goodpix_device)

        #self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
        #                                                           self.all_rhoD1_device,
        #                                                           self.all_rhoD2_device,
        #                                                           self.all_v_device,
        #                                                           self.kinpsf_device,
        #                                                           self.phopsf_device,
        #                                                           1.0/ML_b,
        #                                                           1.0/ML_d1,
        #                                                           1.0/ML_d2,
        #                                                           k1,
        #                                                           k2,
        #                                                           self.rho_avg_device,
        #                                                           self.v_avg_device,
        #                                                           self.vD1_avg_device,
        #                                                           self.vD2_avg_device,
        #                                                           self.LM_avg_device,
        #                                                           self.indexing_device,
        #                                                           self.goodpix_device)

        #self._avg_sigma[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
        #                                                           self.all_rhoD1_device,
        #                                                           self.all_rhoD2_device,
        #                                                           self.all_v_device,
        #                                                           self.all_v2_device,
        #                                                           self.all_sigma2_device,
        #                                                           self.kinpsf_device,
        #                                                           1.0/ML_b,
        #                                                           1.0/ML_d1,
        #                                                           1.0/ML_d2,
        #                                                           k1,
        #                                                           k2,
        #                                                           self.v_avg_device,
        #                                                           self.vD1_avg_device,
        #                                                           self.vD2_avg_device,
        #                                                           self.sigma_avg_device,
        #                                                           self.indexing_device,
        #                                                           self.goodpix_device)

        self._sum_likelihood[self.blockpergrid_3,self.threadsperblock_3](self.rho_avg_device,
                                                                        self.v_avg_device,
                                                                        self.sigma_avg_device,
                                                                        self.LM_avg_device,
                                                                        self.ibrightness_data_device,
                                                                        self.v_data_device,
                                                                        self.sigma_data_device,
                                                                        self.ML_data_device,
                                                                        self.ibrightness_error_device,
                                                                        self.v_error_device,
                                                                        self.sigma_error_device,
                                                                        self.ML_error_device,
                                                                        sys_rho,
                                                                        self.goodpix_device,
                                                                        self.lk_device)

        lk = self.lk_device.copy_to_host()
        #if np.isnan(lk):
        #    rho = self.all_sigma2_device.copy_to_host()
        #    #rho = self.all_rhoB_device.copy_to_host()
        #    r = self.r_proj_device.copy_to_host()
        #    #print((rho[rho<0]))
        #    #print(rho[rho==np.inf])
        #    print(rho[np.isnan(rho)])
        #    print(r[np.isnan(rho)]/Rb,r[self.central_index]/Rb)
            #print(r[rho==np.inf]/Rb)
            
            #r_rshp = r.reshape((self.new_N,self.new_J))
            #print(np.min(r),r_rshp[int(itmp),int(jtmp)])
            #print(np.argmin(r),self.central_index)
            #print(lk)
        return lk[0]


    def model(self, 
              all_params,
              *args,
              **kwargs):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Compute likelihood of the model.
            
            Parameters:
                all_params: dict
                    Dictionary containing all the model parameters

            Returns:
                rho: 1D arr float32
                    Total surface brighness
                rho_B: 1D arr float32
                    Bulge surface brightness
                rho_D1: 1D arr float32
                    Disc1 surface brightness
                rho_D2: 1D arr float32
                    Disc2 surface brightness
                v: 1D arr float32
                    Mean los velocity
                v_B: 1D arr float32
                    Bulge los velocity
                v_D1: 1D arr float32
                    Disc1 los velocity
                v_D2: 1D arr float32
                    Disc2 los velocity
                sigma: 1D arr float32
                    Mean los velocity dispersion
                sigma_B: 1D arr float32
                    Bulge los velocity dispersion
                sigma_D1: 1D arr float32
                    Disc1 los velocity dispersion
                sigma_D2: 1D arr float32
                    Disc2 los velocity dispersion
                LM: 1D arr float32
                    Mean L/M ratio
                LM_B: 1D arr float32
                    Bulge L/M ratio
                LM_D1: 1D arr float32
                    Disc1 L/M ratio
                LM_D2: 1D arr float32
                    Disc2 L/M ratio


        '''
        if self.counter==0:
            cuda.select_device(self.gpu_device)
            self._init_grid()
            self._load_to_device()
            self._init_data()
            self._function_definition()
            self._load_tabulated()
            from utils_numba_1D_32bit import avg_rho_v_all_output
            from utils_numba_1D_32bit import avg_sigma_all_output
            self._avg_sigma                 = avg_sigma_all_output
            self._avg_rho_v                 = avg_rho_v_all_output
            self._load_device_model()
            self.counter +=1
        else: 
            cuda.select_device(self.gpu_device)
            self._init_grid()
            self._load_to_device()
            self._init_data()
            self._function_definition()
            self._load_tabulated()
            from utils_numba_1D_32bit import avg_rho_v_all_output
            from utils_numba_1D_32bit import avg_sigma_all_output
            self._avg_sigma                 = avg_sigma_all_output
            self._avg_rho_v                 = avg_rho_v_all_output
            self._load_device_model()
            self.counter +=1

        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        f              = all_params['log10_halo_fraction']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        c              = all_params[self.halo_radius_name]
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']
        gamma          = all_params['gamma']
        
        #i0,j0 = np.round(x0/self.dx),np.round(y0/self.dx)
        #self.central_index = (self.central_i+i0)*self.J+self.central_j+j0
        #self.central_index = int(self.central_index)

        Mb           = 10**Mb
        Md1          = 10**Md1    
        Md2          = 10**Md2   
        f            = 10**f 
        Rb           = 10**Rb    
        Rd1          = 10**Rd1    
        Rd2          = 10**Rd2    
        #sys_rho      = 10**sys_rho

        if self.halo_type=='NFW':
            pass
        else:
            c = 10**c
        # GRID DEFINITION
        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
                                                                                   self.y_device,
                                                                                   x0,
                                                                                   y0,
                                                                                   theta,
                                                                                   incl,
                                                                                   self.r_proj_device,
                                                                                   self.r_true_device,
                                                                                   self.phi_device)
        self.central_index = np.argmin(self.r_proj_device.copy_to_host())
        #self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
        #                                                           self.r_proj_device,
        #                                                           self.Xs_device,
        #                                                           self.central_integration_radius,
        #                                                           self.central_index)
        ### BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        #self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
        #                                                           Rb,
        #                                                           self.r_proj_device,
        #                                                           self.all_rhoB_device,
        #                                                           self.Xs_device,
        #                                                           self.central_integration_radius,
        #                                                           self.central_index)
        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
        #self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mb,
        #                                                             Rb,
        #                                                             self.r_proj_device,
        #                                                             self.all_sigma2_device,
        #                                                             self.Xs_device,
        #                                                             self.central_integration_radius,
        #                                                             self.central_index)
        self._bulge[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                                Rb,
                                                                gamma,
                                                                self.r_proj_device,
                                                                self.s_grid_device,
                                                                self.gamma_grid_device,
                                                                self.rho_grid_device,
                                                                self.sigma_grid_device,
                                                                self.all_rhoB_device,
                                                                self.all_sigma2_device,
                                                                self.central_integration_radius,
                                                                self.central_index,
                                                                self.central_rho_device,
                                                                self.central_sigma_device)
        ## DISC SURFACE DENSITY
        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                               Rd1,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD1_device)

        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                               Rd2,
                                                               incl,
                                                               self.r_true_device,
                                                               self.all_rhoD2_device)

        ## DISC VELOCITY 
        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](f*(Md1+Md2+Mb),
                                                             c,
                                                             self.rho_crit,
                                                             self.r_true_device,
                                                             self.v_halo_device)
        self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
                                                            Rb,
                                                            gamma,
                                                            self.r_true_device,
                                                            self.v_bulge_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
                                                             Rd1,
                                                             self.r_true_device,
                                                             self.v_exp1_device)
        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
                                                             Rd2,
                                                             self.r_true_device,
                                                             self.v_exp2_device)
        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
                                                               self.v_exp2_device,
                                                               self.v_bulge_device,
                                                               self.v_halo_device,
                                                               incl,
                                                               self.phi_device,
                                                               self.all_v_device,
                                                               self.all_v2_device)

        #  RHO, V E SIGMA with psf
        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_rhoD2_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.kinpsf_device,
                                                                   self.phopsf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   1.0/ML_d2,
                                                                   k1,
                                                                   k2,
                                                                   self.rho_avg_device,
                                                                   self.rho_avg_B_device,
                                                                   self.rho_avg_D1_device,
                                                                   self.rho_avg_D2_device,
                                                                   self.v_avg_device,
                                                                   self.v_avg_B_device,
                                                                   self.v_avg_D1_device,
                                                                   self.v_avg_D2_device,
                                                                   self.LM_avg_device,
                                                                   self.LM_avg_B_device,
                                                                   self.LM_avg_D1_device,
                                                                   self.LM_avg_D2_device,
                                                                   self.indexing_device,
                                                                   model_goodpix)

        self._avg_sigma[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
                                                                   self.all_rhoD1_device,
                                                                   self.all_rhoD2_device,
                                                                   self.all_v_device,
                                                                   self.all_v2_device,
                                                                   self.all_sigma2_device,
                                                                   self.kinpsf_device,
                                                                   1.0/ML_b,
                                                                   1.0/ML_d1,
                                                                   1.0/ML_d2,
                                                                   k1,
                                                                   k2,
                                                                   self.v_avg_device,
                                                                   self.sigma_avg_device,
                                                                   self.sigma_avg_B_device,
                                                                   self.sigma_avg_D1_device,
                                                                   self.sigma_avg_D2_device,
                                                                   self.indexing_device,
                                                                   model_goodpix)

        #self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
        #                                                           self.all_rhoD1_device,
        #                                                           self.all_rhoD2_device,
        #                                                           self.all_v_device,
        #                                                           self.kinpsf_device,
        #                                                           self.phopsf_device,
        #                                                           1.0/ML_b,
        #                                                           1.0/ML_d1,
        #                                                           1.0/ML_d2,
        #                                                           k1,
        #                                                           k2,
        #                                                           self.rho_avg_device,
        #                                                           self.v_avg_device,
        #                                                           self.vD1_avg_device,
        #                                                           self.vD2_avg_device,
        #                                                           self.LM_avg_device,
        #                                                           self.indexing_device,
        #                                                           model_goodpix)

        #self._avg_sigma[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
        #                                                           self.all_rhoD1_device,
        #                                                           self.all_rhoD2_device,
        #                                                           self.all_v_device,
        #                                                           self.all_v2_device,
        #                                                           self.all_sigma2_device,
        #                                                           self.kinpsf_device,
        #                                                           1.0/ML_b,
        #                                                           1.0/ML_d1,
        #                                                           1.0/ML_d2,
        #                                                           k1,
        #                                                           k2,
        #                                                           self.v_avg_device,
        #                                                           self.vD1_avg_device,
        #                                                           self.vD2_avg_device,
        #                                                           self.sigma_avg_device,
        #                                                           self.indexing_device,
        #                                                           model_goodpix)
        


        rho      = self.rho_avg_device.copy_to_host()
        rho_B    = self.rho_avg_B_device.copy_to_host()
        rho_D1   = self.rho_avg_D1_device.copy_to_host()
        rho_D2   = self.rho_avg_D2_device.copy_to_host()
        v        = self.v_avg_device.copy_to_host()
        v_B      = self.v_avg_B_device.copy_to_host()
        v_D1     = self.v_avg_D1_device.copy_to_host()
        v_D2     = self.v_avg_D2_device.copy_to_host()
        sigma    = self.sigma_avg_device.copy_to_host()
        sigma_B  = self.sigma_avg_B_device.copy_to_host()
        sigma_D1 = self.sigma_avg_D1_device.copy_to_host()
        sigma_D2 = self.sigma_avg_D2_device.copy_to_host()
        LM       = self.LM_avg_device.copy_to_host()
        LM_B     = self.LM_avg_B_device.copy_to_host()
        LM_D1    = self.LM_avg_D1_device.copy_to_host()
        LM_D2    = self.LM_avg_D2_device.copy_to_host()

        return rho,rho_B,rho_D1,rho_D2,v,v_B,v_D1,v_D2,sigma,sigma_B,sigma_D1,sigma_D2,LM,LM_B,LM_D1,LM_D2
        
               


class model_cpu():
    def __init__(self,
                 fast_math,
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
                 rho_crit):
        ''' 
        A class for representing a "Bulge + Disc + Disc + Halo" model on CPU
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------                                           
            Attributes:
                    fast_math: Bool
                        Whether to use fastmath operation of not
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
            Methods:
                    _function_definition():
                        Define all functions depending on the condition above (bulge_type, halo_type ...)
                    likelihood(all_params,
                               *args,
                               **kwargs):
                        Compute likelihood of the model
                    model(all_params,
                               *args,
                               **kwargs):
                        Compute model all outputs

        '''
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data               = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error              = ML_error.copy(order='C')
        self.kin_psf               = kin_psf.copy(order='C')
        self.pho_psf               = pho_psf.copy(order='C')
        self.bulge_type               = bulge_type
        self.indexing               = indexing
        self.goodpix                = goodpix
        self.halo_type              = halo_type

        self.size = self.x_g.shape[0]
        self.goodpix = goodpix

        self.rho_crit = rho_crit
        self.new_N,self.new_J = int(self.size**0.5),int(self.size**0.5)

        self.dx = np.abs(self.x_g[1]-self.x_g[0])

        self.central_integration_radius = self.dx/2**0.5
        self._load_tabulated_CPU()
        self._function_definition_CPU()


    def _function_definition_CPU(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Define all functions depending on the condition above (bulge_type, halo_type ...)
            
            Parameters:

            Returns:

        '''
        if self.halo_type=='hernquist':
            from utils_easy import model_BDD_all_outputs
            from utils_easy import likelihood_BDD
            self._model      = model_BDD_all_outputs
            self._likelihood = likelihood_BDD
            self.halo_radius_name = 'log10_halo_radius'
        elif self.halo_type=='NFW':
            from utils_easy import model_BDD_all_outputs
            from utils_easy import likelihood_BDD
            self._model      = model_BDD_all_outputs
            self._likelihood = likelihood_BDD
            self.halo_radius_name = 'concentration'
        
    def _load_tabulated_CPU(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Load Dehnen interpolated quantities

            Parameters:

            Returns:
            
        '''
        path = './dehnen/'
        s_grid        = np.load(path+'s_grid.npy')
        gamma_grid    = np.load(path+'gamma_grid.npy') 
        rho_grid      = np.load(path+'dehnen_brightness.npy')
        sigma_grid    = np.load(path+'dehnen_sigma2.npy')
        central_rho   = np.load(path+'dehnen_luminosity.npy')
        central_sigma = np.load(path+'dehnen_avg_sigma2.npy')

        self.s_grid        = s_grid
        self.gamma_grid    = gamma_grid
        self.rho_grid      = rho_grid
        self.sigma_grid    = sigma_grid
        self.central_rho   = central_rho
        self.central_sigma = central_sigma

    def likelihood(self,
                   all_params,
                   *args,
                   **kwargs):
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        f              = all_params['log10_halo_fraction']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        if self.halo_type == 'NFW':
            c              = all_params[self.halo_radius_name]
            halo_TYPE = 1 
        elif self.halo_type == 'hernquist':
            c              = 10**all_params[self.halo_radius_name]
            halo_TYPE = 0 

        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']

        if self.bulge_type=='hernquist':
            gamma = 1
        elif self.bulge_type=='jaffe':
            gamma = 2
        elif self.bulge_type=='dehnen':
            gamma = all_params['gamma']

        central_index = np.argmin((self.x_g-x0)**2+(self.y_g-y0)**2)
        sys_rho        = all_params['sys_rho']

        lk = self._likelihood(self.x_g,          
                              self.y_g,
                              self.N,
                              self.J,
                              self.K,          
                              Mb,                  
                              Rb,                  
                              gamma,                  
                              Md1,                  
                              Rd1,                  
                              Md2,                  
                              Rd2,                  
                              f,                 
                              c,   # this is not in log10       
                              self.rho_crit,           
                              x0,                 
                              y0,                 
                              theta,              
                              incl,                
                              1.0/ML_b,        
                              1.0/ML_d1,       
                              1.0/ML_d2,        
                              k1,       
                              k2,
                              sys_rho,
                              self.s_grid,
                              self.gamma_grid,
                              self.rho_grid,
                              self.sigma_grid,
                              self.central_integration_radius,
                              self.central_rho,
                              self.central_sigma,
                              self.kin_psf,
                              self.pho_psf,
                              self.indexing, 
                              self.goodpix,
                              central_index,
                              halo_TYPE,
                              self.ibrightness_data,
                              self.v_data,
                              self.sigma_data,
                              self.ML_data,
                              self.ibrightness_error,
                              self.v_error,
                              self.sigma_error,
                              self.ML_error) 
        return lk 

    def model(self,
              all_params,
              *args,
              **kwargs):

        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        f              = all_params['log10_halo_fraction']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        if self.halo_type == 'NFW':
            c              = all_params[self.halo_radius_name]
            halo_TYPE = 1 
        elif self.halo_type == 'hernquist':
            c              = 10**all_params[self.halo_radius_name]
            halo_TYPE = 0 

        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']

        if self.bulge_type=='hernquist':
            gamma = 1
        elif self.bulge_type=='jaffe':
            gamma = 2
        elif self.bulge_type=='dehnen':
            gamma = all_params['gamma']

        model_goodpix = np.ones((self.N*self.J))
        central_index = np.argmin((self.x_g-x0)**2+(self.y_g-y0)**2)

        tot =  self._model(self.x_g,          
                           self.y_g,
                           self.N,
                           self.J,
                           self.K,          
                           Mb,                  
                           Rb,                  
                           gamma,                  
                           Md1,                  
                           Rd1,                  
                           Md2,                  
                           Rd2,                  
                           f,                 
                           c,   # this is not in log10 
                           self.rho_crit,                 
                           x0,                 
                           y0,                 
                           theta,              
                           incl,                
                           1.0/ML_b,        
                           1.0/ML_d1,       
                           1.0/ML_d2,        
                           k1,       
                           k2,
                           self.s_grid,
                           self.gamma_grid,
                           self.rho_grid,
                           self.sigma_grid,
                           self.central_integration_radius,
                           self.central_rho,
                           self.central_sigma,
                           self.kin_psf,
                           self.pho_psf,
                           self.indexing, 
                           model_goodpix,
                           central_index,
                           halo_TYPE)
        rho      = tot[:,0]
        rho_B    = tot[:,1]
        rho_D1   = tot[:,2]
        rho_D2   = tot[:,3]
        v        = tot[:,4]
        v_B      = tot[:,5]
        v_D1     = tot[:,6]
        v_D2     = tot[:,7]
        sigma    = tot[:,8]
        sigma_B  = tot[:,9]
        sigma_D1 = tot[:,10]
        sigma_D2 = tot[:,11]
        LM       = tot[:,12]
        LM_B     = tot[:,13]
        LM_D1    = tot[:,14]
        LM_D2    = tot[:,15]

        return rho,rho_B,rho_D1,rho_D2,v,v_B,v_D1,v_D2,sigma,sigma_B,sigma_D1,sigma_D2,LM,LM_B,LM_D1,LM_D2


class model_cpu_BD():
    def __init__(self,
                 fast_math,
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
                 rho_crit):
        ''' 
        A class for representing a "Bulge + Disc + Disc + Halo" model on CPU
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------                                           
            Attributes:
                    fast_math: Bool
                        Whether to use fastmath operation of not
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
            Methods:
                    _function_definition():
                        Define all functions depending on the condition above (bulge_type, halo_type ...)
                    likelihood(all_params,
                               *args,
                               **kwargs):
                        Compute likelihood of the model
                    model(all_params,
                               *args,
                               **kwargs):
                        Compute model all outputs

        '''
        
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data               = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error              = ML_error.copy(order='C')
        self.kin_psf               = kin_psf.copy(order='C')
        self.pho_psf               = pho_psf.copy(order='C')
        self.bulge_type               = bulge_type
        self.indexing               = indexing
        self.goodpix                = goodpix
        self.halo_type              = halo_type

        self.size = self.x_g.shape[0]
        self.goodpix = goodpix

        self.rho_crit = rho_crit
        self.new_N,self.new_J = int(self.size**0.5),int(self.size**0.5)

        self.dx = np.abs(self.x_g[1]-self.x_g[0])

        self.central_integration_radius = self.dx/2**0.5
        self._load_tabulated_CPU_BD()
        self._function_definition_CPU_BD()


    def _function_definition_CPU_BD(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Define all functions depending on the condition above (bulge_type, halo_type ...)
            
            Parameters:

            Returns:

        '''
        if self.halo_type=='hernquist':
            from utils_easy import model_BD_all_outputs
            from utils_easy import likelihood_BD
            self._model      = model_BD_all_outputs
            self._likelihood = likelihood_BD
            self.halo_radius_name = 'log10_halo_radius'
        elif self.halo_type=='NFW':
            from utils_easy import model_BD_all_outputs
            from utils_easy import likelihood_BD
            self._model      = model_BD_all_outputs
            self._likelihood = likelihood_BD
            self.halo_radius_name = 'concentration'
        
    def _load_tabulated_CPU_BD(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Load Dehnen interpolated quantities

            Parameters:

            Returns:
            
        '''
        path = './dehnen/'
        s_grid        = np.load(path+'s_grid.npy')
        gamma_grid    = np.load(path+'gamma_grid.npy') 
        rho_grid      = np.load(path+'dehnen_brightness.npy')
        sigma_grid    = np.load(path+'dehnen_sigma2.npy')
        central_rho   = np.load(path+'dehnen_luminosity.npy')
        central_sigma = np.load(path+'dehnen_avg_sigma2.npy')

        self.s_grid        = s_grid
        self.gamma_grid    = gamma_grid
        self.rho_grid      = rho_grid
        self.sigma_grid    = sigma_grid
        self.central_rho   = central_rho
        self.central_sigma = central_sigma


    def likelihood(self,
                   all_params,
                   *args):
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        f              = all_params['log10_halo_fraction']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        if self.halo_type == 'NFW':
            c              = all_params[self.halo_radius_name]
            halo_TYPE = 1 
        elif self.halo_type == 'hernquist':
            c              = 10**all_params[self.halo_radius_name]
            halo_TYPE = 0 

        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        k1             = all_params['k1']
        k2             = all_params['k2']

        if self.bulge_type=='hernquist':
            gamma = 1
        elif self.bulge_type=='jaffe':
            gamma = 2
        elif self.bulge_type=='dehnen':
            gamma = all_params['gamma']

        central_index = np.argmin((self.x_g-x0)**2+(self.y_g-y0)**2)
        sys_rho        = all_params['sys_rho']

        lk = self._likelihood(self.x_g,          
                              self.y_g,
                              self.N,
                              self.J,
                              self.K,          
                              Mb,                  
                              Rb,                  
                              gamma,                  
                              Md1,                  
                              Rd1,                  
                              f,                 
                              c,   # this is not in log10       
                              self.rho_crit,           
                              x0,                 
                              y0,                 
                              theta,              
                              incl,                
                              1.0/ML_b,        
                              1.0/ML_d1,       
                              k1,       
                              sys_rho,
                              self.s_grid,
                              self.gamma_grid,
                              self.rho_grid,
                              self.sigma_grid,
                              self.central_integration_radius,
                              self.central_rho,
                              self.central_sigma,
                              self.kin_psf,
                              self.pho_psf,
                              self.indexing, 
                              self.goodpix,
                              central_index,
                              halo_TYPE,
                              self.ibrightness_data,
                              self.v_data,
                              self.sigma_data,
                              self.ML_data,
                              self.ibrightness_error,
                              self.v_error,
                              self.sigma_error,
                              self.ML_error) 
        return lk 

    def model(self,
              all_params,
              *args):
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        f              = all_params['log10_halo_fraction']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        if self.halo_type == 'NFW':
            c              = all_params[self.halo_radius_name]
            halo_TYPE = 1 
        elif self.halo_type == 'hernquist':
            c              = 10**all_params[self.halo_radius_name]
            halo_TYPE = 0 

        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        k1             = all_params['k1']
        k2             = all_params['k2']

        if self.bulge_type=='hernquist':
            gamma = 1
        elif self.bulge_type=='jaffe':
            gamma = 2
        elif self.bulge_type=='dehnen':
            gamma = all_params['gamma']
        
        model_goodpix = np.ones_like(self.goodpix)
        central_index = np.argmin((self.x_g-x0)**2+(self.y_g-y0)**2)
        tot =  self._model(self.x_g,          
                           self.y_g,
                           self.N,
                           self.J,
                           self.K,          
                           Mb,                  
                           Rb,                  
                           gamma,                  
                           Md1,                  
                           Rd1,                  
                           f,                 
                           c,   # this is not in log10 
                           self.rho_crit,                 
                           x0,                 
                           y0,                 
                           theta,              
                           incl,                
                           1.0/ML_b,        
                           1.0/ML_d1,       
                           k1,       
                           self.s_grid,
                           self.gamma_grid,
                           self.rho_grid,
                           self.sigma_grid,
                           self.central_integration_radius,
                           self.central_rho,
                           self.central_sigma,
                           self.kin_psf,
                           self.pho_psf,
                           self.indexing, 
                           model_goodpix,
                           central_index,
                           halo_TYPE)
        rho      = tot[:,0]
        rho_B    = tot[:,1]
        rho_D1   = tot[:,2]
        v        = tot[:,3]
        v_B      = tot[:,4]
        v_D1     = tot[:,5]
        sigma    = tot[:,6]
        sigma_B  = tot[:,7]
        sigma_D1 = tot[:,8]
        LM       = tot[:,9]
        LM_B     = tot[:,10]
        LM_D1    = tot[:,11]

        return rho,rho_B,rho_D1,np.zeros_like(rho),v,v_B,v_D1,np.zeros_like(rho),sigma,sigma_B,sigma_D1,np.zeros_like(rho),LM,LM_B,LM_D1,np.zeros_like(rho)


class model_cpu_DD():
    def __init__(self,
                 fast_math,
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
                 rho_crit):
        ''' 
        A class for representing a "Bulge + Disc + Disc + Halo" model on CPU
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------                                           
            Attributes:
                    fast_math: Bool
                        Whether to use fastmath operation of not
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
            Methods:
                    _function_definition():
                        Define all functions depending on the condition above (bulge_type, halo_type ...)
                    likelihood(all_params,
                               *args,
                               **kwargs):
                        Compute likelihood of the model
                    model(all_params,
                               *args,
                               **kwargs):
                        Compute model all outputs

        '''
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data               = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error              = ML_error.copy(order='C')
        self.kin_psf               = kin_psf.copy(order='C')
        self.pho_psf               = pho_psf.copy(order='C')
        self.bulge_type               = bulge_type
        self.indexing               = indexing
        self.goodpix                = goodpix
        self.halo_type              = halo_type

        self.size = self.x_g.shape[0]
        self.goodpix = goodpix

        self.rho_crit = rho_crit
        self.new_N,self.new_J = int(self.size**0.5),int(self.size**0.5)

        self.dx = np.abs(self.x_g[1]-self.x_g[0])

        self.central_integration_radius = self.dx/2**0.5
        self._function_definition_CPU_DD()


    def _function_definition_CPU_DD(self):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Define all functions depending on the condition above (bulge_type, halo_type ...)
            
            Parameters:

            Returns:

        '''
        if self.halo_type=='hernquist':
            from utils_easy import model_DD_all_outputs
            from utils_easy import likelihood_DD
            self._model      = model_DD_all_outputs
            self._likelihood = likelihood_DD
            self.halo_radius_name = 'log10_halo_radius'
        elif self.halo_type=='NFW':
            from utils_easy import model_DD_all_outputs
            from utils_easy import likelihood_DD
            self._model      = model_DD_all_outputs
            self._likelihood = likelihood_DD
            self.halo_radius_name = 'concentration'
        

    def likelihood(self,
                   all_params,
                   *args,
                   **kwargs):
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        f              = all_params['log10_halo_fraction']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        if self.halo_type == 'NFW':
            c              = all_params[self.halo_radius_name]
            halo_TYPE = 1 
        elif self.halo_type == 'hernquist':
            c              = 10**all_params[self.halo_radius_name]
            halo_TYPE = 0 

        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']


        sys_rho        = all_params['sys_rho']

        lk = self._likelihood(self.x_g,          
                              self.y_g,
                              self.N,
                              self.J,
                              self.K,          
                              Md1,                  
                              Rd1,                  
                              Md2,                  
                              Rd2,                  
                              f,                 
                              c,   # this is not in log10       
                              self.rho_crit,           
                              x0,                 
                              y0,                 
                              theta,              
                              incl,                
                              1.0/ML_d1,       
                              1.0/ML_d2,        
                              k1,       
                              k2,
                              sys_rho,
                              self.kin_psf,
                              self.pho_psf,
                              self.indexing, 
                              self.goodpix,
                              halo_TYPE,
                              self.ibrightness_data,
                              self.v_data,
                              self.sigma_data,
                              self.ML_data,
                              self.ibrightness_error,
                              self.v_error,
                              self.sigma_error,
                              self.ML_error) 
        return lk 

    def model(self,
              all_params,
              *args,
              **kwargs):

        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        f              = all_params['log10_halo_fraction']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        if self.halo_type == 'NFW':
            c              = all_params[self.halo_radius_name]
            halo_TYPE = 1 
        elif self.halo_type == 'hernquist':
            c              = 10**all_params[self.halo_radius_name]
            halo_TYPE = 0 

        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']


        model_goodpix = np.ones((self.N*self.J))

        tot =  self._model(self.x_g,          
                           self.y_g,
                           self.N,
                           self.J,
                           self.K,          
                           Md1,                  
                           Rd1,                  
                           Md2,                  
                           Rd2,                  
                           f,                 
                           c,   # this is not in log10 
                           self.rho_crit,                 
                           x0,                 
                           y0,                 
                           theta,              
                           incl,                
                           1.0/ML_d1,       
                           1.0/ML_d2,        
                           k1,       
                           k2,
                           self.kin_psf,
                           self.pho_psf,
                           self.indexing, 
                           model_goodpix,
                           halo_TYPE)
        rho      = tot[:,0]
        rho_D1   = tot[:,1]
        rho_D2   = tot[:,2]
        v        = tot[:,3]
        v_D1     = tot[:,4]
        v_D2     = tot[:,5]
        sigma    = tot[:,6]
        sigma_D1 = tot[:,7]
        sigma_D2 = tot[:,8]
        LM       = tot[:,9]
        LM_D1    = tot[:,10]
        LM_D2    = tot[:,11]

        return rho,np.zeros_like(rho),rho_D1,rho_D2,v,np.zeros_like(rho),v_D1,v_D2,sigma,np.zeros_like(rho),sigma_D1,sigma_D2,LM,np.zeros_like(rho),LM_D1,LM_D2



class model_cpu_BH():
    def __init__(self,
                 fast_math,
                 bulge_type,
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
                 psf):
        ''' 
        '''
        
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data                  = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error                 = ML_error.copy(order='C')
        self.psf                      = psf.copy(order='C')
        self.bulge_type               = bulge_type

        if self.bulge_type=='hernquist':
            from utils_easy import model_hernquist_BH
            from utils_easy import likelihood_hernquist_BH
            self._model      = model_hernquist_BH 
            self._likelihood = likelihood_hernquist_BH 
        elif self.bulge_type=='jaffe':
            print('Error: Jaffe + BH is not implemented')
            exit()
        else:
            print('Error')
            exit()


    def likelihood(self,
                   all_params,
                   *args):
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        Mh             = all_params['log10_halo_mass']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        Rh             = all_params['log10_halo_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']
        sys_rho        = all_params['sys_rho']
        lk = self._likelihood(
                              self.x_g,
                              self.y_g,
                              Mbh,
                              Mb,
                              Rb,
                              Md1,
                              Rd1,
                              Md2,
                              Rd2,
                              Mh,
                              Rh,
                              x0,
                              y0,
                              theta,
                              incl,
                              1.0/ML_b,
                              1.0/ML_d1,
                              1.0/ML_d2,
                              k1,
                              k2,
                              sys_rho,
                              self.psf,
                              self.ibrightness_data,
                              self.v_data,
                              self.sigma_data,
                              self.ML_data,
                              self.ibrightness_error,
                              self.v_error,
                              self.sigma_error,
                              self.ML_error)
        return lk 

    def model(self,
              all_params,
              *args):
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Md2            = all_params['log10_disc2_mass']
        Mh             = all_params['log10_halo_mass']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rd2            = all_params['log10_disc2_radius']
        Rh             = all_params['log10_halo_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        ML_d2          = all_params['ML_d2']
        k1             = all_params['k1']
        k2             = all_params['k2']
        tot =  self._model(
                           self.x_g,
                           self.y_g,
                           Mbh,
                           Mb,
                           Rb,
                           Md1,
                           Rd1,
                           Md2,
                           Rd2,
                           Mh,
                           Rh,
                           x0,
                           y0,
                           theta,
                           incl,
                           1.0/ML_b,
                           1.0/ML_d1,
                           1.0/ML_d2,
                           k1,
                           k2,
                           self.psf)
        rho,v,sigma,LM = tot[:,0],tot[:,1],tot[:,2],tot[:,3]
        return rho,v,sigma,LM


class model_cpu_BD_BH():
    def __init__(self,
                 fast_math,
                 bulge_type,
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
                 psf):
        ''' 
        '''
        
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data                  = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error                 = ML_error.copy(order='C')
        self.psf                      = psf.copy(order='C')
        self.bulge_type               = bulge_type

        if self.bulge_type=='hernquist':
            from utils_easy import model_hernquist_BD_BH
            from utils_easy import likelihood_hernquist_BD_BH
            self._model      = model_hernquist_BD_BH 
            self._likelihood = likelihood_hernquist_BD_BH 
        elif self.bulge_type=='jaffe':
            print('Error: Jaffe + BH is not implemented')
            exit()
        else:
            print('Error')
            exit()


    def likelihood(self,
                   all_params,
                   *args):
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Mh             = all_params['log10_halo_mass']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rh             = all_params['log10_halo_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        k1             = all_params['k1']
        sys_rho        = all_params['sys_rho']
        lk = self._likelihood(
                              self.x_g,
                              self.y_g,
                              Mbh,
                              Mb,
                              Rb,
                              Md1,
                              Rd1,
                              Mh,
                              Rh,
                              x0,
                              y0,
                              theta,
                              incl,
                              1.0/ML_b,
                              1.0/ML_d1,
                              k1,
                              sys_rho,
                              self.psf,
                              self.ibrightness_data,
                              self.v_data,
                              self.sigma_data,
                              self.ML_data,
                              self.ibrightness_error,
                              self.v_error,
                              self.sigma_error,
                              self.ML_error)
        return lk 

    def model(self,
              all_params,
              *args):
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Md1            = all_params['log10_disc1_mass']
        Mh             = all_params['log10_halo_mass']
        Rb             = all_params['log10_bulge_radius']
        Rd1            = all_params['log10_disc1_radius']
        Rh             = all_params['log10_halo_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        theta          = all_params['orientation']
        incl           = all_params['inclination']
        ML_b           = all_params['ML_b']
        ML_d1          = all_params['ML_d1']
        k1             = all_params['k1']
        tot =  self._model(
                           self.x_g,
                           self.y_g,
                           Mbh,
                           Mb,
                           Rb,
                           Md1,
                           Rd1,
                           Mh,
                           Rh,
                           x0,
                           y0,
                           theta,
                           incl,
                           1.0/ML_b,
                           1.0/ML_d1,
                           k1,
                           self.psf)
        rho,v,sigma,LM = tot[:,0],tot[:,1],tot[:,2],tot[:,3]
        return rho,v,sigma,LM


class model_cpu_B_BH():
    def __init__(self,
                 fast_math,
                 bulge_type,
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
                 psf):
        ''' 
        '''
        
        self.N                        = N
        self.J                        = J
        self.K                        = K
        self.x_g                      = x_g.copy(order='C')
        self.y_g                      = y_g.copy(order='C')
        self.ibrightness_data         = ibrightness_data.copy(order='C')
        self.v_data                   = v_data.copy(order='C')
        self.sigma_data               = sigma_data.copy(order='C')
        self.ML_data                  = ML_data.copy(order='C')
        self.ibrightness_error        = ibrightness_error.copy(order='C')
        self.v_error                  = v_error.copy(order='C')
        self.sigma_error              = sigma_error.copy(order='C')
        self.ML_error                 = ML_error.copy(order='C')
        self.psf                      = psf.copy(order='C')
        self.bulge_type               = bulge_type

        if self.bulge_type=='hernquist':
            from utils_easy import model_hernquist_B_BH
            from utils_easy import likelihood_hernquist_B_BH
            self._model      = model_hernquist_B_BH 
            self._likelihood = likelihood_hernquist_B_BH 
        elif self.bulge_type=='jaffe':
            print('Error: Jaffe + BH is not implemented')
            exit()
        else:
            print('Error')
            exit()


    def likelihood(self,
                   all_params,
                   *args):
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Rb             = all_params['log10_bulge_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        ML_b           = all_params['ML_b']
        sys_rho        = all_params['sys_rho']
        lk = self._likelihood(
                              self.x_g,
                              self.y_g,
                              Mbh,
                              Mb,
                              Rb,
                              x0,
                              y0,
                              1.0/ML_b,
                              sys_rho,
                              self.psf,
                              self.ibrightness_data,
                              self.v_data,
                              self.sigma_data,
                              self.ML_data,
                              self.ibrightness_error,
                              self.v_error,
                              self.sigma_error,
                              self.ML_error)
        return lk 

    def model(self,
              all_params,
              *args):
        Mbh            = all_params['log10_bh_mass']
        Mb             = all_params['log10_bulge_mass']
        Rb             = all_params['log10_bulge_radius']
        x0             = all_params['x0']
        y0             = all_params['y0']
        ML_b           = all_params['ML_b']
        tot =  self._model(
                           self.x_g,
                           self.y_g,
                           Mbh,
                           Mb,
                           Rb,
                           x0,
                           y0,
                           1.0/ML_b,
                           self.psf)
        rho,v,sigma,LM = tot[:,0],tot[:,1],tot[:,2],tot[:,3]
        return rho,v,sigma,LM



#class model_superresolution_cpu():
#    def __init__(self,
#                 device,
#                 bulge_type,
#                 N,     # this must be high resolution grid
#                 J,     # this must be high resolution grid
#                 K,     
#                 x_g,   # this must be high resolution grid
#                 y_g,   # this must be high resolution grid
#                 ibrightness_data,
#                 v_data,
#                 sigma_data,
#                 ML_data,
#                 ibrightness_error,
#                 v_error,
#                 sigma_error,
#                 ML_error,
#                 psf):  
#        '''    
#        '''
#        
#        self.N                        = N
#        self.J                        = J
#        self.K                        = K
#        self.x_g                      = x_g.copy(order='C')
#        self.y_g                      = y_g.copy(order='C')
#        self.ibrightness_data         = ibrightness_data.copy(order='C')
#        self.v_data                   = v_data.copy(order='C')
#        self.sigma_data               = sigma_data.copy(order='C')
#        self.ML_data               = ML_data.copy(order='C')
#        self.ibrightness_error        = ibrightness_error.copy(order='C')
#        self.v_error                  = v_error.copy(order='C')
#        self.sigma_error              = sigma_error.copy(order='C')
#        self.ML_error              = ML_error.copy(order='C')
#        self.psf                      = psf.copy(order='C')
#        self.bulge_type               = bulge_type
#
#        self.counter = 0
#
#        self.device = torch.device('cpu')
#        if self.bulge_type=='hernquist':
#            from utils_easy import model_hernquist
#            from utils_easy import likelihood_superresolution
#            self._model      = model_hernquist 
#            self._likelihood = likelihood_superresolution 
#            #from utils_python import model_BDD
#            #from utils_python import log_likelihood
#            from utils_python import super_resolution
#            #self._model      = model_BDD 
#            #self._likelihood = log_likelihood 
#            self._super_resolution = super_resolution
#        elif self.bulge_type=='jaffe':
#            print('Error: Jaffe model has not be implemented with super resolution')
#        else:
#            print('Error')
#            exit()
#
#        
#        #self._not_prepare_data()
#
#    def _prepare_data(self):
#        torch.set_num_threads(1)
#        # select 2D grid, x_2D and y_2D should be equal to the x and y in the data file
#        self.x_g = self.x_g.reshape((self.N,self.J,self.K))
#        self.y_g = self.y_g.reshape((self.N,self.J,self.K))
#        k = int(np.ceil(self.K**0.5))
#        k_x = int(np.floor(self.K/2))
#        k_y = int(np.floor(k/2))
#        x_2D = self.x_g[:,:,k_x]
#        y_2D = self.y_g[:,:,k_y]
#
#        # this will go in data.py
#        xmin,xmax = np.min(x_2D),np.max(x_2D)
#        ymin,ymax = np.min(y_2D),np.max(y_2D)
#
#        self.new_N,self.new_J = int(np.ceil(self.N/4)),int(np.ceil(self.J/4))
#
#        x = np.linspace(xmin,xmax,self.new_N)
#        y = np.linspace(ymin,ymax,self.new_J)
#
#        x,y = np.meshgrid(x,y,indexing='ij')
#        import data as d
#        d1 = d.data(x.ravel(),y.ravel())
#        self.x_g,self.y_g,_ = d1.refined_grid(size=k)
#
#
#        from network import FSRCNN_net
#        out1,out2 = 30,10
#        self.NN_model = FSRCNN_net(1,         # filter imager grey or RGB
#                       out1,      # filter for feat extraction
#                       out2)      # reduced filter for mapping 
#
#
#        checkpoint = torch.load('trained_model/model_weights.pth', map_location=self.device)
#        self.NN_model.load_state_dict(checkpoint['model_state_dict'])
#        self.NN_model = self.NN_model.to(self.device)
#
#        self.tot_tmp = np.zeros((4,1,self.new_N,self.new_J))
#        self.new_tot = np.zeros((int(self.N*self.J),4))
#        
#
#    def likelihood(self,
#                   all_params,
#                   *args):
#        if self.counter == 0:
#            self._prepare_data()
#            self.counter += 1
#
#        Mb             = all_params['log10_bulge_mass']
#        Md1            = all_params['log10_disc1_mass']
#        Md2            = all_params['log10_disc2_mass']
#        Mh             = all_params['log10_halo_mass']
#        Rb             = all_params['log10_bulge_radius']
#        Rd1            = all_params['log10_disc1_radius']
#        Rd2            = all_params['log10_disc2_radius']
#        Rh             = all_params['log10_halo_radius']
#        x0             = all_params['x0']
#        y0             = all_params['y0']
#        theta          = all_params['orientation']
#        incl           = all_params['inclination']
#        ML_b           = all_params['ML_b']
#        ML_d1          = all_params['ML_d1']
#        ML_d2          = all_params['ML_d2']
#        k1             = all_params['k1']
#        k2             = all_params['k2']
#        sys_rho        = all_params['sys_rho']
#
#        tot =  self._model(
#                           self.x_g,
#                           self.y_g,
#                           Mb,
#                           Rb,
#                           Md1,
#                           Rd1,
#                           Md2,
#                           Rd2,
#                           Mh,
#                           Rh,
#                           x0,
#                           y0,
#                           theta,
#                           incl,
#                           1.0/ML_b,
#                           1.0/ML_d1,
#                           1.0/ML_d2,
#                           k1,
#                           k2,
#                           self.psf)
#        self.tot_tmp = np.zeros((4,1,self.new_N,self.new_J))
#        self.tot_tmp[0,0,:,:] = np.log10(tot[:,0]).reshape((self.new_N,self.new_J))
#        self.tot_tmp[1,0,:,:] = tot[:,1].reshape((self.new_N,self.new_J))
#        self.tot_tmp[2,0,:,:] = tot[:,2].reshape((self.new_N,self.new_J))
#        self.tot_tmp[3,0,:,:] = tot[:,3].reshape((self.new_N,self.new_J))
#        self.tot_tmp = torch.tensor(self.tot_tmp,dtype=torch.float32,device=self.device,requires_grad=False,pin_memory=False)
#        
#        tot = self._super_resolution(self.NN_model,self.tot_tmp)
#        tot = tot.cpu().detach().numpy()
#        self.new_tot[:,0] = tot[0,0,:,:].ravel()
#        self.new_tot[:,1] = tot[1,0,:,:].ravel()
#        self.new_tot[:,2] = tot[2,0,:,:].ravel()
#        self.new_tot[:,3] = tot[3,0,:,:].ravel()
#
#        lk = self._likelihood(sys_rho,
#                              self.new_tot,
#                              self.ibrightness_data,
#                              self.v_data,
#                              self.sigma_data,
#                              self.ML_data,
#                              self.ibrightness_error,
#                              self.v_error,
#                              self.sigma_error,
#                              self.ML_error)
#        return lk 
#
#    def model(self,
#              all_params,
#              *args):
#        if self.counter == 0:
#            self._prepare_data()
#            self.counter += 1
#        Mb             = all_params['log10_bulge_mass']
#        Md1            = all_params['log10_disc1_mass']
#        Md2            = all_params['log10_disc2_mass']
#        Mh             = all_params['log10_halo_mass']
#        Rb             = all_params['log10_bulge_radius']
#        Rd1            = all_params['log10_disc1_radius']
#        Rd2            = all_params['log10_disc2_radius']
#        Rh             = all_params['log10_halo_radius']
#        x0             = all_params['x0']
#        y0             = all_params['y0']
#        theta          = all_params['orientation']
#        incl           = all_params['inclination']
#        ML_b           = all_params['ML_b']
#        ML_d1          = all_params['ML_d1']
#        ML_d2          = all_params['ML_d2']
#        k1             = all_params['k1']
#        k2             = all_params['k2']
#        tot =  self._model(
#                           self.x_g,
#                           self.y_g,
#                           Mb,
#                           Rb,
#                           Md1,
#                           Rd1,
#                           Md2,
#                           Rd2,
#                           Mh,
#                           Rh,
#                           x0,
#                           y0,
#                           theta,
#                           incl,
#                           1.0/ML_b,
#                           1.0/ML_d1,
#                           1.0/ML_d2,
#                           k1,
#                           k2,
#                           self.psf)
#        self.tot_tmp = np.zeros((4,1,self.new_N,self.new_J))
#        self.tot_tmp[0,0,:,:] = np.log10(tot[:,0]).reshape((self.new_N,self.new_J))
#        self.tot_tmp[1,0,:,:] = tot[:,1].reshape((self.new_N,self.new_J))
#        self.tot_tmp[2,0,:,:] = tot[:,2].reshape((self.new_N,self.new_J))
#        self.tot_tmp[3,0,:,:] = tot[:,3].reshape((self.new_N,self.new_J))
#        self.tot_tmp = torch.tensor(self.tot_tmp,dtype=torch.float32,device=self.device,requires_grad=False,pin_memory=False)
#        tot =  self._super_resolution(self.NN_model,self.tot_tmp)
#
#        tot = tot.cpu().detach().numpy()
#
#        rho   = tot[0,0,:,:].ravel()
#        v     = tot[1,0,:,:].ravel()
#        sigma = tot[2,0,:,:].ravel()
#        LM    = tot[3,0,:,:].ravel()
#        return 10**rho,v,sigma,LM
#
#
#class model_superresolution_gpu():
#    def __init__(self,
#                 device,
#                 bulge_type,
#                 N,     # this must be high resolution grid
#                 J,     # this must be high resolution grid
#                 K,     
#                 x_g,   # this must be high resolution grid
#                 y_g,   # this must be high resolution grid
#                 ibrightness_data,
#                 v_data,
#                 sigma_data,
#                 ML_data,
#                 ibrightness_error,
#                 v_error,
#                 sigma_error,
#                 ML_error,
#                 psf):  
#        ''' 
#        This is not full on gpu but it combines this... probably the fastest combination is 
#
#        lr img on GPU
#        upsampling on CPU
#        likelihood eval on CPU   
#        '''
#        
#        self.N                        = N
#        self.J                        = J
#        self.K                        = K
#        self.x_g                      = x_g.copy(order='C')
#        self.y_g                      = y_g.copy(order='C')
#        self.ibrightness_data         = ibrightness_data.copy(order='C')
#        self.v_data                   = v_data.copy(order='C')
#        self.sigma_data               = sigma_data.copy(order='C')
#        self.ML_data               = ML_data.copy(order='C')
#        self.ibrightness_error        = ibrightness_error.copy(order='C')
#        self.v_error                  = v_error.copy(order='C')
#        self.sigma_error              = sigma_error.copy(order='C')
#        self.ML_error              = ML_error.copy(order='C')
#        self.psf                      = psf.copy(order='C')
#        self.bulge_type               = bulge_type
#
#        self.counter = 0
#
#        self.device = torch.device('cuda')
#
#        if self.bulge_type=='hernquist':
#            from utils_easy import likelihood_superresolution
#            self._likelihood = likelihood_superresolution 
#            from utils_python import super_resolution
#            self._super_resolution = super_resolution
#        elif self.bulge_type=='jaffe':
#            print('Error: Jaffe model has not be implemented with super resolution')
#        else:
#            print('Error')
#            exit()
#
#        
#        #self._not_prepare_data()
#    def _function_definition_SR(self):
#        if self.bulge_type=='hernquist':
#            from utils_numba_1D_32bit import herquinst_rho
#            from utils_numba_1D_32bit import herquinst_sigma
#            from utils_numba_1D_32bit import v_H
#            self._bulge_rho   = herquinst_rho 
#            self._bulge_sigma = herquinst_sigma 
#            self._bulge_vel   = v_H 
#        elif self.bulge_type=='jaffe':
#            from utils_numba_1D_32bit import jaffe_rho        
#            from utils_numba_1D_32bit import jaffe_sigma  
#            from utils_numba_1D_32bit import v_J
#            self._bulge_rho   = jaffe_rho
#            self._bulge_sigma = jaffe_sigma
#            self._bulge_vel   = v_J
#        else:
#            print('Error')
#            exit()
#
#        from utils_numba_1D_32bit import rho_D
#        from utils_numba_1D_32bit import v_D
#        from utils_numba_1D_32bit import v_H
#        from utils_numba_1D_32bit import v_tot
#        from utils_numba_1D_32bit import avg_rho_v
#        from utils_numba_1D_32bit import avg_LM
#        from utils_numba_1D_32bit import coordinate_transformation
#        from utils_numba_1D_32bit import Xfunction
#        print('using fast math')
#       
#        self._coordinate_transformation = coordinate_transformation
#        self._Xfunction                 = Xfunction
#        self._rho_D                     = rho_D
#        self._v_D                       = v_D
#        self._halo_vel                  = v_H
#        self._v_tot                     = v_tot
#        self._avg_rho_v                 = avg_rho_v
#        self._avg_LM                    = avg_LM
#
#        from utils_numba_1D_32bit import sum_likelihood
#        self._sum_likelihood            = sum_likelihood
#
#    def _prepare_data(self):
#        torch.set_num_threads(1)
#        # select 2D grid, x_2D and y_2D should be equal to the x and y in the data file
#        self.x_g = self.x_g.reshape((self.N,self.J,self.K))
#        self.y_g = self.y_g.reshape((self.N,self.J,self.K))
#        k = int(np.ceil(self.K**0.5))
#        k_x = int(np.floor(self.K/2))
#        k_y = int(np.floor(k/2))
#        x_2D = self.x_g[:,:,k_x]
#        y_2D = self.y_g[:,:,k_y]
#
#        # this will go in data.py
#        xmin,xmax = np.min(x_2D),np.max(x_2D)
#        ymin,ymax = np.min(y_2D),np.max(y_2D)
#
#        self.new_N,self.new_J = int(np.ceil(self.N/4)),int(np.ceil(self.J/4))
#
#        x = np.linspace(xmin,xmax,self.new_N)
#        y = np.linspace(ymin,ymax,self.new_J)
#
#        x,y = np.meshgrid(x,y,indexing='ij')
#        import data as d
#        d1 = d.data(x.ravel(),y.ravel())
#        self.x_g,self.y_g,_ = d1.refined_grid(size=k)
#
#
#        from network import FSRCNN_net
#        out1,out2 = 30,10
#        self.NN_model = FSRCNN_net(1,         # filter imager grey or RGB
#                       out1,      # filter for feat extraction
#                       out2)      # reduced filter for mapping 
#
#
#        checkpoint = torch.load('trained_model/model_weights.pth', map_location=self.device)
#        self.NN_model.load_state_dict(checkpoint['model_state_dict'])
#        self.NN_model = self.NN_model.to(self.device)
#
#        self.tot_tmp = np.zeros((4,1,self.new_N,self.new_J))
#        self.new_tot = np.zeros((int(self.N*self.J),4))
#        
#    def _init_grid_SR(self):
#        Nthreads_1 = 64
#        Nthreads_3 = 64
#        Nblocks_1 = math.ceil(self.new_N*self.new_J*self.K/Nthreads_1)
#        Nblocks_3 = math.ceil(self.new_N*self.new_J/Nthreads_3)
#
#        self.threadsperblock_1 = (Nthreads_1)
#        self.threadsperblock_3 = (Nthreads_3)
#        self.blockpergrid_1 = (Nblocks_1)
#        self.blockpergrid_3 = (Nblocks_3)
#
#    def _load_to_device_SR(self):
#        self._prepare_data()
#        self.x_device = cuda.to_device(np.float32(self.x_g))
#        self.y_device = cuda.to_device(np.float32(self.y_g))
#        
#        self.r_proj_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.r_true_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.phi_device           = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.all_rhoB_device      = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.all_sigma2_device    = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.all_rhoD1_device     = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.all_rhoD2_device     = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.all_v_device         = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.all_v2_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.Xs_device            = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.v_exp1_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.v_exp2_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.v_bulge_device       = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#        self.v_halo_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
#
#        self.LM_avg_device            = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
#        self.rho_avg_device           = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
#        self.v_avg_device             = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
#        self.sigma_avg_device         = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
#
#        self.tot = cuda.device_array((4,1,self.new_N,self.new_J),dtype=np.float32)
#        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
#
#    def _init_data_SR(self):
#        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
#        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
#        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
#        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
#        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
#        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
#        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
#        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
#        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
#
#    def likelihood(self,
#                   all_params,
#                   *args):
#        ''' 
#        '''
#        Mb             = all_params['log10_bulge_mass']
#        Md1            = all_params['log10_disc1_mass']
#        Md2            = all_params['log10_disc2_mass']
#        Mh             = all_params['log10_halo_mass']
#        Rb             = all_params['log10_bulge_radius']
#        Rd1            = all_params['log10_disc1_radius']
#        Rd2            = all_params['log10_disc2_radius']
#        Rh             = all_params['log10_halo_radius']
#        x0             = all_params['x0']
#        y0             = all_params['y0']
#        theta          = all_params['orientation']
#        incl           = all_params['inclination']
#        ML_b           = all_params['ML_b']
#        ML_d1          = all_params['ML_d1']
#        ML_d2          = all_params['ML_d2']
#        k1             = all_params['k1']
#        k2             = all_params['k2']
#        sys_rho        = all_params['sys_rho']
#        if self.counter==0:
#            self._load_to_device_SR()
#            self._init_grid_SR()
#            self._function_definition_SR()
#            self._init_data_SR()
#            self.counter +=1
#        else: 
#            pass
#
#        Mb           = 10**Mb
#        Md1          = 10**Md1    
#        Md2          = 10**Md2    
#        Mh           = 10**Mh    
#        Rb           = 10**Rb    
#        Rd1          = 10**Rd1    
#        Rd2          = 10**Rd2    
#        Rh           = 10**Rh
#        sys_rho      = 10**sys_rho
#
#        # GRID DEFINITION
#        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
#                                                                                    self.y_device,
#                                                                                    x0,
#                                                                                    y0,
#                                                                                    theta,
#                                                                                    incl,
#                                                                                    self.r_proj_device,
#                                                                                    self.r_true_device,
#                                                                                    self.phi_device)
#
#
#        self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
#                                                                   self.r_proj_device,
#                                                                   self.Xs_device)
#        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
#        self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
#                                                                   Rb,
#                                                                   self.r_proj_device,
#                                                                   self.all_rhoB_device,
#                                                                   self.Xs_device)
#        # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
#        self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mb,
#                                                                     Rb,
#                                                                     self.r_proj_device,
#                                                                     self.all_sigma2_device,
#                                                                     self.Xs_device)
#
#        ## DISC SURFACE DENSITY
#        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
#                                                               Rd1,
#                                                               incl,
#                                                               self.r_true_device,
#                                                               self.all_rhoD1_device)
#
#        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
#                                                               Rd2,
#                                                               incl,
#                                                               self.r_true_device,
#                                                               self.all_rhoD2_device)
#
#        ## DISC VELOCITY 
#        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](Mh,
#                                                             Rh,
#                                                             self.r_true_device,
#                                                             self.v_halo_device)
#        self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
#                                                            Rb,
#                                                            self.r_true_device,
#                                                            self.v_bulge_device)
#        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
#                                                             Rd1,
#                                                             self.r_true_device,
#                                                             self.v_exp1_device)
#        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
#                                                             Rd2,
#                                                             self.r_true_device,
#                                                             self.v_exp2_device)
#        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
#                                                               self.v_exp2_device,
#                                                               self.v_bulge_device,
#                                                               self.v_halo_device,
#                                                               incl,
#                                                               self.phi_device,
#                                                               self.all_v_device,
#                                                               self.all_v2_device)
#
#        #  RHO, V E SIGMA with psf
#        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
#                                                                   self.all_rhoD1_device,
#                                                                   self.all_rhoD2_device,
#                                                                   self.all_v_device,
#                                                                   self.all_v2_device,
#                                                                   self.all_sigma2_device,
#                                                                   self.psf_device,
#                                                                   1.0/ML_b,
#                                                                   1.0/ML_d1,
#                                                                   1.0/ML_d2,
#                                                                   k1,
#                                                                   k2,
#                                                                   self.rho_avg_device,
#                                                                   self.v_avg_device,
#                                                                   self.sigma_avg_device)
#
#
#        self._avg_LM[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
#                                                                self.all_rhoD1_device,
#                                                                self.all_rhoD2_device,
#                                                                self.psf_device,
#                                                                1.0/ML_b,
#                                                                1.0/ML_d1,
#                                                                1.0/ML_d2,
#                                                                self.LM_avg_device)
#
#        #tot = np.zeros((4,1,self.new_N,self.new_J))
#        #tot[0,0,:,:] = np.log10(self.rho_avg_device.copy_to_host()).reshape((self.new_N,self.new_J))
#        #tot[1,0,:,:] = self.v_avg_device.copy_to_host().reshape((self.new_N,self.new_J))
#        #tot[2,0,:,:] = self.sigma_avg_device.copy_to_host().reshape((self.new_N,self.new_J))
#        #tot[3,0,:,:] = self.LM_avg_device.copy_to_host().reshape((self.new_N,self.new_J))
#
#        
#        self.tot[0,0,:,:] = np.log10(self.rho_avg_device).reshape((self.new_N,self.new_J))
#        self.tot[1,0,:,:] = self.v_avg_device.reshape((self.new_N,self.new_J))
#        self.tot[2,0,:,:] = self.sigma_avg_device.reshape((self.new_N,self.new_J))
#        self.tot[3,0,:,:] = self.LM_avg_device.reshape((self.new_N,self.new_J))
#
#        tot = torch.tensor(self.tot,dtype=torch.float32,device=self.device,requires_grad=False,pin_memory=False)
#        tot = self._super_resolution(self.NN_model,tot)
#        #tot = tot.cpu().detach().numpy()
#        #self.new_tot[:,0] = tot[0,0,:,:].ravel()
#        #self.new_tot[:,1] = tot[1,0,:,:].ravel()
#        #self.new_tot[:,2] = tot[2,0,:,:].ravel()
#        #self.new_tot[:,3] = tot[3,0,:,:].ravel()
#
#        # this is in cpu format
#        #lk = self._likelihood(sys_rho,
#        #                      self.new_tot,
#        #                      self.ibrightness_data,
#        #                      self.v_data,
#        #                      self.sigma_data,
#        #                      self.ML_data,
#        #                      self.ibrightness_error,
#        #                      self.v_error,
#        #                      self.sigma_error,
#        #                      self.ML_error)
#        #return lk 
#        self._sum_likelihood[self.blockpergrid_3,self.threadsperblock_3](tot[0,0,:,:].ravel(),
#                                                                        tot[1,0,:,:].ravel(),
#                                                                        tot[2,0,:,:].ravel(),
#                                                                        tot[3,0,:,:].ravel(),
#                                                                        self.ibrightness_data_device,
#                                                                        self.v_data_device,
#                                                                        self.sigma_data_device,
#                                                                        self.ML_data_device,
#                                                                        self.ibrightness_error_device,
#                                                                        self.v_error_device,
#                                                                        self.sigma_error_device,
#                                                                        self.ML_error_device,
#                                                                        sys_rho,
#                                                                        self.lk_device)
#
#        lk = self.lk_device.copy_to_host()
#
#        return lk[0]
#
#
#    def model(self,
#              all_params,
#              *args):
#        Mb             = all_params['log10_bulge_mass']
#        Md1            = all_params['log10_disc1_mass']
#        Md2            = all_params['log10_disc2_mass']
#        Mh             = all_params['log10_halo_mass']
#        Rb             = all_params['log10_bulge_radius']
#        Rd1            = all_params['log10_disc1_radius']
#        Rd2            = all_params['log10_disc2_radius']
#        Rh             = all_params['log10_halo_radius']
#        x0             = all_params['x0']
#        y0             = all_params['y0']
#        theta          = all_params['orientation']
#        incl           = all_params['inclination']
#        ML_b           = all_params['ML_b']
#        ML_d1          = all_params['ML_d1']
#        ML_d2          = all_params['ML_d2']
#        k1             = all_params['k1']
#        k2             = all_params['k2']
#        sys_rho        = all_params['sys_rho']
#        if self.counter==0:
#            self._load_to_device_SR()
#            self._init_grid_SR()
#            self._function_definition_SR()
#            self._init_data_SR()
#            self.counter +=1
#        else: 
#            pass
#
#        Mb           = 10**Mb
#        Md1          = 10**Md1    
#        Md2          = 10**Md2    
#        Mh           = 10**Mh    
#        Rb           = 10**Rb    
#        Rd1          = 10**Rd1    
#        Rd2          = 10**Rd2    
#        Rh           = 10**Rh
#        sys_rho      = 10**sys_rho
#
#        # GRID DEFINITION
#        self._coordinate_transformation[self.blockpergrid_1,self.threadsperblock_1](self.x_device,
#                                                                                    self.y_device,
#                                                                                    x0,
#                                                                                    y0,
#                                                                                    theta,
#                                                                                    incl,
#                                                                                    self.r_proj_device,
#                                                                                    self.r_true_device,
#                                                                                    self.phi_device)
#
#
#        self._Xfunction[self.blockpergrid_1,self.threadsperblock_1](Rb,
#                                                                   self.r_proj_device,
#                                                                   self.Xs_device)
#        ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
#        self._bulge_rho[self.blockpergrid_1,self.threadsperblock_1](Mb,
#                                                                   Rb,
#                                                                   self.r_proj_device,
#                                                                   self.all_rhoB_device,
#                                                                   self.Xs_device)
#        # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
#        self._bulge_sigma[self.blockpergrid_1,self.threadsperblock_1](Mb,
#                                                                     Rb,
#                                                                     self.r_proj_device,
#                                                                     self.all_sigma2_device,
#                                                                     self.Xs_device)
#
#        ## DISC SURFACE DENSITY
#        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
#                                                               Rd1,
#                                                               incl,
#                                                               self.r_true_device,
#                                                               self.all_rhoD1_device)
#
#        self._rho_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
#                                                               Rd2,
#                                                               incl,
#                                                               self.r_true_device,
#                                                               self.all_rhoD2_device)
#
#        ## DISC VELOCITY 
#        self._halo_vel[self.blockpergrid_1,self.threadsperblock_1](Mh,
#                                                             Rh,
#                                                             self.r_true_device,
#                                                             self.v_halo_device)
#        self._bulge_vel[self.blockpergrid_1,self.threadsperblock_1](Mb,
#                                                            Rb,
#                                                            self.r_true_device,
#                                                            self.v_bulge_device)
#        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md1,
#                                                             Rd1,
#                                                             self.r_true_device,
#                                                             self.v_exp1_device)
#        self._v_D[self.blockpergrid_1,self.threadsperblock_1](Md2,
#                                                             Rd2,
#                                                             self.r_true_device,
#                                                             self.v_exp2_device)
#        self._v_tot[self.blockpergrid_1,self.threadsperblock_1](self.v_exp1_device,
#                                                               self.v_exp2_device,
#                                                               self.v_bulge_device,
#                                                               self.v_halo_device,
#                                                               incl,
#                                                               self.phi_device,
#                                                               self.all_v_device,
#                                                               self.all_v2_device)
#
#        #  RHO, V E SIGMA with psf
#        self._avg_rho_v[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
#                                                                   self.all_rhoD1_device,
#                                                                   self.all_rhoD2_device,
#                                                                   self.all_v_device,
#                                                                   self.all_v2_device,
#                                                                   self.all_sigma2_device,
#                                                                   self.psf_device,
#                                                                   1.0/ML_b,
#                                                                   1.0/ML_d1,
#                                                                   1.0/ML_d2,
#                                                                   k1,
#                                                                   k2,
#                                                                   self.rho_avg_device,
#                                                                   self.v_avg_device,
#                                                                   self.sigma_avg_device)
#
#
#        self._avg_LM[self.blockpergrid_3,self.threadsperblock_3](self.all_rhoB_device,
#                                                                self.all_rhoD1_device,
#                                                                self.all_rhoD2_device,
#                                                                self.psf_device,
#                                                                1.0/ML_b,
#                                                                1.0/ML_d1,
#                                                                1.0/ML_d2,
#                                                                self.LM_avg_device)
#
#        #tot = np.zeros((4,1,self.new_N,self.new_J))
#        #tot[0,0,:,:] = np.log10(self.rho_avg_device.copy_to_host()).reshape((self.new_N,self.new_J))
#        #tot[1,0,:,:] = self.v_avg_device.copy_to_host().reshape((self.new_N,self.new_J))
#        #tot[2,0,:,:] = self.sigma_avg_device.copy_to_host().reshape((self.new_N,self.new_J))
#        #tot[3,0,:,:] = self.LM_avg_device.copy_to_host().reshape((self.new_N,self.new_J))
#
#        #tot = torch.tensor(tot,dtype=torch.float32,device=self.device,requires_grad=False,pin_memory=False)
#        #tot = self._super_resolution(self.NN_model,self.tot)
#        #tot = tot.cpu().detach().numpy()
#
#        #rho   = tot[0,0,:,:].ravel()
#        #v     = tot[1,0,:,:].ravel()
#        #sigma = tot[2,0,:,:].ravel()
#        #LM    = tot[3,0,:,:].ravel()
#
#        self.tot[0,0,:,:] = np.log10(self.rho_avg_device).reshape((self.new_N,self.new_J))
#        self.tot[1,0,:,:] = self.v_avg_device.reshape((self.new_N,self.new_J))
#        self.tot[2,0,:,:] = self.sigma_avg_device.reshape((self.new_N,self.new_J))
#        self.tot[3,0,:,:] = self.LM_avg_device.reshape((self.new_N,self.new_J))
#
#        tot = torch.tensor(self.tot,dtype=torch.float32,device=self.device,requires_grad=False,pin_memory=False)
#        tot = self._super_resolution(self.NN_model,tot)
#
#        rho   = tot[0,0,:,:].cpu().detach().numpy().ravel()
#        v     = tot[1,0,:,:].cpu().detach().numpy().ravel()
#        sigma = tot[2,0,:,:].cpu().detach().numpy().ravel()
#        LM    = tot[3,0,:,:].cpu().detach().numpy().ravel()
#
#        return 10**rho,v,sigma,LM
#
#
#
#
class model(model_gpu,model_cpu,model_gpu_BH,model_cpu_BH,
            model_gpu_BD,model_cpu_BD,model_gpu_BD_BH,model_cpu_BD_BH,model_gpu_DD,
            model_gpu_B_BH,model_cpu_B_BH,model_cpu_DD):#,
            #model_superresolution_cpu,model_superresolution_gpu):
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
            Methods:
                    likelihood(all_params,
                               *args,
                               **kwargs):
                        Compute likelihood of the model
                    model(all_params,
                               *args,
                               **kwargs):
                        Compute model all outputs

        '''
        self.device  = device
        # include[0] if True BH is present if false without BH
        # include[1] if True Bulge is present if false without Bulge
        # include[2] if True First disc is present if false without First disc
        # include[3] if True Second disc is present if false without Second disc
        # include[4] if True Halo is present if false without Halo
        self.include = include
        if self.device == 'GPU':
            if self.include[0] == True and self.include[3]==True and self.include[1] == True \
               and self.include[2] == True and self.include[4] == True:
                # BH+B+D1+D2+H
                print('Error: This model is actually UNSTABLE')
                exit()
                model_gpu_BH.__init__(self,
                                   fast_math,
                                   bulge_type,
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
                                   psf)
                self.__likelihood = model_gpu_BH.likelihood
                self.__model      = model_gpu_BH.model 
            elif self.include[0] == False and self.include[3]==True and self.include[1] == True \
                 and self.include[2] == True and self.include[4] == True:
                # B+D1+D2+H
                model_gpu.__init__(self,
                                   gpu_device,
                                   fast_math,
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
                self.__likelihood = model_gpu.likelihood
                self.__model      = model_gpu.model  
            elif self.include[0] == True and self.include[3]==False and self.include[1] == True \
                 and self.include[2] == True and self.include[4] == True:
                # BH+B+D1+H
                print('Error: This model is actually UNSTABLE')
                exit()
                model_gpu_BD_BH.__init__(self,
                                   fast_math,
                                   bulge_type,
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
                                   psf)
                self.__likelihood = model_gpu_BD_BH.likelihood
                self.__model      = model_gpu_BD_BH.model                 
            
            elif self.include[0] == False and self.include[3]==False and self.include[1] == True \
                 and self.include[2] == True and self.include[4] == True:
                # B+D1+H
                model_gpu_BD.__init__(self,
                                   gpu_device,
                                   fast_math,
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
                self.__likelihood = model_gpu_BD.likelihood
                self.__model      = model_gpu_BD.model 
            elif self.include[0] == False and self.include[1] == False and self.include[2]==True \
                 and self.include[3]==True:
                # D1+D2
                model_gpu_DD.__init__(self,
                                   gpu_device,
                                   fast_math,
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
                self.__likelihood = model_gpu_DD.likelihood
                self.__model      = model_gpu_DD.model   
            
            elif self.include[0] == True and self.include[1] == True and self.include[2]==False \
                 and self.include[3]==False:
                # BH+B
                print('Error: This model is actually UNSTABLE')
                exit()
                model_gpu_B_BH.__init__(self,
                                   fast_math,
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
                self.__likelihood = model_gpu_B_BH.likelihood
                self.__model      = model_gpu_B_BH.model                 
            else:
                print('Error this combination has not be implemented yet!')
                exit()

        elif self.device == 'CPU':
            if self.include[0] == True and self.include[3]==True and self.include[1] == True \
               and self.include[2] == True and self.include[4] == True:
                # BH+B+D1+D2+H
                print('Error: This model is actually UNSTABLE')
                exit()
                model_cpu_BH.__init__(self,
                                   fast_math,
                                   bulge_type,
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
                                   psf) 
                self.__likelihood = model_cpu_BH.likelihood
                self.__model      = model_cpu_BH.model
            elif self.include[0] == False and self.include[3]==True and self.include[1] == True \
               and self.include[2] == True and self.include[4] == True:
                # B+D1+D2+H
                model_cpu.__init__(self,
                                   fast_math,
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
                self.__likelihood = model_cpu.likelihood
                self.__model      = model_cpu.model
            elif self.include[0] == True and self.include[3]==False and self.include[1] == True \
               and self.include[2] == True and self.include[4] == True:
                # BH+B+D1+H
                print('Error: This model is actually UNSTABLE')
                exit()
                model_cpu_BD_BH.__init__(self,
                                   fast_math,
                                   bulge_type,
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
                                   psf)
                self.__likelihood = model_cpu_BD_BH.likelihood
                self.__model      = model_cpu_BD_BH.model                 
            
            elif self.include[0] == False and self.include[3]==False and self.include[1] == True \
               and self.include[2] == True and self.include[4] == True:
                # B+D1+H
                model_cpu_BD.__init__(self,
                                   fast_math,
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
                self.__likelihood = model_cpu_BD.likelihood
                self.__model      = model_cpu_BD.model 
            elif self.include[0] == False and self.include[1] == False and self.include[2]==True \
                 and self.include[3]==True:
                # D1+D2+H
                model_cpu_DD.__init__(self,
                                   fast_math,
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
                self.__likelihood = model_cpu_DD.likelihood
                self.__model      = model_cpu_DD.model 
            
            elif self.include[0] == True and self.include[1] == True and self.include[2]==False \
                 and self.include[3]==False:
                # BH+B
                print('Error: This model is actually UNSTABLE')
                exit()
                model_cpu_B_BH.__init__(self,
                                   fast_math,
                                   bulge_type,
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
                                   psf)
                self.__likelihood = model_cpu_B_BH.likelihood
                self.__model      = model_cpu_B_BH.model                 
            else:
                print('Error this combination has not be implemented yet!')
                exit()
        
        elif self.device == 'super resolution CPU':
            pass
            ## B+D1+D2+H
            #if self.include[0] == False and self.include[3]==True and self.include[1] == True \
            #   and self.include[2] == True and self.include[4] == True:
            #    model_superresolution_cpu.__init__(self,
            #                       'CPU',
            #                       bulge_type,
            #                       N,
            #                       J,
            #                       K,
            #                       x_g,
            #                       y_g,
            #                       ibrightness_data,
            #                       v_data,
            #                       sigma_data,
            #                       ML_data,
            #                       ibrightness_error,
            #                       v_error,
            #                       sigma_error,
            #                       ML_error,
            #                       psf)
            #    self.__likelihood = model_superresolution_cpu.likelihood
            #    self.__model      = model_superresolution_cpu.model 

            #else:
            #    print('Error this combination has not be implemented yet!')
            #    exit()
        elif self.device == 'super resolution GPU':
            pass
            ## B+D1+D2+H
            #if self.include[0] == False and self.include[3]==True and self.include[1] == True \
            #   and self.include[2] == True and self.include[4] == True:
            #    model_superresolution_gpu.__init__(self,
            #                       'GPU',
            #                       bulge_type,
            #                       N,
            #                       J,
            #                       K,
            #                       x_g,
            #                       y_g,
            #                       ibrightness_data,
            #                       v_data,
            #                       sigma_data,
            #                       ML_data,
            #                       ibrightness_error,
            #                       v_error,
            #                       sigma_error,
            #                       ML_error,
            #                       psf)
            #    self.__likelihood = model_superresolution_gpu.likelihood
            #    self.__model      = model_superresolution_gpu.model 
#
            #else:
            #    print('Error this combination has not be implemented yet!')
            #    exit()

    def likelihood(self,
                   all_params,
                   *args,
                   **kwargs):
        return self.__likelihood(self,
                                 all_params,
                                 *args,
                                 **kwargs)

    def model(self,
              all_params,
              **kwargs):
        return self.__model(self,
                            all_params,
                            **kwargs)


if __name__ == '__main__':
    # Example on how to use the model class
    import data as d
    import matplotlib.pyplot as plt 
    from astropy.cosmology import FlatLambdaCDM


    # Declare options (you will specify them in the config.yaml file)
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
    x,y = np.meshgrid(x,y,indexing = 'ij')
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



    # Declare all parameters
    Mb,Rb,Md1,Rd1,Md2,Rd2,f,c = 10.15,-1.348,9.80,0.0245,9.667,0.880,-1,8.
    x0,y0,theta,incl = -0.06010,-0.0226,2.68746,1.1323
    k1,k2,ML_b,ML_d1,ML_d2 = 0.5,0.9,2.20,0.636,0.101
    sys_rho = -1.0
    gamma = 1.5


    all_par = {'log10_bulge_mass':Mb,
               'log10_disc1_mass':Md1,
               'log10_disc2_mass':Md2,
               'log10_halo_fraction':f,
               'log10_bulge_radius':Rb,
               'log10_disc1_radius':Rd1,
               'log10_disc2_radius':Rd2,
               'concentration':c,
               'x0':x0,
               'y0':y0,
               'orientation':theta,
               'inclination':incl,
               'ML_b':ML_b,
               'ML_d1':ML_d1,
               'ML_d2':ML_d2,
               'k1':k1,
               'k2':k2,
               'sys_rho':sys_rho,
               'gamma':gamma}
    

    
    '''

     # Create model obj
    my_model_GPU = model('GPU',
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

    rho_GPU,rho_B_GPU,rho_D1_GPU,rho_D2_GPU,\
    v_GPU,v_B_GPU,v_D1_GPU,v_D2_GPU,\
    sigma_GPU,sigma_B_GPU,sigma_D1_GPU,sigma_D2_GPU,\
    LM_GPU,LM_B_GPU,LM_D1_GPU,LM_D2_GPU = my_model_GPU.model(all_par) 
    
    #lk = my_model_GPU.likelihood(all_par)   


    fig,ax,pmesh,cbar = data_obj.brightness_map(np.log10(rho_GPU))
    B_lim = pmesh.get_clim()
    fig.savefig('test_model/B_gpu.png')
    fig.show()


    fig,ax,pmesh,cbar = data_obj.velocity_map(v_GPU)
    v_lim = pmesh.get_clim()
    fig.savefig('test_model/v_gpu.png')
    fig.show()
   
    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_GPU)
    sigma_lim = pmesh.get_clim()
    fig.savefig('test_model/sigma_gpu.png')
    fig.show()

    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_B_GPU)
    fig.show()

    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_D1_GPU)
    fig.show()

    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_D2_GPU)
    fig.show()
   
    fig,ax,pmesh,cbar = data_obj.ML_map(1.0/LM_GPU)
    ML_lim = pmesh.get_clim()
    fig.savefig('test_model/ML_gpu.png')
    fig.show()
    '''

    # Create model obj
    my_model = model('CPU',
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

    rho,rho_B,rho_D1,rho_D2,\
    v,v_B,v_D1,v_D2,\
    sigma,sigma_B,sigma_D1,sigma_D2,\
    LM,LM_B,LM_D1,LM_D2 = my_model.model(all_par) 

    #lk = my_model.likelihood(all_par)   


    # create mock dataset
    tmp = np.concatenate((x.reshape(-1,1),
                          y.reshape(-1,1),
                          x.reshape(-1,1),
                          y.reshape(-1,1),
                          rho.reshape(-1,1),
                          v.reshape(-1,1),
                          sigma.reshape(-1,1),
                          1.0/LM.reshape(-1,1),
                          0.05*rho.reshape(-1,1),
                          0.05*np.abs(v).reshape(-1,1),
                          0.05*sigma.reshape(-1,1),
                          0.05*1.0/LM.reshape(-1,1),
                          goodpix.reshape(-1,1)),axis=1)
    np.save('mock_data.npy',tmp)

    fig,ax,pmesh,cbar = data_obj.brightness_map(np.log10(rho))
    #pmesh.set_clim(B_lim)
    fig.savefig('test_model/B_cpu.png')
    fig.show()


    fig,ax,pmesh,cbar = data_obj.velocity_map(v)
    #pmesh.set_clim(v_lim)
    fig.savefig('test_model/v_cpu.png')
    fig.show()
   
    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma)
    #pmesh.set_clim(sigma_lim)
    fig.savefig('test_model/sigma_cpu.png')
    fig.show()

    fig,ax,pmesh,cbar = data_obj.ML_map(1.0/LM)
    #pmesh.set_clim(ML_lim)
    fig.savefig('test_model/ML_cpu.png')
    fig.show()

    plt.show()



 