import numpy as np 
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

def grid_conversion(x,y,z):
    ''' 
    Utils Function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    conversion from arcsec to kpc assuming FlatLambdaCDM cosmology with H0 = 70.0 (Km/s)/Mpc and Om0 = 0.3
        
        Parameters:
            x: 1D arr float32
                x position in arcsecond
            y: 1D arr float32
                y position in arcsecond
            z: float32
                redshift
        Returns:
            x: 1D arr float32
                x position in kpc
            y: 1D arr float32
                y position in kpc
    '''

    #coordinate in radiants rad = (pi/180*3600)*arcsec
    x = (np.pi/(180*3600)) * x
    y = (np.pi/(180*3600)) * y

    H0 = 70.0           #(Km/s)/Mpc
    Om0 = 0.3
    cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)

    D = cosmo.angular_diameter_distance(z)       #distance in Mpc
    D = D.value
    D = D * 1e3         #distance in Kpc
    x = D * np.tan(x)
    y = D * np.tan(y)
    D_angular = D

    return x,y

def psf_sigma_FWHM(FWHM_arcsec,z):
    ''' 
    Utils Function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    conversion of FWHM of the psf from arcsec to kpc 
    assuming FlatLambdaCDM cosmology with H0 = 70.0 (Km/s)/Mpc and Om0 = 0.3
        
        Parameters:
            FWHM_arcsec: float32
                FWHM of the psf in arcsec
            z: float32
                redshift
        Returns:
            sigma_kpc: float32
                sigma of the psf in kpc (assuming Gaussian)
            FWHM_kpc: float32
                FWHM of the psf in kpc (assuming Gaussian)
    '''
    H0 = 70.0           #(Km/s)/Mpc
    Om0 = 0.3
    cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
    D = cosmo.angular_diameter_distance(z)     #distance in Mpc
    D = D.value
    D = D * 1e3         #distance in Kpc

    sigma_arcsec = FWHM_arcsec/2.355

    FWHM_kpc  = np.tan((np.pi/(180*3600)) * FWHM_arcsec) * D
    sigma_kpc = np.tan((np.pi/(180*3600)) * sigma_arcsec) * D

    return(sigma_kpc,FWHM_kpc)



def nMgy_to_cgs(x):
    ''' 
    Utils Function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    conversion of flux from nanomaggies (SDSS) to cgs (erg/cm^2/s/Hz)
        
        Parameters:
            x: 1D arr float32
                flux in nanomaggies
        Returns:
            x_cgs: 1D arr float32
                flux in erg/s*cm^2*Hz 
    '''
    # conversion factor from nanomaggies to jansky
    conv_nMgy_Jy = 3.631e-6
    x_Jy = x * conv_nMgy_Jy

    # conversion factor from jansky to erg/s*cm^2*Hz 
    conv_Jy_cgs = 1e-23
    x_cgs = x_Jy * conv_Jy_cgs

    return x_cgs

def conversion(x,F_sun,z):
    ''' 
    Utils Function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    conversion of flux from cgs (erg/cm^2/s/Hz or erg/cm^2/s/AA) to brightness in  
    L_sun/pix^2, L_sun/arcsec^2 or L_sun/kpc^2 depending on the units of x
        
        Parameters:
            x: 1D arr float32
                flux in erg/s/cm^2/AA or erg/s/cm^2/Hz  
                (note that real units of x is erg/s/cm^2/Hz/(pix^2 or arcsec^2 or kpc^2 or pc^2)
            F_sun: float32
                sun flux in the same band of x at 10 pc (must have same units as x) 
        Returns:
            L_scaled: 1D arr float32
                luminosity in L_sun/pix^2, L_sun/arcsec^2 or L_sun/kpc^2 depending on the units of x
    '''


    dLE   = 3.08568e19                  # 10 pc in cm distance of the absolute magnitude
    L_sun = F_sun * 4 * np.pi * dLE**2  # Solar luminosity in that band erg/s/Hz or erg/s/AA

    H0 = 70.0                            #(Km/s)/Mpc
    Om0 = 0.3                            #Omega matter 0
    cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
    D = cosmo.luminosity_distance(z)
    D = D.value
    D = D * 1e6                   # distance in pc
    D = D * 3.08568e18            # distance in cm 
    L = 4 * np.pi * D**2 * x      # luminosity in erg/s/Hz erg/s/AA

    L_scaled = L/L_sun            # Luminosity in L_sun (per spaxel, i.e. per pixel, or per arcsec or per kpc or ...) 

    L_scaled = L_scaled           # Surface brightness in L_sun/ (pix^2  or arcsec^2 or kpc^2 or ...)


    return L_scaled

class data:

    def __init__(self,
                 x,
                 y,
                 *args, **kwargs):
        ''' 
        A class for data handling (grid manipolation) and plotting.
        WE ASSUME, SINCE WE ARE WORKING WITH MANGA THAT x and y has the same dimension
        and that they have the same number of pixels in x and y. N=J=x.shape[0]**0.5.
        x coordinates are assumed to be sorted from max to min
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------                                           
            Attributes:
                    x: 1D arr float32
                        x positions of the grid in kpc or arcsec (computed from in data.py)
                    y: 1D arr float32
                        y positions of the grid in kpc or arcsec (computed from in data.py)
            Methods:
                    refined_grid_indexing(sigma_psf,N_px,resolution):
                        Compute grids starting from x,y using N_px for the convolution and possibly 
                        increasing the resolution. Compute also weights of the psf.

                    plot_2D(Z,xlabel,ylabel,zlabel):
                        2D pcolormesh plot

                    plot_3D(Z,xlabel,ylabel,zlabel):
                        3D plot

                    brightness_map(brightness,xlabel,ylabel,zlabel):
                        brightness 2D plot (useful if using default labels)

                    velocity_map(brightness,xlabel,ylabel,zlabel):
                        velocity 2D plot (useful if using default labels)

                    dispersion_map(brightness,xlabel,ylabel,zlabel):
                        dispersion 2D plot (useful if using default labels)

                    ML_map(brightness,xlabel,ylabel,zlabel):
                        Mass to light 2D plot (useful if using default labels)

                    brightness_3D(brightness,xlabel,ylabel,zlabel):
                        brightness 3D plot (useful if using default labels)

                    velocity_3D(brightness,xlabel,ylabel,zlabel):
                        velocity 3D plot (useful if using default labels)

                    dispersion_3D(brightness,xlabel,ylabel,zlabel):
                        dispersion 3D plot (useful if using default labels)

                    ML_3D(brightness,xlabel,ylabel,zlabel):
                        Mass to light 3D plot (useful if using default labels)

                    all_maps_kpc(brightness_data,brightness_model,v_data,v_model,sigma_data,sigma_model,ML_data,ML_model,
                                 brightness_error,v_error,sigma_error,ML_error,plot_ML=True,lim='data',vmin=None,vmax=None):
                        Combine 2D maps of 3X3 or 3X4 plot with labels in kpc

                    all_maps_arcsec(brightness_data,brightness_model,v_data,v_model,sigma_data,sigma_model,ML_data,ML_model,
                                    v_error,sigma_error,ML_error,plot_ML=True,lim='data',vmin=None,vmax=None):
                        Combine 2D maps of 3X3 or 3X4 plot with labels in arcsec

                    all_1D_profile(arcsec_to_kpc,r_avg_brightness,brightness_avg_data,brightness_avg_model,r_avg_v,v_avg_data,
                                    v_avg_model,r_avg_sigma,sigma_avg_data,sigma_avg_model,r_avg_ML,ML_avg_data,ML_avg_model,
                                    brightness_avg_error,v_avg_error,sigma_avg_error,ML_avg_error,plot_ML=True):
                        Combine azimuthally averaged profiles on 1X3 or 1X4 plot with labels both in arcsec and kpc

                    all_1D_profile_all_outputs(arcsec_to_kpc,r_avg_brightness,brightness_avg_data,brightness_avg_model,
                                                   B_B_avg,B_D1_avg,B_D2_avg,r_avg_v,v_avg_data,v_avg_model,v_B_avg,v_D1_avg,v_D2_avg,
                                                   r_avg_sigma,sigma_avg_data,sigma_avg_model,sigma_B_avg,sigma_D1_avg,sigma_D2_avg,r_avg_ML,
                                                   ML_avg_data,ML_avg_model,ML_B_avg,ML_D1_avg,ML_D2_avg,brightness_avg_error,v_avg_error,
                                                   sigma_avg_error,ML_avg_error,plot_ML=True):
                        Combine azimuthally averaged profiles (of each component) on 1X3 or 1X4 plot with labels both in arcsec and kpc
        Note:
            WE ASSUME, SINCE WE ARE WORKING WITH MANGA THAT x and y has the same dimension
            and that they have the same number of pixels in x and y. N=J=x.shape[0]**0.5.
            x coordinates are assumed to be sorted from max to min
        '''

        self.x          = x
        self.y          = y

        #min and max of the grid
        self.xmin, self.xmax = np.min(x), np.max(x)
        self.ymin, self.ymax = np.min(y), np.max(y)


        self.J,self.N = int(self.x.shape[0]**0.5),int(self.x.shape[0]**0.5)
        #vectors with only the numbers without repetition
        #x_true has dimension N, while y_true has dimension J
        self.x_true = self.x[0:self.N]  #x[0 : J*N : J]
        self.y_true = self.y[0:self.J*self.N :self.J] 

        #dimension of the grid, dx size N-1 and dy size J-1. The grid is not supposed to be const
        self.dx = abs(self.x_true[0]-self.x_true[1])
        self.dy = self.dx


       

#    def refined_grid(self,sigma_psf=0.2664348539150563,N_px=10):
#
#        self.x_true = np.linspace(self.xmin,self.xmax,self.J)
#        self.y_true = np.linspace(self.ymin,self.ymax,self.N)
#        
#        self.x_true = self.x_true[::-1]
#        x,y = np.meshgrid(self.x_true,self.y_true,indexing = 'ij')
#        x,y = x.ravel(),y.ravel()
#        
#        self.dx = abs(self.x_true[1]-self.x_true[0])
#        
#        self.k = 2*N_px+1
#        self.K = int(self.k*self.k)
#        
#        self.new_true_x = np.linspace(-N_px*self.dx,N_px*self.dx,2*N_px+1)
#        self.new_true_y = np.linspace(-N_px*self.dx,N_px*self.dx,2*N_px+1)
#
#        self.new_true_x = self.new_true_x[::-1]
#
#        print('Number of points for the convolution is {}'.format(self.K))
#
#        x_rand = np.zeros((self.x.shape[0],self.K))
#        y_rand = np.zeros((self.x.shape[0],self.K))
#
#        x_tmp_g,y_tmp_g = np.meshgrid(self.new_true_x,self.new_true_y,indexing='ij')
#        x_tmp_g,y_tmp_g = x_tmp_g.ravel(),y_tmp_g.ravel()
#        r = np.sqrt(x_tmp_g**2+y_tmp_g**2)
#
#        for i in range(self.x.shape[0]):
#            x_rand[i,:] = x_tmp_g+x[i]
#            y_rand[i,:] = y_tmp_g+y[i]
#
#        x_rand,y_rand = x_rand.ravel(),y_rand.ravel()
#        psf    = np.exp(-r*r/(2*sigma_psf*sigma_psf))
#
#        return(x_rand,y_rand,psf,self.K)


    def refined_grid_indexing(self,sigma_psf=0.2664348539150563,N_px=10,resolution=1):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Compute grids starting from x,y using N_px for the convolution and possibly 
        increasing the resolution. Compute also weights of the psf.
            
            Parameters:
                sigma_psf: float32
                    sigma of the psf in kpc assuming Gaussian
                N_px: int16
                    number of pixels to consider in the convolution
                resolution: int16
                    use pixel (resolution=1) or subpixel (resolution>1) resolution for
                    psf convolution

            Returns:
                new_grid_x_flat: 1D arr float32
                    new grid in x. It covers the same points as self.x but it has 
                    N_px pixels more in each direction.
                new_grid_y_flat: 1D arr float32
                    new grid in y. It covers the same points as self.y but it has 
                    N_px pixels more in each direction.
                psf: 1D arr float32
                    Weights of the psf. It is a vector of dimension K. 
                    Since the dimension of the pixels is constant it is not necessary to compute
                    weights for all the pixels.
                K: float32
                    Number of pixels to consider for the psf convolution. The value at point x[i],y[i]
                    is the weighted average over K points around x[i],y[i]
                indexing: 1D arr int16
                    Vector of K x N x J points. The first indexing[:K] numbers are the indexes of the points
                    to be considere for the psf. It means that for example:
                    rho(x[0],y[0]) is the "average" of psf*rho(new_grid_x_flat[:K],new_grid_y_flat[:K]) 
        Note:
            x coordinates are from max to min
        '''
        print('remember that x coordinates are assumed to be from max to min')

        if not isinstance(N_px, int):
            print('error N_px must be integer')
            exit()
        if not isinstance(resolution,int):
            print('error resolution must be integer')
            exit()

        if N_px == 0:
            self.x_true = np.linspace(self.xmin,self.xmax,self.J)
            self.y_true = np.linspace(self.ymin,self.ymax,self.N)
            ind_min_x     = np.argmin(self.x_true*self.x_true)
            ind_min_y     = np.argmin(self.y_true*self.y_true)
            self.x_true = self.x_true - self.x_true[ind_min_x] 
            self.y_true = self.y_true - self.y_true[ind_min_y]
            self.x_true =self.x_true[::-1]
            new_grid_x,new_grid_y = np.meshgrid(self.x_true,self.y_true)

            return new_grid_x.ravel(),new_grid_y.ravel(),np.ones(1),1,np.arange(self.x.shape[0],dtype=np.int32)

        self.x_true = np.linspace(self.xmin,self.xmax,self.J)
        self.y_true = np.linspace(self.ymin,self.ymax,self.N)

        self.dx = abs(self.x_true[1]-self.x_true[0])
        hr_vec = np.linspace(0.,self.dx,resolution+1)
        #self.dx = np.round(self.dx,decimals=4)

        new_true_x_tmp = np.linspace(self.xmin-N_px*self.dx,self.xmax+N_px*self.dx,self.J+2*N_px)
        new_true_y_tmp = np.linspace(self.ymin-N_px*self.dx,self.ymax+N_px*self.dx,self.N+2*N_px)

        # Make sure it is in zero
        ind_min_new_x = np.argmin(new_true_x_tmp*new_true_x_tmp)
        ind_min_new_y = np.argmin(new_true_y_tmp*new_true_y_tmp)
        ind_min_x     = np.argmin(self.x_true*self.x_true)
        ind_min_y     = np.argmin(self.y_true*self.y_true)
        new_true_x_tmp= new_true_x_tmp-new_true_x_tmp[ind_min_new_x]
        new_true_y_tmp= new_true_y_tmp-new_true_y_tmp[ind_min_new_y] 
        self.x_true = self.x_true - self.x_true[ind_min_x] 
        self.y_true = self.y_true - self.y_true[ind_min_y]

        self.new_true_x = np.array([])
        self.new_true_y = np.array([])
        if resolution < 1:
            print('Error: resolution lower than pixel size is not allowed ')
            exit()
        elif resolution >1:
            hr_vec = hr_vec[:-1]
            for i in range(new_true_x_tmp.shape[0]):
                self.new_true_x = np.append(self.new_true_x,hr_vec+new_true_x_tmp[i])
            for i in range(new_true_y_tmp.shape[0]):
                self.new_true_y = np.append(self.new_true_y,hr_vec+new_true_y_tmp[i])
        else: 
            self.new_true_x = np.copy(new_true_x_tmp)
            self.new_true_y = np.copy(new_true_y_tmp)
        
        # for this case coordinate are from max to min in the x axis
        self.x_true =self.x_true[::-1]
        self.new_true_x = self.new_true_x[::-1]
        new_grid_x,new_grid_y = np.meshgrid(self.new_true_x,self.new_true_y)#,indexing = 'ij')
        new_N,new_J = new_grid_x.shape[0],new_grid_x.shape[1]
        
        x = new_grid_x[N_px:-N_px,N_px:-N_px]
        y = new_grid_y[N_px:-N_px,N_px:-N_px]
        
        new_grid_x_flat,new_grid_y_flat = new_grid_x.ravel(),new_grid_y.ravel()


        # questa posso pensare di prenderla da un sottoinsieme dell'altra
        #x,y = np.meshgrid(self.x_true,self.y_true)#,indexing = 'ij')
        x,y = x.ravel(),y.ravel()

        indexing = np.zeros((x.shape[0],(2*N_px*resolution+1)*(2*N_px*resolution+1)),dtype=np.int32)

        integer_index = np.array([i for i in range(-N_px*resolution,resolution*N_px+1)])
        for i in range(x.shape[0]):
            # Per questioni di confronto numerico faccio così:
            #       - scelgo i punti che distano (N_px+1)*dx dal centro
            #       - li ordino per distanza dal centro 
            #       - scelgo i (2N_px+1)*(2N_px+1) più vicini
            #       - sorto gli indici dal più piccolo al più grande
            x_tmp,y_tmp = x[i],y[i]
            r_mat = (x_tmp-new_grid_x)**2+(y_tmp-new_grid_y)**2
            min_index = np.unravel_index(np.argmin(r_mat, axis=None), r_mat.shape)
            x_index = integer_index+min_index[0]
            y_index = integer_index+min_index[1]
            x_index,y_index = np.meshgrid(x_index,y_index,indexing='ij')
            x_index,y_index = x_index.ravel(),y_index.ravel()

            flat_index = y_index + x_index*new_J
            flat_index = np.int32(flat_index)
            indexing[i,:] = flat_index
        
        self.K = indexing.shape[1]
        # compute psf weight
        x_tmp,y_tmp = x[0],y[0]
        ind_tmp = indexing[0,:]
        r = ((new_grid_x_flat[ind_tmp]-x_tmp)**2+(new_grid_y_flat[ind_tmp]-y_tmp)**2)**0.5    

        psf = np.exp(-r*r/2./(sigma_psf**2))
        
        indexing = indexing.ravel()
        print('Number of points for the convolution is {}'.format(self.K))

        return new_grid_x_flat,new_grid_y_flat,psf,self.K,indexing

    def plot_2D(self,Z,xlabel,ylabel,zlabel):
        X = self.x.reshape((self.N,self.J))
        Y = self.y.reshape((self.N,self.J))

        fig = plt.figure()
        ax = fig.add_subplot()    
        ax.set_aspect('equal', 'box')  
        pmesh = ax.pcolormesh(X,Y,Z.T,cmap = 'viridis',shading='nearest')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        cbar = plt.colorbar(pmesh)
        cbar.ax.set_ylabel(zlabel)

        return (fig,ax,pmesh,cbar)

    def plot_3D(self,Z,xlabel,ylabel,zlabel):
        fig = plt.figure()
        ax =fig.add_subplot(projection = '3d')
        ax.scatter(self.x,self.y,Z,c = 'black',s = 1.5,marker = '.')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)

        return (fix,ax)

    def brightness_map(self,brightness,xlabel='x [kpc]',ylabel='y [kpc]',zlabel= r'$log_{10}(B)$ $[L_{\odot}/kpc^2]$'):
        if brightness.shape == (self.N,self.J):
            B_matrix = np.copy(brightness)
        else:
            B_matrix = brightness.reshape((self.N,self.J))
        fig,ax,pmesh,cbar = self.plot_2D(B_matrix,xlabel,ylabel,zlabel)
        return (fig,ax,pmesh,cbar)

    def velocity_map(self,v_los,xlabel='x [kpc]',ylabel='y [kpc]',zlabel= r'$v_{los}$ $[km/s]$'):
        if v_los.shape == (self.N,self.J):
            v_matrix = np.copy(v_los)
        else:
            v_matrix = v_los.reshape((self.N,self.J))
        fig,ax,pmesh,cbar = self.plot_2D(v_matrix,xlabel,ylabel,zlabel)
        return(fig,ax,pmesh,cbar)

    def dispersion_map(self,sigma_los,xlabel='x [kpc]',ylabel='y [kpc]',zlabel= r'$\sigma_{los}$ $[km/s]$'):
        if sigma_los.shape == (self.N,self.J):
            sigma_matrix = np.copy(sigma_los)
        else:
            sigma_matrix = sigma_los.reshape((self.N,self.J))
        fig,ax,pmesh,cbar = self.plot_2D(sigma_matrix,xlabel,ylabel,zlabel)
        return(fig,ax,pmesh,cbar)
    
    def ML_map(self,ML_ratio,xlabel='x [kpc]',ylabel='y [kpc]',zlabel= r'$M/L$ $[M_{\odot}/L_{\odot}]$'):
        if ML_ratio.shape == (self.N,self.J):
            ML_matrix = np.copy(ML_ratio)
        else:
            ML_matrix = ML_ratio.reshape((self.N,self.J))
        fig,ax,pmesh,cbar = self.plot_2D(ML_matrix,xlabel,ylabel,zlabel)
        return(fig,ax,pmesh,cbar)


    def brightness_3D(self,brightness,xlabel='x [kpc]',ylabel='y [kpc]',zlabel= r'$log_{10}(B)$ $[L_{\odot}/kpc^2]$'):
        fig,ax = self.plot_3D(brightness,xlabel,ylabel,zlabel)
        return(fig,ax)

    def velocity_3D(self,v_los,xlabel='x [kpc]',ylabel='y [kpc]',zlabel= r'$v_{los}$ $[km/s]$'):
        fig,ax = self.plot_3D(v_los,xlabel,ylabel,zlabel)
        return(fig,ax)

    def dispersion_3D(self,sigma_los,xlabel='x [kpc]',ylabel='y [kpc]',zlabel= r'$\sigma_{los}$ $[km/s]$'):
        fig,ax = self.plot_3D(sigma_los,xlabel,ylabel,zlabel)
        return(fig,ax)

    def ML_ratio_3D(self,ML_ratio,xlabel='x [kpc]',ylabel='y [kpc]',zlabel=  r'$M/L$ $[M_{\odot}/L_{\odot}]$'):
        fig,ax = self.plot_3D(ML_ratio,xlabel,ylabel,zlabel)
        return(fig,ax)

    def all_maps_kpc(self,
                     brightness_data,
                     brightness_model,
                     v_data,
                     v_model,
                     sigma_data,
                     sigma_model,
                     ML_data,
                     ML_model,
                     brightness_error,
                     v_error,
                     sigma_error,
                     ML_error,
                     plot_ML=True,
                     lim='data',
                     vmin=None,
                     vmax=None):
        
        if brightness_data.shape == (self.N,self.J):
            B_matrix_data = np.copy(brightness_data)
        else:
            B_matrix_data = brightness_data.reshape((self.N,self.J))
        if brightness_model.shape == (self.N,self.J):
            B_matrix_model = np.copy(brightness_model)
        else:
            B_matrix_model = brightness_model.reshape((self.N,self.J))
        if brightness_error.shape == (self.N,self.J):
            B_matrix_err = np.copy(brightness_error)    
        else:
            B_matrix_err = brightness_error.reshape((self.N,self.J))
        if v_data.shape == (self.N,self.J):
            v_matrix_data = np.copy(v_data)
        else:
            v_matrix_data = v_data.reshape((self.N,self.J))
        if v_model.shape == (self.N,self.J):
            v_matrix_model = np.copy(v_model)
        else:
            v_matrix_model = v_model.reshape((self.N,self.J))
        if v_error.shape == (self.N,self.J):
            v_matrix_err = np.copy(v_error)    
        else:
            v_matrix_err = v_error.reshape((self.N,self.J))
        if sigma_data.shape == (self.N,self.J):
            sigma_matrix_data = np.copy(sigma_data)
        else:
            sigma_matrix_data = sigma_data.reshape((self.N,self.J))
        if sigma_model.shape == (self.N,self.J):
            sigma_matrix_model = np.copy(sigma_model)
        else:
            sigma_matrix_model = sigma_model.reshape((self.N,self.J))
        if sigma_error.shape == (self.N,self.J):
            sigma_matrix_err = np.copy(sigma_error)    
        else:
            sigma_matrix_err = sigma_error.reshape((self.N,self.J))    
        if ML_data.shape == (self.N,self.J):
            ML_matrix_data = np.copy(ML_data)
        else:
            ML_matrix_data = ML_data.reshape((self.N,self.J))
        if ML_model.shape == (self.N,self.J):
            ML_matrix_model = np.copy(ML_model)
        else:
            ML_matrix_model = ML_model.reshape((self.N,self.J))
        if ML_error.shape == (self.N,self.J):
            ML_matrix_err = np.copy(ML_error)    
        else:
            ML_matrix_err = ML_error.reshape((self.N,self.J))  

        if lim=='data':
            vmin = [np.nanmin(brightness_data),np.nanmin(v_data),np.nanmin(sigma_data),np.nanmin(ML_data)]
            vmax = [np.nanmax(brightness_data),np.nanmax(v_data),np.nanmax(sigma_data),np.nanmax(ML_data)]
        elif lim=='model':
            vmin = [np.nanmin(brightness_model),np.nanmin(v_model),np.nanmin(sigma_model),np.nanmin(ML_model)]
            vmax = [np.nanmax(brightness_model),np.nanmax(v_model),np.nanmax(sigma_model),np.nanmax(ML_model)]
        elif lim=='user':
            if vmin==None or vmax==None:
                print('error: must specify limits')
        else:
            print('lim can be data or model or user')

        X = self.x.reshape((self.N,self.J))
        Y = self.y.reshape((self.N,self.J))

        fig = plt.figure()
        fig.set_figheight(20)
        fig.set_figwidth(25)
        
        if plot_ML == True: 
            ax1  = fig.add_subplot(3,4,1) 
            ax2  = fig.add_subplot(3,4,2) 
            ax3  = fig.add_subplot(3,4,3) 
            ax4  = fig.add_subplot(3,4,4) 
            ax5  = fig.add_subplot(3,4,5) 
            ax6  = fig.add_subplot(3,4,6) 
            ax7  = fig.add_subplot(3,4,7) 
            ax8  = fig.add_subplot(3,4,8) 
            ax9  = fig.add_subplot(3,4,9)
            ax10 = fig.add_subplot(3,4,10) 
            ax11 = fig.add_subplot(3,4,11) 
            ax12 = fig.add_subplot(3,4,12) 
            ax1.set_xlabel('x [kpc]'),ax1.set_ylabel('y [kpc]')
            ax2.set_xlabel('x [kpc]'),ax2.set_ylabel('y [kpc]')
            ax3.set_xlabel('x [kpc]'),ax3.set_ylabel('y [kpc]')
            ax4.set_xlabel('x [kpc]'),ax4.set_ylabel('y [kpc]')
            ax5.set_xlabel('x [kpc]'),ax5.set_ylabel('y [kpc]')
            ax6.set_xlabel('x [kpc]'),ax6.set_ylabel('y [kpc]')
            ax7.set_xlabel('x [kpc]'),ax7.set_ylabel('y [kpc]')
            ax8.set_xlabel('x [kpc]'),ax8.set_ylabel('y [kpc]')
            ax9.set_xlabel('x [kpc]'),ax9.set_ylabel('y [kpc]')
            ax10.set_xlabel('x [kpc]'),ax10.set_ylabel('y [kpc]')
            ax11.set_xlabel('x [kpc]'),ax11.set_ylabel('y [kpc]')
            ax12.set_xlabel('x [kpc]'),ax12.set_ylabel('y [kpc]')
            ax1.set_xlim(self.xmin,self.xmax),ax1.set_ylim(self.ymin,self.ymax)
            ax2.set_xlim(self.xmin,self.xmax),ax2.set_ylim(self.ymin,self.ymax)
            ax3.set_xlim(self.xmin,self.xmax),ax3.set_ylim(self.ymin,self.ymax)
            ax4.set_xlim(self.xmin,self.xmax),ax4.set_ylim(self.ymin,self.ymax)
            ax5.set_xlim(self.xmin,self.xmax),ax5.set_ylim(self.ymin,self.ymax)
            ax6.set_xlim(self.xmin,self.xmax),ax6.set_ylim(self.ymin,self.ymax)
            ax7.set_xlim(self.xmin,self.xmax),ax7.set_ylim(self.ymin,self.ymax)
            ax8.set_xlim(self.xmin,self.xmax),ax8.set_ylim(self.ymin,self.ymax)
            ax9.set_xlim(self.xmin,self.xmax),ax9.set_ylim(self.ymin,self.ymax)
            ax10.set_xlim(self.xmin,self.xmax),ax10.set_ylim(self.ymin,self.ymax)
            ax11.set_xlim(self.xmin,self.xmax),ax11.set_ylim(self.ymin,self.ymax)
            ax12.set_xlim(self.xmin,self.xmax),ax12.set_ylim(self.ymin,self.ymax)
            ax1.set_aspect('equal', 'box')  
            ax2.set_aspect('equal', 'box')  
            ax3.set_aspect('equal', 'box')  
            ax4.set_aspect('equal', 'box')  
            ax5.set_aspect('equal', 'box')  
            ax6.set_aspect('equal', 'box')  
            ax7.set_aspect('equal', 'box')  
            ax8.set_aspect('equal', 'box')  
            ax9.set_aspect('equal', 'box')  
            ax10.set_aspect('equal', 'box')  
            ax11.set_aspect('equal', 'box')  
            ax12.set_aspect('equal', 'box')  

            # BRIGHTNESS
            pmesh1 = ax1.pcolormesh(X,Y,B_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[0],vmax=vmax[0])
            cbar1  = plt.colorbar(pmesh1,ax=ax1,label=r'$log_{10}(B)$ $[L_{\odot}/arcsec^2]$')
            #B_lim  = pmesh1.get_clim()
            pmesh5 = ax5.pcolormesh(X,Y,B_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[0],vmax=vmax[0])
            cbar5  = plt.colorbar(pmesh5,ax=ax5,label=r'$log_{10}(B)$ $[L_{\odot}/arcsec^2]$' )
            #pmesh5.set_clim(B_lim)
            #pmesh9 = ax9.pcolormesh(X,Y,(10**B_matrix_data-10**B_matrix_model)/10**B_matrix_data,cmap = 'viridis',shading='nearest')
            pmesh9 = ax9.pcolormesh(X,Y,(10**B_matrix_data-10**B_matrix_model)/B_matrix_err,cmap = 'viridis',shading='nearest')
            #cbar9  = plt.colorbar(pmesh9,ax=ax9,label=r'$(B_{data}-B_{model})$ / $B_{data}$')
            cbar9  = plt.colorbar(pmesh9,ax=ax9,label=r'$(B_{data}-B_{model})$ / $\delta B$')
            #pmesh9.set_clim((-1.,1.))
            pmesh9.set_clim((-10.,10.))

            # VELOCITY
            pmesh2 = ax2.pcolormesh(X,Y,v_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[1],vmax=vmax[1])
            cbar2  = plt.colorbar(pmesh2,ax=ax2,label=r'$v_{los}$ $[km/s]$')
            #v_lim  = pmesh2.get_clim()
            pmesh6 = ax6.pcolormesh(X,Y,v_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[1],vmax=vmax[1])
            cbar6  = plt.colorbar(pmesh6,ax=ax6,label=r'$v_{los}$ $[km/s]$')
            #pmesh6.set_clim((v_lim))
            pmesh10 = ax10.pcolormesh(X,Y,(v_matrix_data-v_matrix_model)/v_matrix_err,cmap = 'viridis',shading='nearest')
            cbar10  = plt.colorbar(pmesh10,ax=ax10,label=r'$(v_{data}-v_{model})$ / $\delta v$')
            pmesh10.set_clim((-10.,10.))

            # SIGMA
            pmesh3 = ax3.pcolormesh(X,Y,sigma_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[2],vmax=vmax[2])
            cbar3  = plt.colorbar(pmesh3,ax=ax3,label=r'$\sigma_{los}$ $[km/s]$')
            #sigma_lim  = pmesh3.get_clim()
            pmesh7 = ax7.pcolormesh(X,Y,sigma_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[2],vmax=vmax[2])
            cbar7  = plt.colorbar(pmesh7,ax=ax7,label=r'$\sigma_{los}$ $[km/s]$')
            #pmesh7.set_clim((sigma_lim))
            pmesh11 = ax11.pcolormesh(X,Y,(sigma_matrix_data-sigma_matrix_model)/sigma_matrix_err,cmap = 'viridis',shading='nearest')
            cbar11  = plt.colorbar(pmesh11,ax=ax11,label=r'$(\sigma_{data}-\sigma_{model})$ / $\delta \sigma$')
            pmesh11.set_clim((-10.,10.))

            # ML ratio
            pmesh4 = ax4.pcolormesh(X,Y,ML_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[3],vmax=vmax[3])
            cbar4  = plt.colorbar(pmesh4,ax=ax4,label=r'$ML$ $[M_{\odot}/L_{\odot}]$')
            #ML_lim  = pmesh4.get_clim()
            pmesh8 = ax8.pcolormesh(X,Y,ML_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[3],vmax=vmax[3])
            cbar8  = plt.colorbar(pmesh8,ax=ax8,label=r'$ML$ $[M_{\odot}/L_{\odot}]$')
            #pmesh8.set_clim((ML_lim))
            pmesh12 = ax12.pcolormesh(X,Y,(ML_matrix_data-ML_matrix_model)/ML_matrix_err,cmap = 'viridis',shading='nearest')
            cbar12  = plt.colorbar(pmesh12,ax=ax12,label=r'$(ML_{data}-ML_{model})$ / $\delta ML$')
            pmesh12.set_clim((-10.,10.))

            plt.subplots_adjust(top=0.99,
                                bottom=0.07,
                                left=0.005,
                                right=0.98,
                                hspace=0.305,
                                wspace=0.18)

        else: 
            ax1  = fig.add_subplot(3,3,1)
            ax2  = fig.add_subplot(3,3,2)
            ax3  = fig.add_subplot(3,3,3)
            ax4  = fig.add_subplot(3,3,4)
            ax5  = fig.add_subplot(3,3,5)
            ax6  = fig.add_subplot(3,3,6)
            ax7  = fig.add_subplot(3,3,7)
            ax8  = fig.add_subplot(3,3,8)
            ax9  = fig.add_subplot(3,3,9)
        
            ax1.set_xlabel('x [kpc]'),ax1.set_ylabel('y [kpc]')
            ax2.set_xlabel('x [kpc]'),ax2.set_ylabel('y [kpc]')
            ax3.set_xlabel('x [kpc]'),ax3.set_ylabel('y [kpc]')
            ax4.set_xlabel('x [kpc]'),ax4.set_ylabel('y [kpc]')
            ax5.set_xlabel('x [kpc]'),ax5.set_ylabel('y [kpc]')
            ax6.set_xlabel('x [kpc]'),ax6.set_ylabel('y [kpc]')
            ax7.set_xlabel('x [kpc]'),ax7.set_ylabel('y [kpc]')
            ax8.set_xlabel('x [kpc]'),ax8.set_ylabel('y [kpc]')
            ax9.set_xlabel('x [kpc]'),ax9.set_ylabel('y [kpc]')
            ax1.set_xlim(self.xmin,self.xmax),ax1.set_ylim(self.ymin,self.ymax)
            ax2.set_xlim(self.xmin,self.xmax),ax2.set_ylim(self.ymin,self.ymax)
            ax3.set_xlim(self.xmin,self.xmax),ax3.set_ylim(self.ymin,self.ymax)
            ax4.set_xlim(self.xmin,self.xmax),ax4.set_ylim(self.ymin,self.ymax)
            ax5.set_xlim(self.xmin,self.xmax),ax5.set_ylim(self.ymin,self.ymax)
            ax6.set_xlim(self.xmin,self.xmax),ax6.set_ylim(self.ymin,self.ymax)
            ax7.set_xlim(self.xmin,self.xmax),ax7.set_ylim(self.ymin,self.ymax)
            ax8.set_xlim(self.xmin,self.xmax),ax8.set_ylim(self.ymin,self.ymax)
            ax9.set_xlim(self.xmin,self.xmax),ax9.set_ylim(self.ymin,self.ymax)
            ax1.set_aspect('equal', 'box')  
            ax2.set_aspect('equal', 'box')  
            ax3.set_aspect('equal', 'box')  
            ax4.set_aspect('equal', 'box')  
            ax5.set_aspect('equal', 'box')  
            ax6.set_aspect('equal', 'box')  
            ax7.set_aspect('equal', 'box')  
            ax8.set_aspect('equal', 'box')  
            ax9.set_aspect('equal', 'box') 
            # BRIGHTNESS
            pmesh1 = ax1.pcolormesh(X,Y,B_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[0],vmax=vmax[0])
            cbar1  = plt.colorbar(pmesh1,ax=ax1,label=r'$log_{10}(B)$ $[L_{\odot}/arcsec^2]$')
            #B_lim  = pmesh1.get_clim()
            pmesh4 = ax4.pcolormesh(X,Y,B_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[0],vmax=vmax[0])
            cbar4  = plt.colorbar(pmesh4,ax=ax4,label=r'$log_{10}(B)$ $[L_{\odot}/arcsec^2]$')
            #pmesh4.set_clim(B_lim)
            pmesh7 = ax7.pcolormesh(X,Y,(10**B_matrix_data-10**B_matrix_model)/10**B_matrix_data,cmap = 'viridis',shading='nearest')
            cbar7  = plt.colorbar(pmesh7,ax=ax7,label=r'$(B_{data}-B_{model})$ / $B_{data}$')
            pmesh7.set_clim((-1.,1.))

            # VELOCITY
            pmesh2 = ax2.pcolormesh(X,Y,v_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[1],vmax=vmax[1])
            cbar2  = plt.colorbar(pmesh2,ax=ax2,label=r'$v_{los}$ $[km/s]$')
            #v_lim  = pmesh2.get_clim()
            pmesh5 = ax5.pcolormesh(X,Y,v_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[1],vmax=vmax[1])
            cbar5  = plt.colorbar(pmesh5,ax=ax5,label=r'$v_{los}$ $[km/s]$')
            #pmesh5.set_clim((v_lim))
            pmesh8 = ax8.pcolormesh(X,Y,(v_matrix_data-v_matrix_model)/v_matrix_err,cmap = 'viridis',shading='nearest')
            cbar8  = plt.colorbar(pmesh8,ax=ax8,label=r'$(v_{data}-v_{model})$ / $\delta v$')
            pmesh8.set_clim((-10.,10.))

            # SIGMA
            pmesh3 = ax3.pcolormesh(X,Y,sigma_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[2],vmax=vmax[2])
            cbar3  = plt.colorbar(pmesh3,ax=ax3,label=r'$\sigma_{los}$ $[km/s]$')
            #sigma_lim  = pmesh3.get_clim()
            pmesh6 = ax6.pcolormesh(X,Y,sigma_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[2],vmax=vmax[2])
            cbar6  = plt.colorbar(pmesh6,ax=ax6,label=r'$\sigma_{los}$ $[km/s]$')
            #pmesh6.set_clim((sigma_lim))
            pmesh9 = ax9.pcolormesh(X,Y,(sigma_matrix_data-sigma_matrix_model)/sigma_matrix_err,cmap = 'viridis',shading='nearest')
            cbar9  = plt.colorbar(pmesh9,ax=ax9,label=r'$(\sigma_{data}-\sigma_{model})$ / $\delta \sigma$')
            pmesh9.set_clim((-10.,10.))
        
            plt.subplots_adjust(top=0.98,
                                bottom=0.075,
                                left=0.1,
                                right=0.905,
                                hspace=0.29,
                                wspace=0.2)


        return fig


    def all_maps_arcsec(self,
                     brightness_data,
                     brightness_model,
                     v_data,
                     v_model,
                     sigma_data,
                     sigma_model,
                     ML_data,
                     ML_model,
                     v_error,
                     sigma_error,
                     ML_error,
                     plot_ML=True,
                     lim='data',
                     vmin=None,
                     vmax=None):
        
        if brightness_data.shape == (self.N,self.J):
            B_matrix_data = np.copy(brightness_data)
        else:
            B_matrix_data = brightness_data.reshape((self.N,self.J))
        if brightness_model.shape == (self.N,self.J):
            B_matrix_model = np.copy(brightness_model)
        else:
            B_matrix_model = brightness_model.reshape((self.N,self.J))
        if v_data.shape == (self.N,self.J):
            v_matrix_data = np.copy(v_data)
        else:
            v_matrix_data = v_data.reshape((self.N,self.J))
        if v_model.shape == (self.N,self.J):
            v_matrix_model = np.copy(v_model)
        else:
            v_matrix_model = v_model.reshape((self.N,self.J))
        if v_error.shape == (self.N,self.J):
            v_matrix_err = np.copy(v_error)    
        else:
            v_matrix_err = v_error.reshape((self.N,self.J))
        if sigma_data.shape == (self.N,self.J):
            sigma_matrix_data = np.copy(sigma_data)
        else:
            sigma_matrix_data = sigma_data.reshape((self.N,self.J))
        if sigma_model.shape == (self.N,self.J):
            sigma_matrix_model = np.copy(sigma_model)
        else:
            sigma_matrix_model = sigma_model.reshape((self.N,self.J))
        if sigma_error.shape == (self.N,self.J):
            sigma_matrix_err = np.copy(sigma_error)    
        else:
            sigma_matrix_err = sigma_error.reshape((self.N,self.J))    
        if ML_data.shape == (self.N,self.J):
            ML_matrix_data = np.copy(ML_data)
        else:
            ML_matrix_data = ML_data.reshape((self.N,self.J))
        if ML_model.shape == (self.N,self.J):
            ML_matrix_model = np.copy(ML_model)
        else:
            ML_matrix_model = ML_model.reshape((self.N,self.J))
        if ML_error.shape == (self.N,self.J):
            ML_matrix_err = np.copy(ML_error)    
        else:
            ML_matrix_err = ML_error.reshape((self.N,self.J))  

        X = self.x.reshape((self.N,self.J))
        Y = self.y.reshape((self.N,self.J))

        fig = plt.figure()
        fig.set_figheight(20)
        fig.set_figwidth(25)

        if lim=='data':
            vmin = [np.min(brightness_data),np.min(v_data),np.min(sigma_data),np.min(ML_data)]
            vmax = [np.max(brightness_data),np.max(v_data),np.max(sigma_data),np.min(ML_data)]
        elif lim=='model':
            vmin = [np.min(brightness_model),np.min(v_model),np.min(sigma_model),np.min(ML_model)]
            vmax = [np.max(brightness_model),np.max(v_model),np.max(sigma_model),np.min(ML_model)]
        elif lim=='user':
            if vmin==None or vmax==None:
                print('error: must specify limits')
        else:
            print('lim can be data or model or user')
        if plot_ML == True: 
            ax1  = fig.add_subplot(3,4,1) 
            ax2  = fig.add_subplot(3,4,2) 
            ax3  = fig.add_subplot(3,4,3) 
            ax4  = fig.add_subplot(3,4,4) 
            ax5  = fig.add_subplot(3,4,5) 
            ax6  = fig.add_subplot(3,4,6) 
            ax7  = fig.add_subplot(3,4,7) 
            ax8  = fig.add_subplot(3,4,8) 
            ax9  = fig.add_subplot(3,4,9)
            ax10 = fig.add_subplot(3,4,10) 
            ax11 = fig.add_subplot(3,4,11) 
            ax12 = fig.add_subplot(3,4,12) 
            ax1.set_xlabel('x [arcsec]'),ax1.set_ylabel('y [arcsec]')
            ax2.set_xlabel('x [arcsec]'),ax2.set_ylabel('y [arcsec]')
            ax3.set_xlabel('x [arcsec]'),ax3.set_ylabel('y [arcsec]')
            ax4.set_xlabel('x [arcsec]'),ax4.set_ylabel('y [arcsec]')
            ax5.set_xlabel('x [arcsec]'),ax5.set_ylabel('y [arcsec]')
            ax6.set_xlabel('x [arcsec]'),ax6.set_ylabel('y [arcsec]')
            ax7.set_xlabel('x [arcsec]'),ax7.set_ylabel('y [arcsec]')
            ax8.set_xlabel('x [arcsec]'),ax8.set_ylabel('y [arcsec]')
            ax9.set_xlabel('x [arcsec]'),ax9.set_ylabel('y [arcsec]')
            ax10.set_xlabel('x [arcsec]'),ax10.set_ylabel('y [arcsec]')
            ax11.set_xlabel('x [arcsec]'),ax11.set_ylabel('y [arcsec]')
            ax12.set_xlabel('x [arcsec]'),ax12.set_ylabel('y [arcsec]')
            ax1.set_xlim(self.xmin,self.xmax),ax1.set_ylim(self.ymin,self.ymax)
            ax2.set_xlim(self.xmin,self.xmax),ax2.set_ylim(self.ymin,self.ymax)
            ax3.set_xlim(self.xmin,self.xmax),ax3.set_ylim(self.ymin,self.ymax)
            ax4.set_xlim(self.xmin,self.xmax),ax4.set_ylim(self.ymin,self.ymax)
            ax5.set_xlim(self.xmin,self.xmax),ax5.set_ylim(self.ymin,self.ymax)
            ax6.set_xlim(self.xmin,self.xmax),ax6.set_ylim(self.ymin,self.ymax)
            ax7.set_xlim(self.xmin,self.xmax),ax7.set_ylim(self.ymin,self.ymax)
            ax8.set_xlim(self.xmin,self.xmax),ax8.set_ylim(self.ymin,self.ymax)
            ax9.set_xlim(self.xmin,self.xmax),ax9.set_ylim(self.ymin,self.ymax)
            ax10.set_xlim(self.xmin,self.xmax),ax10.set_ylim(self.ymin,self.ymax)
            ax11.set_xlim(self.xmin,self.xmax),ax11.set_ylim(self.ymin,self.ymax)
            ax12.set_xlim(self.xmin,self.xmax),ax12.set_ylim(self.ymin,self.ymax)
            ax1.set_aspect('equal', 'box')  
            ax2.set_aspect('equal', 'box')  
            ax3.set_aspect('equal', 'box')  
            ax4.set_aspect('equal', 'box')  
            ax5.set_aspect('equal', 'box')  
            ax6.set_aspect('equal', 'box')  
            ax7.set_aspect('equal', 'box')  
            ax8.set_aspect('equal', 'box')  
            ax9.set_aspect('equal', 'box')  
            ax10.set_aspect('equal', 'box')  
            ax11.set_aspect('equal', 'box')  
            ax12.set_aspect('equal', 'box')  

            # BRIGHTNESS
            pmesh1 = ax1.pcolormesh(X,Y,B_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[0],vmax=vmax[0])
            cbar1  = plt.colorbar(pmesh1,ax=ax1,label=r'$log_{10}(B)$ $[L_{\odot}/arcsec^2]$')
            #B_lim  = pmesh1.get_clim()
            pmesh5 = ax5.pcolormesh(X,Y,B_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[0],vmax=vmax[0])
            cbar5  = plt.colorbar(pmesh5,ax=ax5,label=r'$log_{10}(B)$ $[L_{\odot}/arcsec^2]$' )
            #pmesh5.set_clim(B_lim)
            pmesh9 = ax9.pcolormesh(X,Y,(10**B_matrix_data-10**B_matrix_model)/10**B_matrix_data,cmap = 'viridis',shading='nearest')
            cbar9  = plt.colorbar(pmesh9,ax=ax9,label=r'$(B_{data}-B_{model})$ / $B_{data}$')
            pmesh9.set_clim((-1.,1.))

            # VELOCITY
            pmesh2 = ax2.pcolormesh(X,Y,v_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[1],vmax=vmax[1])
            cbar2  = plt.colorbar(pmesh2,ax=ax2,label=r'$v_{los}$ $[km/s]$')
            #v_lim  = pmesh2.get_clim()
            pmesh6 = ax6.pcolormesh(X,Y,v_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[1],vmax=vmax[1])
            cbar6  = plt.colorbar(pmesh6,ax=ax6,label=r'$v_{los}$ $[km/s]$')
            #pmesh6.set_clim((v_lim))
            pmesh10 = ax10.pcolormesh(X,Y,(v_matrix_data-v_matrix_model)/v_matrix_err,cmap = 'viridis',shading='nearest')
            cbar10  = plt.colorbar(pmesh10,ax=ax10,label=r'$(v_{data}-v_{model})$ / $\delta v$')
            pmesh10.set_clim((-10.,10.))

            # SIGMA
            pmesh3 = ax3.pcolormesh(X,Y,sigma_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[2],vmax=vmax[2])
            cbar3  = plt.colorbar(pmesh3,ax=ax3,label=r'$\sigma_{los}$ $[km/s]$')
            #sigma_lim  = pmesh3.get_clim()
            pmesh7 = ax7.pcolormesh(X,Y,sigma_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[2],vmax=vmax[2])
            cbar7  = plt.colorbar(pmesh7,ax=ax7,label=r'$\sigma_{los}$ $[km/s]$')
            #pmesh7.set_clim((sigma_lim))
            pmesh11 = ax11.pcolormesh(X,Y,(sigma_matrix_data-sigma_matrix_model)/sigma_matrix_err,cmap = 'viridis',shading='nearest')
            cbar11  = plt.colorbar(pmesh11,ax=ax11,label=r'$(\sigma_{data}-\sigma_{model})$ / $\delta \sigma$')
            pmesh11.set_clim((-10.,10.))

            # ML ratio
            pmesh4 = ax4.pcolormesh(X,Y,ML_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[3],vmax=vmax[3])
            cbar4  = plt.colorbar(pmesh4,ax=ax4,label=r'$ML$ $[M_{\odot}/L_{\odot}]$')
            #ML_lim  = pmesh4.get_clim()
            pmesh8 = ax8.pcolormesh(X,Y,ML_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[3],vmax=vmax[3])
            cbar8  = plt.colorbar(pmesh8,ax=ax8,label=r'$ML$ $[M_{\odot}/L_{\odot}]$')
            #pmesh8.set_clim((ML_lim))
            pmesh12 = ax12.pcolormesh(X,Y,(ML_matrix_data-ML_matrix_model)/ML_matrix_err,cmap = 'viridis',shading='nearest')
            cbar12  = plt.colorbar(pmesh12,ax=ax12,label=r'$(ML_{data}-ML_{model})$ / $\delta ML$')
            pmesh12.set_clim((-10.,10.))

            plt.subplots_adjust(top=0.99,
                                bottom=0.07,
                                left=0.005,
                                right=0.98,
                                hspace=0.305,
                                wspace=0.18)

        else: 
            ax1  = fig.add_subplot(3,3,1)
            ax2  = fig.add_subplot(3,3,2)
            ax3  = fig.add_subplot(3,3,3)
            ax4  = fig.add_subplot(3,3,4)
            ax5  = fig.add_subplot(3,3,5)
            ax6  = fig.add_subplot(3,3,6)
            ax7  = fig.add_subplot(3,3,7)
            ax8  = fig.add_subplot(3,3,8)
            ax9  = fig.add_subplot(3,3,9)
        
            ax1.set_xlabel('x [arcsec]'),ax1.set_ylabel('y [arcsec]')
            ax2.set_xlabel('x [arcsec]'),ax2.set_ylabel('y [arcsec]')
            ax3.set_xlabel('x [arcsec]'),ax3.set_ylabel('y [arcsec]')
            ax4.set_xlabel('x [arcsec]'),ax4.set_ylabel('y [arcsec]')
            ax5.set_xlabel('x [arcsec]'),ax5.set_ylabel('y [arcsec]')
            ax6.set_xlabel('x [arcsec]'),ax6.set_ylabel('y [arcsec]')
            ax7.set_xlabel('x [arcsec]'),ax7.set_ylabel('y [arcsec]')
            ax8.set_xlabel('x [arcsec]'),ax8.set_ylabel('y [arcsec]')
            ax9.set_xlabel('x [arcsec]'),ax9.set_ylabel('y [arcsec]')
            ax1.set_xlim(self.xmin,self.xmax),ax1.set_ylim(self.ymin,self.ymax)
            ax2.set_xlim(self.xmin,self.xmax),ax2.set_ylim(self.ymin,self.ymax)
            ax3.set_xlim(self.xmin,self.xmax),ax3.set_ylim(self.ymin,self.ymax)
            ax4.set_xlim(self.xmin,self.xmax),ax4.set_ylim(self.ymin,self.ymax)
            ax5.set_xlim(self.xmin,self.xmax),ax5.set_ylim(self.ymin,self.ymax)
            ax6.set_xlim(self.xmin,self.xmax),ax6.set_ylim(self.ymin,self.ymax)
            ax7.set_xlim(self.xmin,self.xmax),ax7.set_ylim(self.ymin,self.ymax)
            ax8.set_xlim(self.xmin,self.xmax),ax8.set_ylim(self.ymin,self.ymax)
            ax9.set_xlim(self.xmin,self.xmax),ax9.set_ylim(self.ymin,self.ymax)
            ax1.set_aspect('equal', 'box')  
            ax2.set_aspect('equal', 'box')  
            ax3.set_aspect('equal', 'box')  
            ax4.set_aspect('equal', 'box')  
            ax5.set_aspect('equal', 'box')  
            ax6.set_aspect('equal', 'box')  
            ax7.set_aspect('equal', 'box')  
            ax8.set_aspect('equal', 'box')  
            ax9.set_aspect('equal', 'box') 
            # BRIGHTNESS
            pmesh1 = ax1.pcolormesh(X,Y,B_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[0],vmax=vmax[0])
            cbar1  = plt.colorbar(pmesh1,ax=ax1,label=r'$log_{10}(B)$ $[L_{\odot}/arcsec^2]$')
            #B_lim  = pmesh1.get_clim()
            pmesh4 = ax4.pcolormesh(X,Y,B_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[0],vmax=vmax[0])
            cbar4  = plt.colorbar(pmesh4,ax=ax4,label=r'$log_{10}(B)$ $[L_{\odot}/arcsec^2]$')
            #pmesh4.set_clim(B_lim)
            pmesh7 = ax7.pcolormesh(X,Y,(10**B_matrix_data-10**B_matrix_model)/10**B_matrix_data,cmap = 'viridis',shading='nearest')
            cbar7  = plt.colorbar(pmesh7,ax=ax7,label=r'$(B_{data}-B_{model})$ / $B_{data}$')
            pmesh7.set_clim((-1.,1.))

            # VELOCITY
            pmesh2 = ax2.pcolormesh(X,Y,v_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[1],vmax=vmax[1])
            cbar2  = plt.colorbar(pmesh2,ax=ax2,label=r'$v_{los}$ $[km/s]$')
            #v_lim  = pmesh2.get_clim()
            pmesh5 = ax5.pcolormesh(X,Y,v_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[1],vmax=vmax[1])
            cbar5  = plt.colorbar(pmesh5,ax=ax5,label=r'$v_{los}$ $[km/s]$')
            #pmesh5.set_clim((v_lim))
            pmesh8 = ax8.pcolormesh(X,Y,(v_matrix_data-v_matrix_model)/v_matrix_err,cmap = 'viridis',shading='nearest')
            cbar8  = plt.colorbar(pmesh8,ax=ax8,label=r'$(v_{data}-v_{model})$ / $\delta v$')
            pmesh8.set_clim((-10.,10.))

            # SIGMA
            pmesh3 = ax3.pcolormesh(X,Y,sigma_matrix_data,cmap = 'viridis',shading='nearest',vmin=vmin[2],vmax=vmax[2])
            cbar3  = plt.colorbar(pmesh3,ax=ax3,label=r'$\sigma_{los}$ $[km/s]$')
            #sigma_lim  = pmesh3.get_clim()
            pmesh6 = ax6.pcolormesh(X,Y,sigma_matrix_model,cmap = 'viridis',shading='nearest',vmin=vmin[2],vmax=vmax[2])
            cbar6  = plt.colorbar(pmesh6,ax=ax6,label=r'$\sigma_{los}$ $[km/s]$')
            #pmesh6.set_clim((sigma_lim))
            pmesh9 = ax9.pcolormesh(X,Y,(sigma_matrix_data-sigma_matrix_model)/sigma_matrix_err,cmap = 'viridis',shading='nearest')
            cbar9  = plt.colorbar(pmesh9,ax=ax9,label=r'$(\sigma_{data}-\sigma_{model})$ / $\delta \sigma$')
            pmesh9.set_clim((-10.,10.))
        
            plt.subplots_adjust(top=0.98,
                                bottom=0.075,
                                left=0.1,
                                right=0.905,
                                hspace=0.29,
                                wspace=0.2)

        return fig


    def all_1D_profile(self,
                     arcsec_to_kpc,
                     r_avg_brightness,
                     brightness_avg_data,
                     brightness_avg_model,
                     r_avg_v,
                     v_avg_data,
                     v_avg_model,
                     r_avg_sigma,
                     sigma_avg_data,
                     sigma_avg_model,
                     r_avg_ML,
                     ML_avg_data,
                     ML_avg_model,
                     brightness_avg_error,
                     v_avg_error,
                     sigma_avg_error,
                     ML_avg_error,
                     plot_ML=True):
        fig = plt.figure()
        fig.set_figheight(20)
        fig.set_figwidth(25)
        if plot_ML==True:
            ax1  = fig.add_subplot(1,4,1)
            ax2  = fig.add_subplot(1,4,2)
            ax3  = fig.add_subplot(1,4,3)
            ax4  = fig.add_subplot(1,4,4)
            #ax1.set_aspect('equal', 'box')
            #ax2.set_aspect('equal', 'box')
            #ax3.set_aspect('equal', 'box')
            #ax4.set_aspect('equal', 'box')
            
            ax1.plot((r_avg_brightness),(brightness_avg_data),'ok')
            ax1.fill_between((r_avg_brightness), ((brightness_avg_data-brightness_avg_error)), ((brightness_avg_data+brightness_avg_error)),color='black',alpha=0.5)
            ax1.plot((r_avg_brightness),(brightness_avg_model),'b',lw=3)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlabel('r [arcsec]')#,size=10.5)
            ax1.set_ylabel(r'$B$ $[L_{\odot}/kpc^2]$')#,size=10.5)
            
            ax2.plot((r_avg_v),(v_avg_data),'ok',label='data')
            ax2.fill_between((r_avg_v), ((v_avg_data-v_avg_error)), ((v_avg_data+v_avg_error)),color='black',alpha=0.5)
            ax2.plot((r_avg_v),(v_avg_model),'b',label='Fabio',lw=3)
            ax2.set_xlabel('r [arcsec]')#,size=10.5)
            ax2.set_ylabel(r'$v$ [km/s]')#,size=10.5)
            
            ax3.plot((r_avg_sigma),(sigma_avg_data),'ok',label='data')
            ax3.fill_between((r_avg_sigma), ((sigma_avg_data-sigma_avg_error)), ((sigma_avg_data+sigma_avg_error)),color='black',alpha=0.5)
            ax3.plot((r_avg_sigma),(sigma_avg_model),'b',label='Fabio',lw=3)
            ax3.set_xlabel('r [arcsec]')#,size=10.5)
            ax3.set_ylabel(r'$\sigma$ [km/s]')#,size=10.5)
            
            ax4.plot((r_avg_ML),(ML_avg_data),'ok',label='data')
            ax4.fill_between((r_avg_ML), ((ML_avg_data-ML_avg_error)), ((ML_avg_data+ML_avg_error)),color='black',alpha=0.5)
            ax4.plot((r_avg_ML),(ML_avg_model),'b',label='Fabio',lw=3)
            ax4.set_xlabel('r [arcsec]')#,size=10.5)
            ax4.set_ylabel(r'$M/L$ $[M_{\odot}/L_{\odot}]$')#,size=10.5)
            
            plt.subplots_adjust(top=0.74,
                                bottom=0.305,
                                left=0.06,
                                right=0.99,
                                hspace=0.2,
                                wspace=0.33)
        
        else:
            ax1  = fig.add_subplot(1,3,1)
            ax2  = fig.add_subplot(1,3,2)
            ax3  = fig.add_subplot(1,3,3)
            #ax1.set_aspect('equal', 'box')
            #ax2.set_aspect('equal', 'box')
            #ax3.set_aspect('equal', 'box')
            
            ax1.plot((r_avg_brightness),(brightness_avg_data),'ok')
            ax1.fill_between((r_avg_brightness), ((brightness_avg_data-brightness_avg_error)), ((brightness_avg_data+brightness_avg_error)),color='black',alpha=0.5)
            ax1.plot((r_avg_brightness),(brightness_avg_model),'b',lw=3)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlabel('r [arcsec]')#,size=10.5)
            ax1.set_ylabel(r'$B$ $[L_{\odot}/kpc^2]$')#,size=10.5)
            ax2.plot((r_avg_v),(v_avg_data),'ok',label='data')
            ax2.fill_between((r_avg_v), ((v_avg_data-v_avg_error)), ((v_avg_data+v_avg_error)),color='black',alpha=0.5)
            ax2.plot((r_avg_v),(v_avg_model),'b',label='Fabio',lw=3)
            ax2.set_xlabel('r [arcsec]')#,size=10.5)
            ax2.set_ylabel(r'$v$ [km/s]')#,size=10.5)
            ax3.plot((r_avg_sigma),(sigma_avg_data),'ok',label='data')
            ax3.fill_between((r_avg_sigma), ((sigma_avg_data-sigma_avg_error)), ((sigma_avg_data+sigma_avg_error)),color='black',alpha=0.5)
            ax3.plot((r_avg_sigma),(sigma_avg_model),'b',label='Fabio',lw=3)
            ax3.set_xlabel('r [arcsec]')#,size=10.5)
            ax3.set_ylabel(r'$\sigma$ [km/s]')#,size=10.5)

            plt.subplots_adjust(top=0.79,
                                bottom=0.2,
                                left=0.06,
                                right=0.985,
                                hspace=0.245,
                                wspace=0.3)

        return fig


    def all_1D_profile_all_outputs(self,
                     arcsec_to_kpc,
                     r_avg_brightness,
                     brightness_avg_data,
                     brightness_avg_model,
                     B_B_avg,
                     B_D1_avg,
                     B_D2_avg,
                     r_avg_v,
                     v_avg_data,
                     v_avg_model,
                     v_B_avg,
                     v_D1_avg,
                     v_D2_avg,
                     r_avg_sigma,
                     sigma_avg_data,
                     sigma_avg_model,
                     sigma_B_avg,
                     sigma_D1_avg,
                     sigma_D2_avg,
                     r_avg_ML,
                     ML_avg_data,
                     ML_avg_model,
                     ML_B_avg,
                     ML_D1_avg,
                     ML_D2_avg,
                     brightness_avg_error,
                     v_avg_error,
                     sigma_avg_error,
                     ML_avg_error,
                     plot_ML=True):
        fig = plt.figure()
        fig.set_figheight(20)
        fig.set_figwidth(25)
        if plot_ML==True:
            ax1  = fig.add_subplot(1,4,1)
            ax2  = fig.add_subplot(1,4,2)
            ax3  = fig.add_subplot(1,4,3)
            ax4  = fig.add_subplot(1,4,4)
            #ax1.set_aspect('equal', 'box')
            #ax2.set_aspect('equal', 'box')
            #ax3.set_aspect('equal', 'box')
            #ax4.set_aspect('equal', 'box')
            
            ax1.plot(np.log10(r_avg_brightness),(brightness_avg_data),'ok')
            ax1.fill_between(np.log10(r_avg_brightness), ((brightness_avg_data-brightness_avg_error)), ((brightness_avg_data+brightness_avg_error)),color='black',alpha=0.5)
            ax1.plot(np.log10(r_avg_brightness),(brightness_avg_model),'b',lw=3)
            ax1.plot(np.log10(r_avg_brightness),(B_B_avg),'m' ,lw=3)
            ax1.plot(np.log10(r_avg_brightness),(B_D1_avg),'r',lw=3)
            ax1.plot(np.log10(r_avg_brightness),(B_D2_avg),'g',lw=3)
            #ax1.set_xscale('log')
            #ax1.set_yscale('log')
            ax1.set_xlabel(r'$log_{10}{r}$ [arcsec]')#,size=10.5)
            ax1.set_ylabel(r'$log_{10}{B}$ $[L_{\odot}/kpc^2]$')#,size=10.5)
            ax1_sec = ax1.secondary_xaxis('top')
            ax1_sec.set_xticks(np.log10(10**r_avg_brightness*arcsec_to_kpc))
            ax1_sec.set_xlabel(r'$log_{10}{r}$ [kpc]')

            ax2.plot((r_avg_v),(v_avg_data),'ok',label='data')
            ax2.fill_between((r_avg_v), ((v_avg_data-v_avg_error)), ((v_avg_data+v_avg_error)),color='black',alpha=0.5)
            ax2.plot((r_avg_v),(v_avg_model),'b',lw=3)
            ax2.plot((r_avg_v),(v_B_avg),'m',lw=3)
            ax2.plot((r_avg_v),(v_D1_avg),'r',lw=3)
            ax2.plot((r_avg_v),(v_D2_avg),'g',lw=3)
            ax2.set_xlabel('r [arcsec]')#,size=10.5)
            ax2.set_ylabel(r'$v$ [km/s]')#,size=10.5)
            ax2_sec = ax3.secondary_xaxis('top')
            ax2_sec.set_xticks(r_avg_v*arcsec_to_kpc)
            ax2_sec.set_xlabel('r [kpc]')

            ax3.plot((r_avg_sigma),(sigma_avg_data),'ok',label='data')
            ax3.fill_between((r_avg_sigma), ((sigma_avg_data-sigma_avg_error)), ((sigma_avg_data+sigma_avg_error)),color='black',alpha=0.5)
            ax3.plot((r_avg_sigma),(sigma_avg_model),'b',lw=3)
            ax3.plot((r_avg_sigma),(sigma_B_avg),'m',lw=3)
            ax3.plot((r_avg_sigma),(sigma_D1_avg),'r',lw=3)
            ax3.plot((r_avg_sigma),(sigma_D2_avg),'g',lw=3)
            ax3.set_xlabel('r [arcsec]')#,size=10.5)
            ax3.set_ylabel(r'$\sigma$ [km/s]')#,size=10.5)
            ax3_sec = ax3.secondary_xaxis('top')
            ax3_sec.set_xticks(r_avg_sigma*arcsec_to_kpc)
            ax3_sec.set_xlabel('r [kpc]')

            ax4.plot((r_avg_ML),(ML_avg_data),'ok',label='data')
            ax4.fill_between((r_avg_ML), ((ML_avg_data-ML_avg_error)), ((ML_avg_data+ML_avg_error)),color='black',alpha=0.5)
            ax4.plot((r_avg_ML),(ML_avg_model),'b',lw=3)
            ax4.plot((r_avg_ML),(ML_B_avg),'m',    lw=3,label='B')
            ax4.plot((r_avg_ML),(ML_D1_avg),'r',   lw=3,label='D1')
            ax4.plot((r_avg_ML),(ML_D2_avg),'g',   lw=3,label='D2')
            ax4.set_xlabel('r [arcsec]')#,size=10.5)
            ax4.set_ylabel(r'$M/L$ $[M_{\odot}/L_{\odot}]$')#,size=10.5)
            ax4_sec = ax4.secondary_xaxis('top')
            ax4_sec.set_xticks(arcsec_to_kpc*r_avg_ML)
            ax4_sec.set_xlabel('r [kpc]')

            ax4.legend(loc='best')
            plt.subplots_adjust(top=0.74,
                                bottom=0.305,
                                left=0.06,
                                right=0.99,
                                hspace=0.2,
                                wspace=0.33)
        
        else:
            ax1  = fig.add_subplot(1,3,1)
            ax2  = fig.add_subplot(1,3,2)
            ax3  = fig.add_subplot(1,3,3)
            #ax1.set_aspect('equal', 'box')
            #ax2.set_aspect('equal', 'box')
            #ax3.set_aspect('equal', 'box')
            
            ax1.plot((r_avg_brightness),(brightness_avg_data),'ok')
            ax1.fill_between((r_avg_brightness), ((brightness_avg_data-brightness_avg_error)), ((brightness_avg_data+brightness_avg_error)),color='black',alpha=0.5)
            ax1.plot((r_avg_brightness),(brightness_avg_model),'b',lw=3)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlabel('r [arcsec]')#,size=10.5)
            ax1.set_ylabel(r'$B$ $[L_{\odot}/kpc^2]$')#,size=10.5)
            ax2.plot((r_avg_v),(v_avg_data),'ok',label='data')
            ax2.fill_between((r_avg_v), ((v_avg_data-v_avg_error)), ((v_avg_data+v_avg_error)),color='black',alpha=0.5)
            ax2.plot((r_avg_v),(v_avg_model),'b',label='Fabio',lw=3)
            ax2.set_xlabel('r [arcsec]')#,size=10.5)
            ax2.set_ylabel(r'$v$ [km/s]')#,size=10.5)
            ax3.plot((r_avg_sigma),(sigma_avg_data),'ok',label='data')
            ax3.fill_between((r_avg_sigma), ((sigma_avg_data-sigma_avg_error)), ((sigma_avg_data+sigma_avg_error)),color='black',alpha=0.5)
            ax3.plot((r_avg_sigma),(sigma_avg_model),'b',label='Fabio',lw=3)
            ax3.set_xlabel('r [arcsec]')#,size=10.5)
            ax3.set_ylabel(r'$\sigma$ [km/s]')#,size=10.5)

            plt.subplots_adjust(top=0.79,
                                bottom=0.2,
                                left=0.06,
                                right=0.985,
                                hspace=0.245,
                                wspace=0.3)

        return fig

if __name__ == '__main__':

    x,y = np.linspace(-10,10,101),np.linspace(-10,10,101) 
    x,y = np.meshgrid(x,y)#,indexing='ij')
    x,y = x.ravel(),y.ravel()

    z = np.random.normal(1,0.1,x.shape)

    d1 = data(x,y)
    x_g,y_g,psf,K,indexing = d1.refined_grid_indexing(sigma_psf=1.321)
    x_g1,y_g1,psf1,K1,indexing1 = d1.refined_grid_var_resolution(sigma_psf=1.321,N_px=10,resolution=2)
    
    #x_g1,y_g1,psf1,K1 = d1.refined_grid(sigma_psf=1.321)



    fig = d1.all_maps_kpc(z,
                          z,
                          z,
                          z,
                          z,
                          z,
                          z,
                          z,
                          z,
                          z,
                          z,
                          False)
    fig.show()

    plt.show()