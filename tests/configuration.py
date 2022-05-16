import yaml
from BANG.parameter_search import GalaxyModel
import BANG.data as d
import numpy as np 
import cpnest
from BANG.data import data
from BANG.result_analysis import Result_Analysis
from BANG.model_creation import model 
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import os

def read_yaml(file_path):
    ''' 
    Function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    read yaml (i.e. configuration) file
        
        Parameters:
            file_path: str
                absolute path to read
        Returns:
    '''
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def read_user(dataset_path,use_ML):
    ''' 
    Function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    read data (i.e. configuration) file. The user should adapt it to its 
    own dataset
        
        Parameters:
            dataset_path: str
                absolute path of the data file
            use_ML: bool
                If False then ML *= np.nan such that it is not 
                used in the analysis.
        Returns:
            x_arcsec: 1D arr float32
                x positions in arcsecond
            y_arcsec: 1D arr float32
                y positions in arcsecond
            x_kpc: 1D arr float32
                x positions in kpc
            y_kpc: 1D arr float32
                y positions in kpc
            ibrightness_data: 1D arr float32
                log10 brightness (i-band in this case).
                Use np.nan for masking pixels.
            v_data: 1D arr float32
                line of sight velocity data.
                Use np.nan for masking pixels.
            sigma_data: 1D arr float32
                line of sight velocity dispersion data.
                Use np.nan for masking pixels.
            ML_data: 1D arr float32
                Mass to light ratio data.
                Use np.nan for masking pixels.
            ibrightness_error: 1D arr float32
                log10 brightness errror (i-band in this case).
                Use np.nan for masking pixels.
            v_error: 1D arr float32
                line of sight velocity data errror.
                Use np.nan for masking pixels.
            sigma_error: 1D arr float32
                line of sight velocity dispersion data errror.
                Use np.nan for masking pixels.
            ML_error: 1D arr float32
                Mass to light ratio data errror.
                Use np.nan for masking pixels.
            goodpix: 1D arr float32
                If goodpix[i] == 0. then pixel "i" is not considered.
    '''
    a = np.load(dataset_path)
    x_arcsec = a[:,0]
    y_arcsec = a[:,1]
    x_kpc = a[:,2]
    y_kpc = a[:,3]
    ibrightness_data  = a[:,4]
    v_data            = a[:,5]
    sigma_data        = a[:,6]
    ML_data           = a[:,7]
    ibrightness_error = a[:,8]
    v_error           = a[:,9]
    sigma_error       = a[:,10]
    ML_error          = a[:,11]
    goodpix           = a[:,12]   
    SN                = a[:,13]

    if use_ML:
        pass
    else:
        ML_data *= np.nan
        ML_error *= np.nan
    
    return x_arcsec,y_arcsec,x_kpc,y_kpc,ibrightness_data,v_data,sigma_data,ML_data,ibrightness_error,v_error,sigma_error,ML_error,goodpix 

class Config_reader(data,Result_Analysis,model):
    def __init__(self,config_path,gpu_device=0):
        ''' 
        A class for reading configuration files and performing parameter estimation and all diagnostic plots.
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------                                           
            Attributes:
                    config_path: str
                        name/path of the configuration file
                    gpu_device: int16
                        integer referring to the gpu machine. Useful only
                        in case of multi-gpu usage
            Methods:
                parameter_estimation(gpu_device=0):
                    Compute parameter estimation.
                diagnostic(savename='corner.png',show=True,drop_sys_rho=False,drop_halo=False)
                    Plot corner plot of the whole parameter space
                diagnostic_iteration(savename='iteration.png',show=True,drop_sys_rho=False,drop_halo=False)
                    Plot each parameter as a function of iterations
                maps_2D(savename='2D_maps.png',plot_ML=True,show=True,percentile=[50,16,84],write=True,lim='data',savedata='2D_informations.npy',vmin=None,vmax=None,close=False)
                    Plot best fit model (2D maps)
                maps_1D(savename='1D_profiles.png',plot_ML=True,show=True,percentile=[50,16,84],write=True,close=False)
                    Plot best fit model (1D maps, averaged profiles)
                maps_1D_all_outputs(savename='1D_profiles.png',plot_ML=True,show=True,percentile=[50,16,84],savedata='1D_informations.npy',write=True,close=False)
                    Plot best fit model with all the components(1D maps, averaged profiles)
                maps_2D_specify_params(Best_fit_params={},savename='2D_maps.png',plot_ML=True,show=True,write=True,lim='data',vmin=None,vmax=None,close=False)
                    Plot model with using Best_fit_params as parameters (2D maps)
                maps_1D_specify_params(Best_fit_params={},savename='1D_profiles.png',plot_ML=True,show=True,write=True,close=False)
                    Plot model with using Best_fit_params as parameters (1D maps, averaged profiles)
                best_params_all_fits(write=True,filename='all_params.txt')
                    Write best fit params to file while fitting multiples objects
                mass_plot(min_r=0.1,max_r=10,Nbins=20,figname='mass_profile.png')  
                    Plot mass profile, actually available only with Bulge + inner disc+ outer disc model  
        '''
        #super().__init__(config_path)
        #super(Config_reader,self).__init__()
        config = read_yaml(config_path)

        if "SNR_threshold_velocity" in config["Settings"]:
            self.SNR_threshold_velocity = config["Settings"]["SNR_threshold_velocity"]
        if "SNR_threshold_dispersion" in config["Settings"]:
            self.SNR_threshold_dispersion = config["Settings"]["SNR_threshold_dispersion"]
        if "SNR_threshold_ML" in config["Settings"]:
            self.SNR_threshold_ML = config["Settings"]["SNR_threshold_ML"]
        else:
            self.SNR_threshold = 0.
        if "velocity_offset" in config["Settings"]:
            self.vel_offset = config["Settings"]["velocity_offset"]
        else:
            self.vel_offset = False
        if "use_ML" in config["Settings"]:
            self.use_ML = config["Settings"]["use_ML"]
        else:
            self.use_ML = True
            
        self.redshift = config["galaxy"]["redshift"]
        self.galaxy_name = config["galaxy"]["Name"]
        # READ device and fastmath option
        self.device = config["device"]["name"]
        self.fast_math = config["device"]["fastmath"]
        
        # READ input file directory and output directory
        self.input_path = config["Settings"]["input_path"]
        self.output_path = config["Settings"]["output_path"]
        self.output_path += '/'
        # READ psf sigma in kpc
        self.kin_sigma_psf = config["Settings"]['kin_psf_sigma_kpc']
        self.pho_sigma_psf = config["Settings"]['pho_psf_sigma_kpc']
        self.N_px = config["Settings"]['N_pixels']
        self.resolution = config["Settings"]['resolution']

        galactic_components = config["galactic_components"]
        reference_frame     = config["Reference_frame"]
        systematic_errors   = config["systematic_errors"]

        # download all parameter names 
        names,LaTex_names,bounds,value,vary,self.include  = [],[],[],[],[],[]
        self.bulge_type = galactic_components['Bulge']['name']
        self.halo_type = galactic_components['Halo']['name']

        for k in galactic_components:
            # Read which component to include
            self.include.append(galactic_components[k]['include'])
            if galactic_components[k]['include'] == True:
                # choose parameters
                for j in galactic_components[k]["parameters"]:
                    names.append(galactic_components[k]["parameters"][j]['name'])
                    LaTex_names.append(galactic_components[k]["parameters"][j]['LaTex_name'])
                    vary.append(galactic_components[k]["parameters"][j]['vary'])
                    bounds.append(galactic_components[k]["parameters"][j]['bounds'])
                    value.append(galactic_components[k]["parameters"][j]['value'])
            else:
                # exclude component
                pass

        for k in reference_frame:
            names.append(reference_frame[k]["name"])
            LaTex_names.append(reference_frame[k]["LaTex_name"])
            bounds.append(reference_frame[k]["bounds"])
            vary.append(reference_frame[k]["vary"])
            value.append(reference_frame[k]["value"])  

        for k in systematic_errors:
            names.append(systematic_errors[k]["name"])
            LaTex_names.append(systematic_errors[k]["LaTex_name"])
            bounds.append(systematic_errors[k]["bounds"])
            vary.append(systematic_errors[k]["vary"])
            value.append(systematic_errors[k]["value"])

        if config["Settings"]["sys_rho"] == True:
            pass 
        else:
            print("Error likelihood without systematic error is not implemented")
            print("Put fixed=True and value=-10 in systematic_errors")
            exit()


        # Build all the relevant dictionaries
        self.variable_quantities = {names[i]:bounds[i] for i in range(0,len(names)) if vary[i]==True}
        self.fixed_quantities = {names[i]:value[i] for i in range(0,len(names)) if vary[i]==False}
        self.variable_LaTex = {names[i]:LaTex_names[i] for i in range(0,len(names)) if vary[i]==True}
        self.fixed_LaTex = {names[i]:LaTex_names[i] for i in range(0,len(names)) if vary[i]==False}


        # READ dataset
        self.x_arcsec,self.y_arcsec,self.x,self.y,self.ibrightness_data,self.v_data,self.sigma_data,self.ML_data,\
        self.ibrightness_error,self.v_error,self.sigma_error,self.ML_error,self.goodpix = read_user(self.input_path,velocity_offset=self.vel_offset,\
                                                                                                    use_ML=self.use_ML,SNR_threshold_velocity=self.SNR_threshold_velocity,\
                                                                                                    SNR_threshold_dispersion=self.SNR_threshold_dispersion,\
                                                                                                    SNR_threshold_ML=self.SNR_threshold_ML)

        #####################################################################
        #####################################################################
        # CREATE data_object useful for later plots
        #####################################################################
        #####################################################################
        self.data_obj = data(self.x,self.y)
        # 0.009 assumes FWHM of 0.3 arcsec
        #self.x_grid,self.y_grid,self.psf = self.data_obj.refined_grid(size=int(np.sqrt(self.K)),sigma_psf=self.sigma_psf,sampling='linear',N_psf=4)
        self.x_grid,self.y_grid,self.kin_psf,self.K,self.indexing = self.data_obj.refined_grid_indexing(sigma_psf=self.kin_sigma_psf,N_px=self.N_px,resolution=self.resolution)
        _,_,self.pho_psf,_,_ = self.data_obj.refined_grid_indexing(sigma_psf=self.pho_sigma_psf,N_px=self.N_px,resolution=self.resolution)

        self.arcsec_to_kpc = abs(self.x[0]-self.x[1])/abs(self.x_arcsec[0]-self.x_arcsec[1])
        #####################################################################
        #####################################################################
        # CREATE model_object useful for later plots
        #####################################################################
        #####################################################################
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        H_z   = cosmo.H(self.redshift).value # this is in km/Mpc/s
        H_z   = H_z*1e5/(1e6*3.086e18)         # this is in 1/s
        G     = 4.299e-6                       # kcp*km^2/M_sun/s^2
        G_1   = G * (1e5*1e-3*3.24078e-19)**2  # kcp^3/M_sun/s^2
        self.rho_crit = 3*H_z*H_z/(8*np.pi*G_1)

        self.model_obj = model(self.device,
                              gpu_device,
                              self.fast_math,
                              self.include,
                              self.bulge_type,
                              self.halo_type,
                              self.data_obj.N,
                              self.data_obj.J,
                              self.K,
                              self.x_grid,
                              self.y_grid,
                              self.ibrightness_data,
                              self.v_data,
                              self.sigma_data,
                              self.ML_data,
                              self.ibrightness_error,
                              self.v_error,
                              self.sigma_error,
                              self.ML_error,
                              self.kin_psf,
                              self.pho_psf,
                              self.indexing,
                              self.goodpix,
                              self.rho_crit)
        #####################################################################
        #####################################################################
        # CREATE result_object useful for later plots
        #####################################################################
        #####################################################################
        self.result_obj = Result_Analysis(self.output_path,
                                          self.rho_crit,
                                          self.halo_type)

        #####################################################################
        #####################################################################
        # READ SPECIFIC OF NESTED SAMPLING
        #####################################################################
        #####################################################################
        self.verbose      = config["parameter_estimator"]["Nested_sampling"]["verbose"]
        self.poolsize     = config["parameter_estimator"]["Nested_sampling"]["poolsize"]
        self.nthreads     = config["parameter_estimator"]["Nested_sampling"]["nthreads"]
        self.nlive        = config["parameter_estimator"]["Nested_sampling"]["nlive"]
        self.maxmcmc      = config["parameter_estimator"]["Nested_sampling"]["maxmcmc"]
        self.nhamiltonian = config["parameter_estimator"]["Nested_sampling"]["nhamiltonian"]
        self.nslice       = config["parameter_estimator"]["Nested_sampling"]["nslice"]
        self.resume       = config["parameter_estimator"]["Nested_sampling"]["resume"]

    def parameter_estimation(self,gpu_device=0):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Start parameter estimation assuming model specified in configuration file
            
            Parameters:
                gpu_device: int16
                    ID of the GPU. If multi-gpu is not needed, set it to zero.

            Returns:

        '''
        M = GalaxyModel(self.device,
                        gpu_device,
                        self.fast_math,
                        self.include,
                        self.bulge_type,
                        self.halo_type,
                        self.data_obj.N,
                        self.data_obj.J,
                        self.K,
                        self.x_grid,
                        self.y_grid,
                        self.ibrightness_data,
                        self.v_data,
                        self.sigma_data,
                        self.ML_data,
                        self.ibrightness_error,
                        self.v_error,
                        self.sigma_error,
                        self.ML_error,
                        self.kin_psf,
                        self.pho_psf,
                        self.indexing,
                        self.goodpix,
                        self.variable_quantities,  
                        self.fixed_quantities,
                        self.rho_crit)

        work=cpnest.CPNest(M,
                           verbose  = self.verbose,
                           poolsize = self.poolsize,
                           nthreads = self.nthreads ,
                           nlive    = self.nlive,
                           maxmcmc  = self.maxmcmc,
                           output   = self.output_path,
                           nhamiltonian = self.nhamiltonian,
                           nslice   = self.nslice,
                           resume   = self.resume,
                           periodic_checkpoint_interval = 600)
        work.run()

        work.get_posterior_samples()
        work.get_nested_samples()

    def diagnostic(self,savename='corner.png',show=True,drop_sys_rho=False,drop_halo=False):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Plot posterior probabilities of all the parameters, assume existence of 
        "posterior.dat" file.
            
            Parameters:
                savename: str
                    Name for saving the plot
                show: Bool
                    If plot should be shown on screen
                drop_sys_rho: Bool
                    If used and set to True put systematic_error of the brightnees in 
                    cornern plot
                drop_halo: Bool
                    If True does not show halo posteriors

            Returns:
                fig: figure
                    corner plot
        '''
        fig = self.result_obj.corner_plot(self.variable_LaTex)
        fig.savefig(self.output_path+savename)
        if show==True:
            fig.show()
            plt.show()
        return fig
    
    def diagnostic_iteration(self,savename='iteration.png',show=True,drop_sys_rho=False,drop_halo=False):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Plot parameters as a function of iteration in the nested sampling, assume existence of 
        "nested_samples.dat" file.
            
            Parameters:
                savename: str
                    Name for saving the plot
                show: Bool
                    If plot should be shown on screen
                drop_sys_rho: Bool
                    If used and set to True put systematic_error of the brightnees in 
                    cornern plot
                drop_halo: Bool
                    If True does not show halo posteriors

            Returns:
                fig: figure
                    corner plot
        '''
        fig = self.result_obj.iteration_plot(self.variable_LaTex)
        fig.savefig(self.output_path+savename)
        if show==True:
            fig.show()
            plt.show()
        return fig

    def maps_2D(self,savename='2D_maps.png',plot_ML=True,show=True,percentile=[50,16,84],write=True,lim='data',savedata='2D_informations.npy',vmin=None,vmax=None,close=False):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Plot best fit model assuming existence of "posterior.dat" file. Plot is 
        3 X 4 or 3 X 3. 
        First line is the best fit model, second line are the data and third line are the residuals.
        From left to right: brightness,velocity,dispersion and (possibly) Mass to light ratio.
            
            Parameters:
                savename: str
                    Name for saving the plot
                plot_ML: str
                    If True plot also Mass to light ratio
                show: Bool
                    If plot should be shown on screen
                percentile: 1D arr float32 
                    Which "percentiles" to use for computing best fit params and error.
                    Default is:
                            - 50 (median) for best fit params
                            - 16 for lower error
                            - 84 for upper error
                write: Bool
                    Whether or not save to file the best fit params and errors.
                lim: str
                    Wheter to use "data" or "model" or "user" for the limits of the plot
                vmin: 1D arr float32
                    In case lim is set to "user" specify min value of the colormap
                vmax: 1D arr float32
                    In case lim is set to "user" specify max value of the colormap
                savedata: str
                    name of the file for saving best fit model values. Data are saved in the 
                    following format:
                        -tot[:,0] : x in arcsec
                        -tot[:,1] : y in arcsec
                        -tot[:,2] : x in kpc
                        -tot[:,3] : y in kpc
                        -tot[:,4] : brightness data (log10)
                        -tot[:,5] : brightness model (log10)
                        -tot[:,6] : v data
                        -tot[:,7] : v model
                        -tot[:,8] : sigma data
                        -tot[:,9] : sigma model
                        -tot[:,10] : ML data
                        -tot[:,11] : ML model
                        -tot[:,12] : brightness log10 error
                        -tot[:,13] : v data error
                        -tot[:,14] : sigma data error
                        -tot[:,15] : ML data error

            Returns:
                ibrightness_model: 1D arr float32
                    best fit brightness
                v_model: 1D arr float32
                    best fit velocity
                sigma_model: 1D arr float32
                    best fit dispersion
                ML_model: 1D arr float32
                    best fit mass to ligh
                fig: figure

        '''
        Best_fit_params,_,_ = self.result_obj.best_params(percentile=percentile,write=write)
        all_params = {**Best_fit_params, **self.fixed_quantities}
        ibrightness_model,_,_,_,v_model,_,_,_,sigma_model,_,_,_,LM_ratio,_,_,_ = self.model_obj.model(all_params)
        ML_model = 1.0/LM_ratio
        #ML_model = np.log10(1.0/LM_ratio)
        fig = self.data_obj.all_maps_kpc(self.ibrightness_data,
                                    np.log10(ibrightness_model),
                                    self.v_data,
                                    v_model,
                                    self.sigma_data,
                                    sigma_model,
                                    self.ML_data,
                                    ML_model,
                                    10**self.ibrightness_data*self.ibrightness_error*np.log(10),
                                    self.v_error,
                                    self.sigma_error,
                                    self.ML_error,
                                    plot_ML=plot_ML,
                                    lim = lim,
                                    vmin=vmin,
                                    vmax=vmax)
        fig.savefig(self.output_path+savename)
        if show==True:
            fig.show()
            plt.show()
        tot = np.concatenate((self.x_arcsec.reshape(-1,1),
                              self.y_arcsec.reshape(-1,1),
                              self.x.reshape(-1,1),
                              self.y.reshape(-1,1),
                              self.ibrightness_data.reshape(-1,1),
                              np.log10(ibrightness_model).reshape(-1,1),
                              self.v_data.reshape(-1,1),
                              v_model.reshape(-1,1),
                              self.sigma_data.reshape(-1,1),
                              sigma_model.reshape(-1,1),
                              self.ML_data.reshape(-1,1),
                              ML_model.reshape(-1,1),
                              self.v_error.reshape(-1,1),
                              self.sigma_error.reshape(-1,1),
                              self.ML_error.reshape(-1,1),
                              ),axis=1)
        np.save(self.output_path+savedata,tot)

        return ibrightness_model,v_model,sigma_model,ML_model,fig
        
    def maps_1D(self,savename='1D_profiles.png',plot_ML=True,show=True,percentile=[50,16,84],write=True,close=False):
        ''' 
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Plot best fit model assuming existence of "posterior.dat" file. Plot is 
        1 X 4 or 1 X 3. 
        From left to right: brightness,velocity,dispersion and (possibly) Mass to light ratio.
        Profiles are azimuthally averaged.

            Parameters:
                savename: str
                    Name for saving the plot
                plot_ML: str
                    If True plot also Mass to light ratio
                show: Bool
                    If plot should be shown on screen
                percentile: 1D arr float32 
                    Which "percentiles" to use for computing best fit params and error.
                    Default is:
                            - 50 (median) for best fit params
                            - 16 for lower error
                            - 84 for upper error
                write: Bool
                    Whether or not save to file the best fit params and errors.
            Returns:
                fig: figure

        '''
        Best_fit_params,_,_ = self.result_obj.best_params(percentile=percentile,write=write)
        all_params = {**Best_fit_params , **self.fixed_quantities}
        ibrightness_model,_,_,_,v_model,_,_,_,sigma_model,_,_,_,LM_ratio,_,_,_ = self.model_obj.model(all_params)

        ML_model = 1.0/LM_ratio
        #ML_model = np.log10(1.0/LM_ratio)
        minr = 0.4  # dimension of pixel is 0.5 arcsec
        maxr_kin = np.nanmax((self.x_arcsec**2+self.y_arcsec**2)**0.5*self.v_data/self.v_data)
        maxr_phot = 1.2*maxr_kin
        maxr_kin -= 1.5
        Nbin_phot = int((maxr_phot-minr)/1.0)
        Nbin_kin  = int((maxr_kin-minr)/1.0)

        Nbin_phot = max(Nbin_phot,7) 
        Nbin_kin  = max(Nbin_kin,7) 


        r_avg_B,B_avg_model  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,ibrightness_model,quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_avg_data         = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,10**self.ibrightness_data,quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_avg_err          = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,10**self.ibrightness_data*self.ibrightness_error*np.log(10),quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        
        r_avg_v,v_avg_model  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,v_model,quantity="velocity",estimator='mean',
                                   min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_avg_data         = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.v_data,quantity="velocity",estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_avg_err          = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.v_error,quantity="velocity",estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        
        r_avg_sigma,sigma_avg_model = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,sigma_model,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_avg_data            = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.sigma_data,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_avg_err            = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.sigma_error,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        
        r_avg_ML,ML_avg_model = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,ML_model,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_avg_data         = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.ML_data,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_avg_err          = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.ML_error,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)

        #B_avg_err = np.nanmean(10**self.ibrightness_data*self.ibrightness_error*np.log(10))
        #v_avg_err = np.nanmean(self.v_error)
        #sigma_avg_err = np.nanmean(self.sigma_error)
        #ML_avg_err = np.nanmean(10**self.ML_data*self.ML_error*np.log(10))
        #ML_avg_err = np.nanmean(self.ML_error)

        fig = self.data_obj.all_1D_profile(
                                      self.arcsec_to_kpc,
                                      r_avg_B,
                                      B_avg_data,
                                      B_avg_model,
                                      r_avg_v,
                                      v_avg_data,
                                      v_avg_model,
                                      r_avg_sigma,
                                      sigma_avg_data,
                                      sigma_avg_model,
                                      r_avg_ML,
                                      ML_avg_data,
                                      ML_avg_model,
                                      B_avg_err,
                                      v_avg_err,
                                      sigma_avg_err,
                                      ML_avg_err,
                                      plot_ML=plot_ML)
        
        fig.savefig(self.output_path+savename)
        if show==True:
            fig.show()
            plt.show()

        return fig

    def maps_1D_all_outputs(self,savename='1D_profiles.png',plot_ML=True,show=True,percentile=[50,16,84],savedata='1D_informations.npy',write=True,close=False):
        '''
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Plot best fit model assuming existence of "posterior.dat" file. Plot is 
        1 X 4 or 1 X 3. 
        From left to right: brightness,velocity,dispersion and (possibly) Mass to light ratio.
        Profiles are azimuthally averaged. All components are present.
            
            Parameters:
                savename: str
                    Name for saving the plot
                plot_ML: str
                    If True plot also Mass to light ratio
                show: Bool
                    If plot should be shown on screen
                percentile: 1D arr float32 
                    Which "percentiles" to use for computing best fit params and error.
                    Default is:
                            - 50 (median) for best fit params
                            - 16 for lower error
                            - 84 for upper error
                write: Bool
                    Whether or not save to file the best fit params and errors.
                savedata: str
                    name of the file for saving best fit model values. Data are saved in the 
                    
                    see below the format
                       
            Returns:
                ibrightness_model: 1D arr float32
                    best fit brightness
                v_model: 1D arr float32
                    best fit velocity
                sigma_model: 1D arr float32
                    best fit dispersion
                ML_model: 1D arr float32
                    best fit mass to ligh
                fig: figure

        Note: this is probably working only with bulge+ inner disc+ outer disc model
        '''
        Best_fit_params,_,_ = self.result_obj.best_params(percentile=percentile,write=write)
        all_params = {**Best_fit_params , **self.fixed_quantities}
        ibrightness_model,B_B,B_D1,B_D2,v_model,v_B,v_D1,v_D2,\
        sigma_model,sigma_B,sigma_D1,sigma_D2,LM_ratio,LM_B,LM_D1,LM_D2 = self.model_obj.model(all_params)
        ML_model,ML_B,ML_D1,ML_D2 = 1.0/LM_ratio,1.0/LM_B,1.0/LM_D1,1.0/LM_D2
        #ML_model = np.log10(1.0/LM_ratio)

        minr = 0.4  # dimension of pixel is 0.5 arcsec
        maxr_kin = np.nanmax((self.x_arcsec**2+self.y_arcsec**2)**0.5*self.v_data/self.v_data)
        maxr_phot = 1.2**maxr_kin
        maxr_kin -= 1.5
        Nbin_phot = int((maxr_phot-minr)/1.0)
        Nbin_kin  = int((maxr_kin-minr)/1.0)

        Nbin_phot = max(Nbin_phot,7) 
        Nbin_kin  = max(Nbin_kin,7) 

        r_avg_B,B_avg_model  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,np.log10(ibrightness_model),quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_B_avg   = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,np.log10(B_B),quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_D1_avg  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,np.log10(B_D1),quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_D2_avg  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,np.log10(B_D2),quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_avg_data         = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.ibrightness_data,quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_avg_err          = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.ibrightness_error,quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        
        r_avg_v,v_avg_model  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,v_model,quantity="velocity",estimator='mean',
                                   min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_B_avg   = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,v_B,quantity="velocity",estimator='mean',
                                   min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_D1_avg  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,v_D1,quantity="velocity",estimator='mean',
                                   min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_D2_avg  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,v_D2,quantity="velocity",estimator='mean',
                                   min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_avg_data         = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.v_data,quantity="velocity",estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_avg_err          = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.v_error,quantity="velocity",estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        
        r_avg_sigma,sigma_avg_model = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,sigma_model,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_B_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,sigma_B,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_D1_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,sigma_D1,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_D2_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,sigma_D2,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_avg_data            = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.sigma_data,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_avg_err            = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.sigma_error,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        
        r_avg_ML,ML_avg_model = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,ML_model,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_B_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,ML_B,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_D1_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,ML_D1,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_D2_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,ML_D2,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_avg_data         = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.ML_data,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_avg_err          = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.ML_error,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)

        #B_avg_err = np.nanmean(10**self.ibrightness_data*self.ibrightness_error*np.log(10))
        #v_avg_err = np.nanmean(self.v_error)
        #sigma_avg_err = np.nanmean(self.sigma_error)
        #ML_avg_err = np.nanmean(10**self.ML_data*self.ML_error*np.log(10))
        #ML_avg_err = np.nanmean(self.ML_error)

        fig = self.data_obj.all_1D_profile_all_outputs(
                                      self.arcsec_to_kpc,
                                      r_avg_B,
                                      B_avg_data,
                                      B_avg_model,
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
                                      B_avg_err,
                                      v_avg_err,
                                      sigma_avg_err,
                                      ML_avg_err,
                                      plot_ML=plot_ML)
        
        fig.savefig(self.output_path+savename)


        tot = np.ones((int(max(Nbin_phot,Nbin_kin)),28))*np.nan

        tot[:np.size(r_avg_B),0] = r_avg_B
        tot[:np.size(B_avg_model),1] = B_avg_model
        tot[:np.size(B_B_avg),2] = B_B_avg
        tot[:np.size(B_D1_avg),3] = B_D1_avg
        tot[:np.size(B_D2_avg),4] = B_D2_avg
        tot[:np.size(B_avg_data),5] = B_avg_data
        tot[:np.size(B_avg_err),6] = B_avg_err
        tot[:np.size(r_avg_v),7] = r_avg_v
        tot[:np.size(v_avg_model),8] = v_avg_model
        tot[:np.size(v_B_avg),9] = v_B_avg
        tot[:np.size(v_D1_avg),10] = v_D1_avg
        tot[:np.size(v_D2_avg),11] = v_D2_avg
        tot[:np.size(v_avg_data),12] = v_avg_data
        tot[:np.size(v_avg_err),13] = v_avg_err
        tot[:np.size(r_avg_sigma),14] = r_avg_sigma
        tot[:np.size(sigma_avg_model),15] = sigma_avg_model
        tot[:np.size(sigma_B_avg),16] = sigma_B_avg
        tot[:np.size(sigma_D1_avg),17] = sigma_D1_avg
        tot[:np.size(sigma_D2_avg),18] = sigma_D2_avg
        tot[:np.size(sigma_avg_data),19] = sigma_avg_data
        tot[:np.size(sigma_avg_err),20] = sigma_avg_err
        tot[:np.size(r_avg_ML),21] = r_avg_ML
        tot[:np.size(ML_avg_model),22] = ML_avg_model
        tot[:np.size(ML_B_avg),23] = ML_B_avg
        tot[:np.size(ML_D1_avg),24] = ML_D1_avg
        tot[:np.size(ML_D2_avg),25] = ML_D2_avg
        tot[:np.size(ML_avg_data),26] = ML_avg_data
        tot[:np.size(ML_avg_err),27] = ML_avg_err
        np.save(self.output_path+savedata,tot)

        if show==True:
            fig.show()
            plt.show()
        return fig


    def maps_2D_specify_params(self,Best_fit_params={},savename='2D_maps.png',plot_ML=True,show=True,write=True,lim='data',vmin=None,vmax=None,close=False):
        all_params = {**Best_fit_params, **self.fixed_quantities}
        ibrightness_model,_,_,_,v_model,_,_,_,sigma_model,_,_,_,LM_ratio,_,_,_ = self.model_obj.model(all_params)
        ML_model = 1.0/LM_ratio
        #ML_model = np.log10(1.0/LM_ratio)

        fig = self.data_obj.all_maps_kpc(self.ibrightness_data,
                                    np.log10(ibrightness_model),
                                    self.v_data,
                                    v_model,
                                    self.sigma_data,
                                    sigma_model,
                                    self.ML_data,
                                    ML_model,
                                    self.v_error,
                                    self.sigma_error,
                                    self.ML_error,
                                    plot_ML=plot_ML,
                                    lim = lim,
                                    vmin=vmin,
                                    vmax=vmax)
        fig.savefig(self.output_path+savename)
        if show==True:
            fig.show()
            plt.show()
        return ibrightness_model,v_model,sigma_model,ML_model,fig

    def maps_1D_specify_params(self,Best_fit_params={},savename='1D_profiles.png',plot_ML=True,show=True,write=True,close=False):
        all_params = {**Best_fit_params , **self.fixed_quantities}
        ibrightness_model,B_B,B_D1,B_D2,v_model,v_B,v_D1,v_D2,\
        sigma_model,sigma_B,sigma_D1,sigma_D2,LM_ratio,LM_B,LM_D1,LM_D2 = self.model_obj.model(all_params)
        ML_model,ML_B,ML_D1,ML_D2 = 1.0/LM_ratio,1.0/LM_B,1.0/LM_D1,1.0/LM_D2
        #ML_model = np.log10(1.0/LM_ratio)

        minr = 0.4  # dimension of pixel is 0.5 arcsec
        maxr_kin = np.nanmax((self.x_arcsec**2+self.y_arcsec**2)**0.5*self.v_data/self.v_data)
        maxr_phot = 2*maxr_kin
        maxr_kin -= 1.5
        Nbin_phot = int((maxr_phot-minr)/1.0)
        Nbin_kin  = int((maxr_kin-minr)/1.0)

        Nbin_phot = max(Nbin_phot,7) 
        Nbin_kin  = max(Nbin_kin,7) 


        r_avg_B,B_avg_model  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,np.log10(ibrightness_model),quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_B_avg   = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,np.log10(B_B),quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_D1_avg  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,np.log10(B_D1),quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_D2_avg  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,np.log10(B_D2),quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_avg_data         = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.ibrightness_data,quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,B_avg_err          = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.ibrightness_error,quantity=None,estimator='mean',
                                   min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        
        r_avg_v,v_avg_model  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,v_model,quantity="velocity",estimator='mean',
                                   min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_B_avg   = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,v_B,quantity="velocity",estimator='mean',
                                   min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_D1_avg  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,v_D1,quantity="velocity",estimator='mean',
                                   min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_D2_avg  = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,v_D2,quantity="velocity",estimator='mean',
                                   min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_avg_data         = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.v_data,quantity="velocity",estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        _,v_avg_err          = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.v_error,quantity="velocity",estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=False)
        
        r_avg_sigma,sigma_avg_model = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,sigma_model,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_B_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,sigma_B,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_D1_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,sigma_D1,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_D2_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,sigma_D2,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_avg_data            = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.sigma_data,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        _,sigma_avg_err            = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.sigma_error,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_kin,Nbins=Nbin_kin,azimuth_division=True)
        
        r_avg_ML,ML_avg_model = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,ML_model,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_B_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,ML_B,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_D1_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,ML_D1,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_D2_avg = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,ML_D2,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_avg_data         = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.ML_data,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)
        _,ML_avg_err          = self.result_obj.radial_profile(self.x_arcsec,self.y_arcsec,self.ML_error,quantity=None,estimator='mean',
                                                       min_r=minr,max_r=maxr_phot,Nbins=Nbin_phot,azimuth_division=True)

        #B_avg_err = np.nanmean(10**self.ibrightness_data*self.ibrightness_error*np.log(10))
        #v_avg_err = np.nanmean(self.v_error)
        #sigma_avg_err = np.nanmean(self.sigma_error)
        #ML_avg_err = np.nanmean(10**self.ML_data*self.ML_error*np.log(10))
        #ML_avg_err = np.nanmean(self.ML_error)

        fig = self.data_obj.all_1D_profile_all_outputs(
                                      self.arcsec_to_kpc,
                                      r_avg_B,
                                      B_avg_data,
                                      B_avg_model,
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
                                      B_avg_err,
                                      v_avg_err,
                                      sigma_avg_err,
                                      ML_avg_err,
                                      plot_ML=plot_ML)
        
        fig.savefig(self.output_path+savename)
        if show==True:
            fig.show()
            plt.show()

        return fig

    def best_params_all_fits(self,write=True,filename='all_params.txt'):

        Best_Fit,LE_16,UE_84 = self.result_obj.best_params()
        _,LE_05,UE_95 = self.result_obj.best_params(percentile=[50,5,95])

        if os.path.exists(filename):
            f = open(filename,'a')
            f.write(self.galaxy_name+'\t')
            for nn in list(Best_Fit.keys()):
                f.write(str(np.around(Best_Fit[nn],decimals=4))+'\t')
            for nn in list(Best_Fit.keys()):
                f.write(str(np.around(LE_16[nn],decimals=4))+ '\t')
            for nn in list(Best_Fit.keys()):
                f.write(str(np.around(UE_84[nn],decimals=4))+'\t')
            for nn in list(Best_Fit.keys()):
                f.write(str(np.around(LE_05[nn],decimals=4))+'\t')
            for nn in list(Best_Fit.keys()):
                f.write(str(np.around(UE_95[nn],decimals=4))+'\t')
            f.write('\n')
        else:
            f = open(filename,'w')
            f.write('# plate-IFU\t')
            for nn in list(Best_Fit.keys()):
                f.write(nn+'\t')
            for nn in list(Best_Fit.keys()):
                f.write(nn+'_LE16 \t')
            for nn in list(Best_Fit.keys()):
                f.write(nn+'_UE84 \t')
            for nn in list(Best_Fit.keys()):
                f.write(nn+'_LE05 \t')
            for nn in list(Best_Fit.keys()):
                f.write(nn+'_UE95 \t')
            f.write('\n')

            f.write(self.galaxy_name+'\t')
            for nn in list(Best_Fit.keys()):
                f.write(str(np.around(Best_Fit[nn],decimals=4))+'\t')
            for nn in list(Best_Fit.keys()):
                f.write(str(np.around(LE_16[nn],decimals=4))+ '\t')
            for nn in list(Best_Fit.keys()):
                f.write(str(np.around(UE_84[nn],decimals=4))+'\t')
            for nn in list(Best_Fit.keys()):
                f.write(str(np.around(LE_05[nn],decimals=4))+'\t')
            for nn in list(Best_Fit.keys()):
                f.write(str(np.around(UE_95[nn],decimals=4))+'\t')
            f.write('\n')

        f.close()

    def mass_plot(self,min_r=0.1,max_r=10,Nbins=20,figname='mass_profile.png'):
        '''
        Class Method
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Plot mass profile for best fit model
            
        Note: this is probably working only with bulge+ inner disc+ outer disc model
        '''

        r,Mbulge,Mdisc1,Mdisc2,Mhalo = self.result_obj.mass_profile(min_r=min_r,max_r=max_r,Nbins=Nbins)
        fig = plt.figure()
        ax  = fig.add_subplot()
        ax.plot(r,Mbulge,'r',lw=3,label='Bulge')
        ax.plot(r,Mdisc1,'y',lw=3,label='Inner Disc')
        ax.plot(r,Mdisc2,'b',lw=3,label='Outer Disc')
        ax.plot(r,Mhalo,'k',lw=3,label='Halo')
        ax.set_yscale('log')
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel(r'$M [M_{\odot}]$')

        fig.savefig(self.output_path+figname)

if __name__=='__main__':
    pass