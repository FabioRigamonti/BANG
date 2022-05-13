from BANG.configuration import Config_reader
import matplotlib.pyplot as plt

'''
Example for running the code: Parameter estimation and diagnostic plots

In the test directory there should be a file "mock_data.npy". The following code
will try to fit that data. I do not recommend running it on CPU.

For any user the steps for using BANG are the following:

- Get your data, go to the file configuration.py and change the function read_user in 
  ordet to read your data.
  Data should be a N X 13 matrix with the following columns:
        - column 0  should be x in arcsecond
        - column 1  should be y in arcsecond
        - column 2  should be x in kpc
        - column 3  should be y in kpc
        - column 4  should be the logarithm of the surface brightness in M_sun/kpc^2
        - column 5  should be the line of sight velocity in km/s
        - column 6  should be the line of sight velocity dispersion in km/s
        - column 7  should be the M/L ratio in M_sun/L_sun
        - column 8  should be the error of the logarithm of the surface brightness in M_sun/kpc^2
        - column 9  should be the error of the line of sight velocity in km/s
        - column 10 should be the error of the line of sight velocity dispersion in km/s
        - column 11 should be the error of the M/L ratio in M_sun/L_sun
        - column 12 should contain 0.0 in the pixels that you do not want to use for any of 
                    the observed quantities, put 1.0 if you want to use the pixel.

    Up to now support only galaxies with M X M pixels (number of pixels in the x direction
    equal to the number of pixels in the y direction, upgrade will be available soon). So 
    please if you galaxy is of dimension M X L add columns or rows in order to make it M X M
    or L X L. The you can simply put 0.0 in column 12 such that the new mock pixels will not be 
    used in the parameter estimation.

    Consider also that all the data in you data file must be 1D array. So if you have a M X M 
    matrix remember to flatten it. 

    Suppose that your data have a total number of pixels N then if we define n = int(N**0.5):
        y (or x) should be an array like [-1,0,1,-1,0,1,-1,0,1] where the [-1,0,1] slice has
            dimension n and it is repeated n times
        x (or y) should be an array like [3,3,3, 4,4,4, 5,5,5] where you have n different points
            each one repeated n times. 
    
    Note that it is not necessary to have your grid centered in zero.

    Use np.nan in order to mask the data that you do not want to use. For example if you do
    not have M/L ratio measurements simply put in column 7 all np.nan. Or put use_ML = False
    in your config file.

- Look at the config.yaml file and change all the parameters according the model that you want 
  to run. Check the components that you want to include, the ranges for the parameter search 
  and the input/output directories. All parameters are explained in the config.yaml file.

- run a script similar to the one below and wait for the result. On a standard gpu you parameter
  estimation should last ~ 1h. 

Please if you want to try band on the mock_data.npy file go to the test directory and launch the 
command:
                python3 example_script.py

All the useful function should be explained in their docscring.

For any problem in running BANG or in creating the data structure feel free to write to:
    
                            frigamonti@uninsubria.it

'''

if __name__=='__main__':

    # Configuration file name
    config_name = 'config.yaml'

    # Read configuration
    config = Config_reader(config_name)
    
    # Start parameter estimation
    config.parameter_estimation()

    # Plot corner plot with all the posterior probabilities
    corner = config.diagnostic(show=False)

    # Plot all the parameters tried during the fit
    iteration = config.diagnostic_iteration(show=False)
    
    # Compute 2D model with the best fit parameters
    B_model,v_model,sigma_model,ML_model,maps_2D = config.maps_2D(show=False,lim='model')

    # Compute 1D model with the best fit parameters
    maps_1D = config.maps_1D_all_outputs(show=False)
    
    # Show all plots
    corner.show()
    maps_2D.show()
    maps_1D.show()
    iteration.show()
    plt.show()

    # Uncomment following section for plotting model with specific set of parameters
    #Mb,Rb,Md1,Rd1,Md2,Rd2,logf,c = 10.55,0.3,10.1,0.4,10.5,1,1.716,12.966
    #x0,y0,theta,incl = -0.122,-0.063,2.691,1.109
    #k1,k2,ML_b,ML_d1,ML_d2 = 0.861,0.894,4.0,2.,1.
    #gamma = 1.7
    #sys_rho = -1.0
    #params = {'log10_bulge_mass':Mb,
    #           'log10_disc1_mass':Md1,
    #           'log10_disc2_mass':Md2,
    #           'log10_halo_fraction':logf,
    #           'log10_bulge_radius':Rb,
    #           'log10_disc1_radius':Rd1,
    #           'log10_disc2_radius':Rd2,
    #           'concentration':c,
    #           'x0':x0,
    #           'y0':y0,
    #           'orientation':theta,
    #           'inclination':incl,
    #           'ML_b':ML_b,
    #           'ML_d1':ML_d1,
    #           'ML_d2':ML_d2,
    #           'k1':k1,
    #           'k2':k2,
    #           'gamma':gamma,
    #           'sys_rho':sys_rho}

    #B_model,v_model,sigma_model,ML_model,maps_2D = config.maps_2D_specify_params(params,savename='2D_maps_mine.png',show=False,lim='model')

    #maps_1D = config.maps_1D_specify_params(params,show=False,savename='1D_maps_mine.png')

    #corner.show()
    #maps_2D.show()
    #maps_1D.show()
    #iteration.show()
    #plt.show()
