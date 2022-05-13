import numpy as np 
import matplotlib.pyplot as plt 
import corner
from matplotlib import ticker
import math
import os 


def dehenen_mass(M,R,gamma,r):
    Mbulge = M*(r/(r+R))**(3-gamma)
    return Mbulge

def NFW_mass(M,c,R,r):
    s = r/R
    M_halo = M/(np.log(1+c)-c/(1+c))*(np.log(1+s)-s/(1+s))
    return M_halo

def exp_disc_mass(M,R,r):
    Mdisc = M*(1-np.exp(-r/R)*(1+r/R))
    return Mdisc


class Result_Analysis:
    def __init__(self,
                 output_path,
                 rho_crit,
                 halo_type):
        
        self.output_path = output_path 
        self.rho_crit = rho_crit 
        self.halo_type = halo_type 

    def corner_plot(self,LaTex_dictionary,drop_sys_rho=False,drop_halo=False):
        LaTex_names = list(LaTex_dictionary.values())

        file_path  = self.output_path+'posterior.dat'
        f = open(file_path,'r')
        names = f.readline().split()[1:]
        f.close()
        posteriors = np.loadtxt(self.output_path+'posterior.dat')
        posterior_dict = {names[i]:posteriors[:,i] for i in range(len(names)-2)}

        log10M_halo = 0.
        if 'log10_disc2_mass' in posterior_dict:
            log10M_halo += 10**posterior_dict['log10_disc2_mass']
        if 'log10_disc1_mass' in posterior_dict:
            log10M_halo += 10**posterior_dict['log10_disc1_mass']
        if 'log10_bulge_mass' in posterior_dict:
            log10M_halo += 10**posterior_dict['log10_bulge_mass']
        log10M_halo = np.log10(10**posterior_dict['log10_halo_fraction']*log10M_halo)

        posterior_dict = {**posterior_dict,**{'log10_halo_mass':log10M_halo}}
        LaTex_dictionary = {**LaTex_dictionary,**{'log10_halo_mass':r'$log_{10}(M_{h})$'}}

        if self.halo_type == 'NFW':
            log10R_halo = (3*10**log10M_halo/(800.*np.pi*self.rho_crit))**(1/3)/posterior_dict['concentration']
            log10R_halo = np.log10(log10R_halo)
            posterior_dict = {**posterior_dict,**{'log10_halo_radius':log10R_halo}}
            LaTex_dictionary = {**LaTex_dictionary,**{'log10_halo_radius':r'$log_{10}(R_{h})$'}}
            if 'log10_disc2_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc2_radius']
            elif 'log10_disc1_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc1_radius']
            
            c = posterior_dict['concentration']
            s = r_trunc/10**log10R_halo
            M_inside = 10**log10M_halo/(np.log(1+c)-c/(1+c))*(np.log(1+s)-s/(1+s))
            M_inside = np.log10(M_inside)
            posterior_dict = {**posterior_dict,**{'log10_M5_mass':M_inside}}
            LaTex_dictionary = {**LaTex_dictionary,**{'log10_M5_mass':r'$log_{10}(M_{h,5})$'}}
        else:
            if 'log10_disc2_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc2_radius']
                M_inside = 10**posterior_dict['log10_halo_mass'] * (r_trunc**2/(r_trunc+10**posterior_dict['log10_halo_radius'])**2)
                M_inside = np.log10(M_inside)

                posterior_dict = {**posterior_dict,**{'log10_M5_mass':M_inside}}
                LaTex_dictionary = {**LaTex_dictionary,**{'log10_M5_mass':r'$log_{10}(M_{h,5})$'}}

            elif 'log10_disc1_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc1_radius']
                M_inside = 10**posterior_dict['log10_halo_mass'] * (r_trunc**2/(r_trunc+10**posterior_dict['log10_halo_radius'])**2)
                M_inside = np.log10(M_inside)

                posterior_dict = {**posterior_dict,**{'log10_M5_mass':M_inside}}
                LaTex_dictionary = {**LaTex_dictionary,**{'log10_M5_mass':r'$log_{10}(M_{h,5})$'}}


        if drop_sys_rho == True:
            posterior_dict.pop('sys_rho')
            LaTex_dictionary.pop('sys_rho')
        if drop_halo == True:
            posterior_dict.pop('log10_halo_fraction')
            posterior_dict.pop('concentration')
            LaTex_dictionary.pop('log10_halo_fraction')
            LaTex_dictionary.pop('concentration')

        N_keys = len(posterior_dict.keys())
        
        posteriors = np.array(list(posterior_dict.values()))
        posteriors = posteriors.T

        #BF = np.percentile(posteriors,50,axis=0).ravel()
        #LE = BF-np.percentile(posteriors,10,axis=0).ravel()
        #UE = np.percentile(posteriors,90,axis=0).ravel()-BF
        #xticks   = [[BF[i]-LE[i],BF[i]+UE[i]] for i in range(N_keys)]

        min_v,max_v = np.min(posteriors,axis=0),np.max(posteriors,axis=0)
        l = max_v-min_v
        l *= 0.2
        xticks   = [[round(min_v[i]+l[i],4),round(max_v[i]-l[i],4)] for i in range(N_keys)]

        fig = corner.corner(posteriors)

        #pos = [(0.3,-1.5),(0.34,-1.5),(0.38,-1.5),(0.42,-1.5),(0.46,-1.5),(0.5,-1.5),(0.54,-1.5),(0.58,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5)]
        pos = [(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5),(0.5,-1.5)]
        pos2 = [(0,0),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3),(-0.35,0.3)]
        
        labs= [LaTex_dictionary[i] for i in posterior_dict.keys()]
        labs2 = labs[:] 
        labs2.insert(0,None)
        assi = fig.axes[-N_keys:]
        assiy= fig.axes[0:int(N_keys**2):N_keys]  

        for axy,ax,l,l2,p,p2,ticks in zip(assiy,assi,labs,labs2,pos,pos2,xticks):
            ax.set_xlabel(l,position=p,fontsize='small')
            axy.yaxis.set_major_locator(ticker.NullLocator())
            axy.set_ylabel(l2,position=p2,fontsize='small',rotation=45)
            ax.set_xticks(ticks)
            ax.xaxis.set_tick_params(labelsize='small')
            ax.ticklabel_format(useOffset=False, style='plain',axis='x')

        fig.set_figheight(20)
        fig.set_figwidth(25)
        plt.subplots_adjust(top=0.995,
                            bottom=0.094,
                            left=0.024,
                            right=0.995,
                            hspace=0.05,
                            wspace=0.05)
        
        return fig

    def iteration_plot(self,LaTex_dictionary,drop_sys_rho=False,drop_halo=False):
        LaTex_names = list(LaTex_dictionary.values())
        LaTex_dictionary = {**LaTex_dictionary,**{'logL':r'$log_{10}|logL|$'}}
        file_path  = self.output_path+'nested_samples.dat'
        f = open(file_path,'r')
        names = f.readline().split()[1:]
        f.close()
        samples = np.loadtxt(file_path)
        sample_dict = {names[i]:samples[:,i] for i in range(len(names)-1)}
        sample_dict['logL'] = np.log10(np.abs(sample_dict['logL']))
        
        posteriors = np.loadtxt(self.output_path+'posterior.dat')
        posterior_dict = {names[i]:posteriors[:,i] for i in range(len(names)-2)}

        log10M_halo = 0.
        if 'log10_disc2_mass' in posterior_dict:
            log10M_halo += 10**posterior_dict['log10_disc2_mass']
        if 'log10_disc1_mass' in posterior_dict:
            log10M_halo += 10**posterior_dict['log10_disc1_mass']
        if 'log10_bulge_mass' in posterior_dict:
            log10M_halo += 10**posterior_dict['log10_bulge_mass']
        log10M_halo = np.log10(10**posterior_dict['log10_halo_fraction']*log10M_halo)

        posterior_dict = {**posterior_dict,**{'log10_halo_mass':log10M_halo}}
        LaTex_dictionary = {**LaTex_dictionary,**{'log10_halo_mass':r'$log_{10}(M_{h})$'}}
        
        if self.halo_type == 'NFW':
            log10R_halo = (3*10**log10M_halo/(800.*np.pi*self.rho_crit))**(1/3)/posterior_dict['concentration']
            log10R_halo = np.log10(log10R_halo)
            posterior_dict = {**posterior_dict,**{'log10_halo_radius':log10R_halo}}
            LaTex_dictionary = {**LaTex_dictionary,**{'log10_halo_radius':r'$log_{10}(R_{h})$'}}
            if 'log10_disc2_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc2_radius']
            elif 'log10_disc1_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc1_radius']
            
            c = posterior_dict['concentration']
            s = r_trunc/10**log10R_halo
            M_inside = 10**log10M_halo/(np.log(1+c)-c/(1+c))*(np.log(1+s)-s/(1+s))
            M_inside = np.log10(M_inside)
            posterior_dict = {**posterior_dict,**{'log10_M5_mass':M_inside}}
            LaTex_dictionary = {**LaTex_dictionary,**{'log10_M5_mass':r'$log_{10}(M_{h,5})$'}}
        else:
            if 'log10_disc2_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc2_radius']
                M_inside = 10**posterior_dict['log10_halo_mass'] * (r_trunc**2/(r_trunc+10**posterior_dict['log10_halo_radius'])**2)
                M_inside = np.log10(M_inside)

                posterior_dict = {**posterior_dict,**{'log10_M5_mass':M_inside}}
                LaTex_dictionary = {**LaTex_dictionary,**{'log10_M5_mass':r'$log_{10}(M_{h,5})$'}}

            elif 'log10_disc1_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc1_radius']
                M_inside = 10**posterior_dict['log10_halo_mass'] * (r_trunc**2/(r_trunc+10**posterior_dict['log10_halo_radius'])**2)
                M_inside = np.log10(M_inside)

                posterior_dict = {**posterior_dict,**{'log10_M5_mass':M_inside}}
                LaTex_dictionary = {**LaTex_dictionary,**{'log10_M5_mass':r'$log_{10}(M_{h,5})$'}}


        #if 'log10_disc2_radius' in sample_dict:
        #    r_trunc = 5 * 10**sample_dict['log10_disc2_radius']
        #    M_inside = 10**sample_dict['log10_halo_mass'] * (r_trunc**2/(r_trunc+10**sample_dict['log10_halo_radius'])**2)
        #    M_inside = np.log10(M_inside)
#
        #    sample_dict = {**sample_dict,**{'log10_M5_mass':M_inside}}
        #    LaTex_dictionary = {**LaTex_dictionary,**{'log10_M5_mass':r'$log_{10}(M_{h,5})$'}}
        #
        #elif 'log10_disc1_radius' in sample_dict:
        #    r_trunc = 5 * 10**sample_dict['log10_disc1_radius']
        #    M_inside = 10**sample_dict['log10_halo_mass'] * (r_trunc**2/(r_trunc+10**sample_dict['log10_halo_radius'])**2)
        #    M_inside = np.log10(M_inside)
#
        #    sample_dict = {**sample_dict,**{'log10_M5_mass':M_inside}}
        #    LaTex_dictionary = {**LaTex_dictionary,**{'log10_M5_mass':r'$log_{10}(M_{h,5})$'}}
#
        #else:
        #    pass

        if drop_sys_rho == True:
            sample_dict.pop('sys_rho')
            LaTex_dictionary.pop('sys_rho')
        if drop_halo == True:
            sample_dict.pop('log10_halo_mass')
            sample_dict.pop('log10_halo_radius')
            LaTex_dictionary.pop('log10_halo_mass')
            LaTex_dictionary.pop('log10_halo_radius')

        N_keys = len(sample_dict.keys())
        
        iteration = np.arange(0,samples.shape[0])
        N_col,N_row = 4,math.ceil(N_keys/4)
        fig = plt.figure()
        for i in range(0,N_keys):
            ax = fig.add_subplot(N_row,N_col,i+1)
            ax.plot(iteration,sample_dict[list(sample_dict.keys())[i]],'.k',ms=0.7)
            ax.set_xlabel('iteration')
            ax.set_ylabel(LaTex_dictionary[list(sample_dict.keys())[i]])
        
        fig.set_figheight(20)
        fig.set_figwidth(25)
        plt.subplots_adjust(top=0.962,
                            bottom=0.057,
                            left=0.044,
                            right=0.983,
                            hspace=0.394,
                            wspace=0.365)

        return fig

    def best_params(self,percentile=[50,16,84],write=True):
        file_path  = self.output_path+'posterior.dat'
        f = open(file_path,'r')
        names = f.readline().split()[1:]
        f.close()
        posteriors = np.loadtxt(self.output_path+'posterior.dat')
        # Drop likelihood and prior
        posteriors,names = posteriors[:,:-2],names[:-2]

        posterior_dict = {names[i]:posteriors[:,i] for i in range(len(names))}
        
        log10M_halo = 0.
        if 'log10_disc2_mass' in posterior_dict:
            log10M_halo += 10**posterior_dict['log10_disc2_mass']
        if 'log10_disc1_mass' in posterior_dict:
            log10M_halo += 10**posterior_dict['log10_disc1_mass']
        if 'log10_bulge_mass' in posterior_dict:
            log10M_halo += 10**posterior_dict['log10_bulge_mass']
        log10M_halo = np.log10(10**posterior_dict['log10_halo_fraction']*log10M_halo)

        # Calcolo enclosed mass a seconda del profilo utilizzato, per NFW calcolo anche il raggio dell'halo
        if self.halo_type == 'NFW':
            log10R_halo = (3*10**log10M_halo/(800.*np.pi*self.rho_crit))**(1/3)/posterior_dict['concentration']
            log10R_halo = np.log10(log10R_halo)
            posterior_dict = {**posterior_dict,**{'log10_halo_radius':log10R_halo}}
            if 'log10_disc2_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc2_radius']
            elif 'log10_disc1_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc1_radius']
            
            c = posterior_dict['concentration']
            s = r_trunc/10**log10R_halo
            M_inside = 10**log10M_halo/(np.log(1+c)-c/(1+c))*(np.log(1+s)-s/(1+s))
            M_inside = np.log10(M_inside)
            posterior_dict = {**posterior_dict,**{'log10_M5_mass':M_inside}}
        else:
            if 'log10_disc2_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc2_radius']
                M_inside = 10**posterior_dict['log10_halo_mass'] * (r_trunc**2/(r_trunc+10**posterior_dict['log10_halo_radius'])**2)
                M_inside = np.log10(M_inside)
                posterior_dict = {**posterior_dict,**{'log10_M5_mass':M_inside}}
            elif 'log10_disc1_radius' in posterior_dict:
                r_trunc = 5 * 10**posterior_dict['log10_disc1_radius']
                M_inside = 10**posterior_dict['log10_halo_mass'] * (r_trunc**2/(r_trunc+10**posterior_dict['log10_halo_radius'])**2)
                M_inside = np.log10(M_inside)
                posterior_dict = {**posterior_dict,**{'log10_M5_mass':M_inside}}

        # Estimate best value and errors:
        names = list(posterior_dict.keys())
        posteriors = np.array(list(posterior_dict.values())).T
        Best_Fit = np.percentile(posteriors,percentile[0],axis=0).ravel()
        Lower_err = Best_Fit-np.percentile(posteriors,percentile[1],axis=0).ravel()
        Upper_err = np.percentile(posteriors,percentile[2],axis=0).ravel()-Best_Fit

        Best_Fit_dict_par = {names[i]:Best_Fit[i] for i in range(len(names))}
        Best_Fit_dict_LE = {names[i]:Lower_err[i] for i in range(len(names))}
        Best_Fit_dict_UE = {names[i]:Upper_err[i] for i in range(len(names))}

        if write==True:
            f = open(self.output_path+'Best_fit_params.txt','w')
            f.write('parameter name \t best fit \t Lower error \t Upper error\n')
            for i in range(len(names)):
                f.write(names[i]+'\t'+str(Best_Fit[i])+'\t'+str(Lower_err[i])+'\t'+str(Upper_err[i])+'\n')
            f.close()

        print('----------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------\n')
        for i in range(len(names)):
            print(names[i]+'\t'+'{:.3f}+{:.3f}-{:.3f}'.format(Best_Fit[i],Lower_err[i],Upper_err[i]))
        print('----------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------\n')
        return Best_Fit_dict_par,Best_Fit_dict_LE,Best_Fit_dict_UE



    def mass_profile(self,min_r=0.1,max_r=10,Nbins=20):
        # must be modified to include errors
        # radius must be in kpc
        Best_Fit,_,_ = self.best_params()  
        
        r = np.linspace(min_r,max_r,Nbins)

        
        if'log10_bulge_mass' in Best_Fit and \
           'log10_disc1_mass' in Best_Fit and \
           'log10_disc2_mass' in Best_Fit:

            halo_mass = 10**Best_Fit['log10_halo_fraction']*(10**Best_Fit['log10_bulge_mass']+\
                                                              10**Best_Fit["log10_disc1_mass"]+\
                                                              10**Best_Fit["log10_disc2_mass"])
            Mbulge = dehenen_mass(10**Best_Fit['log10_bulge_mass'],10**Best_Fit['log10_bulge_radius'],Best_Fit['gamma'],r)
            Mdisc1 = exp_disc_mass(10**Best_Fit["log10_disc1_mass"],10**Best_Fit["log10_disc1_radius"],r)
            Mdisc2 = exp_disc_mass(10**Best_Fit["log10_disc2_mass"],10**Best_Fit["log10_disc2_radius"],r)
            Mhalo  = NFW_mass(halo_mass,Best_Fit["concentration"],10**Best_Fit['log10_halo_radius'],r)
        
        elif   'log10_disc1_mass' in Best_Fit and \
                'log10_disc2_mass' in Best_Fit:

            halo_mass = 10**Best_Fit['log10_halo_fraction']*(10**Best_Fit["log10_disc1_mass"]+\
                                                              10**Best_Fit["log10_disc2_mass"])
            Mbulge = np.zeros_like(r)
            Mdisc1 = exp_disc_mass(10**Best_Fit["log10_disc1_mass"],10**Best_Fit["log10_disc1_radius"],r)
            Mdisc2 = exp_disc_mass(10**Best_Fit["log10_disc2_mass"],10**Best_Fit["log10_disc2_radius"],r)
            Mhalo  = NFW_mass(halo_mass,Best_Fit["concentration"],10**Best_Fit['log10_halo_radius'],r)

        elif'log10_bulge_mass' in Best_Fit and \
           'log10_disc1_mass' in Best_Fit:

            halo_mass = 10**Best_Fit['log10_halo_fraction']*(10**Best_Fit['log10_bulge_mass']+\
                                                              10**Best_Fit["log10_disc1_mass"])
            Mbulge = dehenen_mass(10**Best_Fit['log10_bulge_mass'],10**Best_Fit['log10_bulge_radius'],Best_Fit['gamma'],r)
            Mdisc1 = exp_disc_mass(10**Best_Fit["log10_disc1_mass"],10**Best_Fit["log10_disc1_radius"],r)
            Mhalo  = NFW_mass(halo_mass,Best_Fit["concentration"],10**Best_Fit['log10_halo_radius'],r)
            Mdisc2 = np.zeros_like(r)
            
        return r,Mbulge,Mdisc1,Mdisc2,Mhalo



    def radial_profile(self,x,y,z,quantity=None,estimator='mean',min_r=0.1,max_r=10,Nbins=20,azimuth_division=True,params=False):
        '''
        CPU FUNCTION
        ----------------------------------------------------------------------
        ----------------------------------------------------------------------
        Radial profiles (azimuthally averaged). 
            Parameters:
                    x(1D float32 arr)  : x coordinate of the grid 
                    y(1D float32 arr)  : y coordinate of the grid 
                    z(1D float32 arr)  : z quantity to be averaged 
                    quantity(str)      : if "velocity" then z is divided by sin(incl)cos(azimuth) if azimuth_division=True
                                         otherwise z is taken in absolute value. Default is azimuth_division.
                    estimator(str)     : whether to use mean of median to do the average.
                    min_r(float 32)    : minimum radius for binning
                    max_r(float 32)    : maximum radius for binning
                    Nbins(int 16)      : Number of bins
                    azimuth_division   : see parameter "variable". Default is true

            Returns: 
                    r_avg(1D float32 arr): radius averaged in bins
                    z_avg(1D float32 arr): the quantity to be averaged (brightness, velocity, dispersion, mass to light) 
        '''
        if params == False:
            Best_Fit,_,_ = self.best_params()
        else:
            Best_Fit = params

        x0,y0 = Best_Fit['x0'],Best_Fit['y0']
        theta,incl = Best_Fit['orientation'],Best_Fit['inclination']

        x1 = x-x0
        y1 = y-y0
        xr = x1*np.cos(theta) + y1*np.sin(theta)
        yr = y1*np.cos(theta) - x1*np.sin(theta)
        yd = yr/np.cos(incl)
        r = np.sqrt(np.square(xr)+np.square(yd))

        r_bin = np.linspace(min_r,max_r,Nbins)

        z_avg = np.zeros(Nbins-1)
        r_avg = np.zeros(Nbins-1)
        if quantity=='velocity':
            if azimuth_division == True:
                phi = np.arctan2(yd,xr)
                z = z/(np.sin(incl)*np.cos(phi))
            else:
                z = np.abs(z)
            for i in range(np.size(r_bin)-1):
                a = r < r_bin[i+1]
                b = r >= r_bin[i]

                z_tmp = z[a*b]
                if np.size(z_tmp[np.isnan(z_tmp)]) > np.size(z_tmp)/4:
                    print('warning: some of your bins have more then 25 percent of NaN \n \
                           you may want to improve your binning')
                if estimator == 'mean':
                    z_avg[i] = np.nanmean(z_tmp)
                    r_avg[i] = np.nanmean(r[a*b])
                elif estimator == 'median':
                    z_avg[i] = np.nanmedian(z_tmp)
                    r_avg[i] = np.nanmedian(r[a*b])

        else:
            for i in range(np.size(r_bin)-1):
                a = r < r_bin[i+1]
                b = r >= r_bin[i]
                z_tmp = z[a*b]
                if np.size(z_tmp[np.isnan(z_tmp)]) > np.size(z_tmp)/4:
                    print('warning: some of your bins have more then 25 percent of NaN \n \
                           you may want to improve your binning')
                if estimator == 'mean':
                    z_avg[i] = np.nanmean(z_tmp)
                    r_avg[i] = np.nanmean(r[a*b])
                elif estimator == 'median':
                    z_avg[i] = np.nanmedian(z_tmp)
                    r_avg[i] = np.nanmedian(r[a*b])

        return r_avg,z_avg

