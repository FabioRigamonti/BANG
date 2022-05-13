import numpy as np 
import data as d
import matplotlib.pyplot as plt
from scipy.special import i0e,i1e,k0e,k1e
import torch

'''this is in Km^2 Kpc / (M_sun s^2).'''
G = 4.299e-6
def read_user(dataset_path):
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

    return x_arcsec,y_arcsec,x_kpc,y_kpc,ibrightness_data,v_data,sigma_data,ML_data,ibrightness_error,v_error,sigma_error,ML_error,goodpix 

#---------------------------------------------------------------------------
                    #Generic
#---------------------------------------------------------------------------

def PSF(r,            #raggio proiettato
        sigma=0.27046 #sigma PSF per califa FWHM 2.5''
        ):
        r2 = np.square(r)
        sigma2 = np.square(sigma)
        
        psf = np.exp(-r2/(2*sigma2)) 
        return psf


#---------------------------------------------------------------------------
                    #HERQUINST
#---------------------------------------------------------------------------


def herquinst(M,Rb,R,N,J,K):
    '''function that return density and
    isotropic velocity dispersion. Ottimizzata:
    calcolo tutti insieme'''
    #center of mass traslation
	
    Rb2 = Rb*Rb
    Rb3 = Rb*Rb2 
    pi2 = np.pi*np.pi
    pi3 = pi2*np.pi


    s = R / Rb

    #definition of variables
    X = np.zeros((N,J,K))
    rho = np.zeros((N,J,K))
    sigma = np.zeros((N,J,K))

    #X
    a = s < 1
    b = s == 1
    c = s > 1
    #s < 1
    sa = s[a]
    sa2 = np.square(sa)
    sqrt_one_minus_sa2 = np.sqrt(1-sa2)
    X[a] = np.log((1 + sqrt_one_minus_sa2) / sa) / (sqrt_one_minus_sa2)
    #s = 1
    X[b] = 1.
    #s > 1
    sc = s[c]
    sc2 = np.square(sc)
    sqrt_sc2_minus_one = np.sqrt(sc2-1)
    X[c] = np.arccos(1/s[c]) / (sqrt_sc2_minus_one)
    
    #rho 
    d = (s >= 0.7) * (s <= 1.3)
    #e = s!=1
    e = (s <= 0.7) + (s >= 1.3)
    se = s[e]
    se2 = np.square(se)
    square_one_minus_se2 = np.square(1-se2)
    sd = s[d]
    sd_minus_one = sd-1.
    square_sd_minus_one = np.square(sd_minus_one)
    cube_sd_minus_one = sd_minus_one*np.square(sd_minus_one)
    fourth_sd_minus_one = np.square(square_sd_minus_one)
    fifth_sd_minus_one = square_sd_minus_one * cube_sd_minus_one

    A = M / (2 * np.pi * Rb2 * square_one_minus_se2)
    B = (2 + se2) * X[e] - 3
    rho[e] = A * B
    rho[d] = M / (2*np.pi*Rb2) *( 4./15. - 16.*sd_minus_one/35. + (8.* square_sd_minus_one)/15 - (368.* cube_sd_minus_one)/693. + (1468.* fourth_sd_minus_one)/3003. - (928.* fifth_sd_minus_one)/2145.)

    #calcolo qui la sigma nei punti d perche gia ho s-1
    sigma[d] = (G*M/Rb) * ( (332.0-105.0*np.pi)/28.0 + (18784.0-5985.0*np.pi)*sd_minus_one/588.0 + (707800.0-225225.0*np.pi)*square_sd_minus_one/22638.0 )

    #sigma
    f = (s < 15.)*(s > 1.3) + (s <= 0.7)
    g = s >= 15.

    sf = s[f]
    sf2 = np.square(sf)
    sf4 = np.square(sf2)
    sf6 = sf4*sf2                               #np.power(sf,6) is slower
    one_minus_sf2 = 1 - sf2
    square_one_minus_sf2 = np.square(one_minus_sf2)
    A = (G * M*M ) / (12 * np.pi * Rb3 * rho[f])
    B = 1 /  (2*one_minus_sf2 * square_one_minus_sf2)
    C = (-3 * sf2) * X[f] * (8*sf6 - 28*sf4 + 35*sf2 - 20) - 24*sf6 + 68*sf4 - 65*sf2 + 6
    D = 6*np.pi*sf
    sigma[f] = A * (B*C - D)

    # s > 6
    sg = s[g]
    sg2 = np.square(sg)
    sg3 = sg*sg2
    A = (G * M * 8) / (15 * s[g] * np.pi * Rb)
    B = (8/np.pi - 75*np.pi/64) / sg
    C = (64/pi2 - 297/56) / sg2
    D = (512/pi3 - 1199/(21*np.pi) - 75*np.pi/512) / sg3
    sigma[g] = A * (1 + B + C + D)

    return(rho,sigma)

#
def v_H (M,a,r):
    '''circular velocity squared
    for an herquinst model.
    ''' 
    #this was to check only herquinst
    if (M == 0.) or (a == 0.):
        return 0*r

    #circular velocity squared
    v_c2 = (G * M * r) / np.square(r + a)

    return v_c2


#---------------------------------------------------------------------------
                    #EXP DISC
#---------------------------------------------------------------------------


def rho_D(Md,R_d,r):
    '''surface density for a thin disc with no
    thickness (no z coord). So I don't have any 
    integration on the LOS, (È GIUSTO STO RAGIONAMENTO?)
    but only rotation,traslation and deprojection of the
    coordinates.'''
    #this was to check only herquinst
    
    if (Md == 0.) or (R_d == 0.):
        return r*0
    

    rho = (Md / (2 * np.pi * R_d*R_d) ) * np.exp(-r/R_d)
    rho[rho<1e-10] = 1e-10
    return(rho)


def v_D(M,R_d,r):
    '''circular velocity for a thin disc with no
    thickness (no z coord). So I don't have any 
    integration on the LOS, (È GIUSTO STO RAGIONAMENTO?)
    but only rotation,traslation and deprojection of the
    coordinates.'''
    
    
    r2 = np.square(r)
    R_d3 = R_d*R_d*R_d
    s = r/(2*R_d)
    v_c2 =  ((G * M * r2) / (2 * R_d3)) * (i0e(s) * k0e(s) - i1e(s) * k1e(s)) 
    return(v_c2)

    

#---------------------------------------------------------------------------
                    #TOTATL QUANTITIES
#---------------------------------------------------------------------------

def rhosigma_tot(Mb,Rb,Md1,Rd1,Md2,Rd2,cos_i,R,r,N,J,K):
    
    rhoH,sigma = herquinst(Mb,Rb,R,N,J,K)

    rhoD1 = rho_D(Md1,Rd1,r) /cos_i
    rhoD2 = rho_D(Md2,Rd2,r) /cos_i

    return (rhoH,rhoD1,rhoD2,sigma)



def v_tot(Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh,sin_i,r,phi):
    v_D1_tmp =  v_D(Md1,Rd1,r)
    v_D2_tmp =  v_D(Md2,Rd2,r)
    v_B_tmp  = v_H(Mb,Rb,r)
    v_H_tmp  = v_H(Mh,Rh,r)
    vtot2 = v_D1_tmp + v_D2_tmp + v_B_tmp + v_H_tmp
    
    vtot = np.sqrt(vtot2)*sin_i*np.cos(phi)
    return vtot,vtot2



def AVG(peso1,peso2,peso3,mu1,mu2,mu3):

    tmp = (peso1*mu1 + peso2*mu2 + peso3*mu3)/(peso1+peso2+peso3)

    return tmp


#---------------------------------------------------------------------------
                    #MODEL
#---------------------------------------------------------------------------

def model_BDD(x,y,N,J,K,psf,Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh,xcm,ycm,theta,incl,LM_b,LM_d1,LM_d2,k1,k2):
    '''
    CPU/GPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute the model assuming bulge+disc1+disc2+halo. This can be used alone or in combination with
    the function 'super_resolution'.

        Parameters:
                x,y (tensor float32): x,y positions of the low/high resolution grid (i.e. ~ N/4 J/4 K or N J K)
                psf (tensor float32): weight of the psf in each subgrid pixel (same dimension as x and y)
                Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh (float32): masses and scale radii of the componenst
                xcm,ycm,theta,incl (float32): intrinsic parameters of the model
                LM_b,LM_d1,LM_d2 (float32): light to mass ratio (mass to light)^-1
                k1,k2 (float32): kinematical decomposition parameters 
        Returns: 
                out(tensor float32): output tensor of the form 4 X 1 X N X J (or 4 X 1 X N/4 X J/4)
                        - out[0,0,:,:] = log brightness
                        - out[1,0,:,:] = line of sight velocity
                        - out[2,0,:,:] = line of sight velocity dispersion
                        - out[3,0,:,:] = light to mass ratio (mass to light ratio)^-1 
    '''

    Mb  = 10**Mb
    Md1 = 10**Md1
    Md2 = 10**Md2
    Mh  = 10**Mh
    Rb  = 10**Rb
    Rd1 = 10**Rd1
    Rd2 = 10**Rd2
    Rh  = 10**Rh


    x0 = x-xcm
    y0 = y-ycm
    c_theta,s_theta,c_incl,s_incl = np.cos(theta),np.sin(theta),np.cos(incl),np.sin(incl)
    xr = x0*c_theta + y0*s_theta
    yr = y0*c_theta - x0*s_theta
    yd = yr/c_incl
    phi = np.arctan2(yd,xr)
    r = np.sqrt(np.square(xr)+np.square(yd))
    R = np.sqrt(np.square(x0)+np.square(y0))
    rhoH,rhoD1,rhoD2,sigmaH2 = rhosigma_tot(Mb,Rb,Md1,Rd1,Md2,Rd2,c_incl,R,r,N,J,K)
    vtmp,vtmp2 =  v_tot(Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh,s_incl,r,phi)
    B_H,B_D1,B_D2 = rhoH*LM_b, rhoD1*LM_d1, rhoD2*LM_d2
    B_tot,rho_tot = (B_H+B_D1+B_D2),(rhoH+rhoD1+rhoD2)
    psf_sum = np.sum(psf,axis=2)
    B = np.sum((psf*B_tot),axis=2)/psf_sum
    B_H_avg = np.sum(psf*B_H,axis=2)/psf_sum
    B_D1_avg = np.sum(psf*B_D1,axis=2)/psf_sum
    B_D2_avg = np.sum(psf*B_D2,axis=2)/psf_sum
    v_D1 = np.sum(psf*B_D1*vtmp,axis=2)/np.sum(psf*B_D1,axis=2)
    v_D2 = np.sum(psf*B_D2*vtmp,axis=2)/np.sum(psf*B_D2,axis=2)
    v_D1_2 = np.sum(psf*B_D1*vtmp2,axis=2)/np.sum(psf*B_D1,axis=2)
    v_D2_2 = np.sum(psf*B_D2*vtmp2,axis=2)/np.sum(psf*B_D2,axis=2)
    sigmaH2_avg = np.sum(psf*sigmaH2*B_H,axis=2)/np.sum(psf*B_H,axis=2)
    v_los = AVG(B_H_avg,B_D1_avg,B_D2_avg,0.,k1*v_D1,k2*v_D2)
    sigma_los = np.sqrt(AVG(B_H_avg,B_D1_avg,B_D2_avg,sigmaH2_avg,(1.-k1*k1)*v_D1_2/3.0,(1.-k2*k2)*v_D2_2/3.0))
    LM_avg = np.sum(psf*(B_tot),axis=2)/np.sum(psf*rho_tot,axis=2)
    out = np.zeros((4,1,N,J))
    out[0,0,:,:] = np.log10(B)
    out[1,0,:,:] = v_los
    out[2,0,:,:] = sigma_los
    out[3,0,:,:] = LM_avg
    return out,B_tot
 


def super_resolution(model,lr):
    '''
    CPU/GPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Given the output model_BDD it gives the super resolution model 4 times more resolved on both axis

        Parameters:
                model with the weights already trained of the network
                lr (float32 tensor): output of model_BDD
        Returns: 
                hr(tensor float32): super resolution tensor of the form 4 X 1 X N X J 
                        - hr[0,0,:,:] = log brightness
                        - hr[1,0,:,:] = line of sight velocity
                        - hr[2,0,:,:] = line of sight velocity dispersion
                        - hr[3,0,:,:] = light to mass ratio (mass to light ratio)^-1 
    '''
    # first normalization of lr_img to [0,1] interval

    min_lr,max_lr = torch.min(torch.min(lr,dim=2).values,dim=2).values,torch.max(torch.max(lr,dim=2).values,dim=2).values
    
    min_lr,max_lr = min_lr.reshape((4,1,1,1)),max_lr.reshape((4,1,1,1))

    min_hr,max_hr = min_lr-torch.abs(0.01*min_lr),max_lr+torch.abs(0.01*max_lr)
    
    lr = (lr-min_lr)/(max_lr-min_lr)

    with torch.no_grad():
        hr = model(lr)
    
    hr = hr * (max_hr-min_hr) + min_hr

    return hr
#---------------------------------------------------------------------------
                    #FIT CM
#---------------------------------------------------------------------------

def log_likelihood(x,y,psf,
                   Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh,
                   xcm,ycm,theta,incl,
                   LM_b,LM_d1,LM_d2,
                   k1,k2,
                   sys_rho,                  
                   data,                   #list of the data different combination
                   error,               #list of the errors
                   good_pix,
                   model,
                   X_device,rho_device,sigma_device,ans_i0_dev,ans_k0_dev,ans_i1_dev,ans_k1_dev,out):
    '''
    CPU/GPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Log likelihood evaluation using model from model_BDD+super_resolution

        Parameters:
                x,y (tensor float32): x,y positions of the low resolution grid (i.e. ~ N/4 J/4 K)
                Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh (float32): masses and scale radii of the componenst
                xcm,ycm,theta,incl (float32): intrinsic parameters of the model
                LM_b,LM_d1,LM_d2 (float32): light to mass ratio (mass to light)^-1
                k1,k2 (float32): kinematical decomposition parameters 
                data (tensor float32): tensor of the form 4 X 1 X N X J containing the data
                                - data[0,0,:,:] is log_brightness
                                - data[1,0,:,:] is v_los
                                - data[2,0,:,:] is sigma_los
                                - data[3,0,:,:] is M/L ratio
                data_err (tensor float32): same of data but containing the associated errors
                good_pix (tensor bool): tensor of the same form of data that is true where we want to evaluate the data
                model : trained weights on the neural network for super resolution
        Returns: 
                logL(float32): log likelihood 
    '''
    mod_tmp = model_BDD(x,y,psf,Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh,xcm,ycm,theta,incl,LM_b,LM_d1,LM_d2,k1,k2,\
                        X_device,rho_device,sigma_device,ans_i0_dev,ans_k0_dev,ans_i1_dev,ans_k1_dev,out)
    mod = super_resolution(model,mod_tmp)
    #mod = mod.cpu().detach().numpy()
    #mod = model_BDD(x,y,N,J,K,psf,Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh,xcm,ycm,theta,incl,LM_b,LM_d1,LM_d2,k1,k2)
    mod[3,0,:,:] = 1.0/mod[3,0,:,:]
    
    #mod_B = mod[0,0,good_pix[0,0]]
    #data_B = data[0,0,good_pix[0,0]]
    #data_err_B = data_err[0,0,good_pix[0,0]]
    #
    #mod_v = mod[1,0,good_pix[1,0]]
    #data_v = data[1,0,good_pix[1,0]]
    #data_err_v = data_err[1,0,good_pix[1,0]]
#
    #mod_sigma = mod[2,0,good_pix[2,0]]
    #data_sigma = data[2,0,good_pix[2,0]]
    #data_err_sigma = data_err[2,0,good_pix[2,0]]
#
    #mod_ML = mod[3,0,good_pix[3,0]]
    #data_ML = data[3,0,good_pix[3,0]]
    #data_err_ML = data_err[3,0,good_pix[3,0]]
#
    #data_err2_B = torch.square(data_err_B)
    #data_err2_v = torch.square(data_err_v)
    #data_err2_sigma = torch.square(data_err_sigma)
    #data_err2_ML = torch.square(data_err_ML)

    mod = torch.ravel(mod)
    #mod = np.ravel(mod)
    mod = mod[good_pix]
    data = data[good_pix]
    error = error[good_pix]
    error2 = error*error

    logL = -0.5 * ( torch.square(data - mod)/error2  + torch.log(2*np.pi*error2) )
    #logL = -0.5 * ( np.square(data - mod)/error2  + np.log(2*np.pi*error2) )

    a = torch.sum(logL)
    #a = np.sum(logL)

    return (a.cpu().item()) 


def lr_grid(x_hr,y_hr,N,J):
    # this will go in data.py
    xmin,xmax = np.min(x_hr),np.max(x_hr)
    ymin,ymax = np.min(y_hr),np.max(y_hr)

    new_N,new_J = int(N/4)+1,int(J/4)+1

    x = np.linspace(xmin,xmax,new_N)
    y = np.linspace(ymin,ymax,new_J)

    x,y = np.meshgrid(x,y,indexing='ij')

    return x.ravel(),y.ravel()

#---------------------------------------------------------------------------
                        #MAIN
#---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    x_arcsec,y_arcsec,x,y,ibrightness_data,v_data,sigma_data,ML_data,\
    ibrightness_error,v_error,sigma_error,ML_error,goodpix = read_user('../dataset.npy')
    
    dx = abs(x[1]-x[0])

    degree = 30
    K = int(degree*degree)
    J = np.size(x[ x == x[0] ])       
    N = np.size(y[ y == y[0] ])   
    npsf = 3   
    sigmapsf =  1.374

    range_ext = 10  

    # grid creation with refinement 
    x_true = x[0:N]  #x[0 : J*N : J]
    y_true = y[0: J*N :J]

    dx = abs(x_true[1]-x_true[0])
    dy = abs(y_true[1]-y_true[0])

    sample_point , weights = np.polynomial.hermite.hermgauss(deg=degree)

    sample_point = sample_point*2**0.5*sigmapsf

    x_subgrid,y_subgrid = np.meshgrid(sample_point,sample_point,indexing='ij')
    
    weight_matrix = np.zeros((degree,degree))

    for i in range(degree):
        for j in range(degree):
            weight_matrix[i,j] = weights[i]*weights[j]            

    x_tmp_g,y_tmp_g = x_subgrid.ravel(),y_subgrid.ravel()
    
    x_rand = np.zeros((x.shape[0],int(degree*degree)))
    y_rand = np.zeros((x.shape[0],int(degree*degree)))
    
    for i in range(x.shape[0]):
        x_rand[i,:] = x_tmp_g+x[i]
        y_rand[i,:] = y_tmp_g+y[i]

    x_rand,y_rand = x_rand.ravel(),y_rand.ravel()
    psf = weight_matrix.ravel()

    
    data_obj = d.data(x.ravel(),y.ravel())
    psf_all = np.zeros((N,J,K))

    for i in range(N):
        for j in range(J):
            psf_all[i,j,:] = psf[:]
    

    x,y = x.reshape(N,J),y.reshape(N,J)
    x_g,y_g = x_rand.reshape((N,J,K)),y_rand.reshape((N,J,K))

    #Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh = 9.388,-1.998,-np.inf,0.,-np.inf,0.7,-np.inf,1
    #xcm,ycm,theta,incl = 0.,0.,np.pi/3,np.pi/4
    #k1,k2,LM_b,LM_d1,LM_d2 = 0.3,0.7,1./0.270,0.8,0.4

    Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh = 9.388,-1.998,10.525,-0.406,9.643,0.837,11.702,1.064
    x0,y0,theta,incl = -0.039,-0.005,2.691,1.116
    k1,k2,ML_b,ML_d1,ML_d2 = 0.3,0.7,0.270,2.000,0.101

    Rb = 0.2
    Rd1 = 0.4

    #Rb = 0.
    #theta = 0. 
    #incl = 0.
    #Rb = 0.5
    out,btot = model_BDD(x_g,
                    y_g,
                    N,
                    J,
                    K,
                    psf_all,
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
                    k2)

    B,v_los,sigma_los,LM = out[0,0,:,:],out[1,0,:,:],out[2,0,:,:],out[3,0,:,:]

    fig,ax,pmesh,cbar = data_obj.brightness_map(B)
    ax.plot(np.linspace(-19.1,-19.1+npsf*sigmapsf,100),np.ones(100)*(-20),'k',lw=3)
    ax.plot(np.linspace(-19.1,-19.1+dx,100),np.ones(100)*(-19.5),'k',lw=3)
    fig.show()


    fig = plt.figure()
    ax  = fig.add_subplot()
    ax.plot(x[32,:],B[32,:],'.b')
    ax.plot(y[:,32],B[:,32],'.r')
    fig.show()

    #plt.scatter(x_g.ravel(), y_g.ravel(),c=np.log10(btot.ravel()),cmap='viridis')
    #plt.plot(np.linspace(-19.1,npsf*sigmapsf,100),np.ones(100)*(-20),'k',lw=3)
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    vmin,vmax = np.min(np.log10(btot)),np.max(np.log10(btot))

    ax.scatter(x_g[0,0,:], y_g[0,0,:],c=np.log10(btot[0,0,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-1,0,:], y_g[-1,0,:],c=np.log10(btot[-1,0,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[0,-1,:], y_g[0,-1,:],c=np.log10(btot[0,-1,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-1,-1,:], y_g[-1,-1,:],c=np.log10(btot[-1,-1,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    #ax.show()

    ax.scatter(x_g[10,10,:], y_g[10,10,:],c=np.log10(btot[10,10,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-10,10,:], y_g[-10,10,:],c=np.log10(btot[-10,10,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[10,-10,:], y_g[10,-10,:],c=np.log10(btot[10,-10,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-10,-10,:], y_g[-10,-10,:],c=np.log10(btot[-10,-10,:]),cmap='viridis',vmin=vmin,vmax=vmax)

    ax.scatter(x_g[20,20,:], y_g[20,20,:],c=np.log10(btot[20,20,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-20,20,:], y_g[-20,20,:],c=np.log10(btot[-20,20,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[20,-20,:], y_g[20,-20,:],c=np.log10(btot[20,-20,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-20,-20,:], y_g[-20,-20,:],c=np.log10(btot[-20,-20,:]),cmap='viridis',vmin=vmin,vmax=vmax)

    ax.scatter(x_g[30,30,:], y_g[30,30,:],c=np.log10(btot[30,30,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-30,30,:], y_g[-30,30,:],c=np.log10(btot[-30,30,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[30,-30,:], y_g[30,-30,:],c=np.log10(btot[30,-30,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-30,-30,:], y_g[-30,-30,:],c=np.log10(btot[-30,-30,:]),cmap='viridis',vmin=vmin,vmax=vmax)

    a = ax.scatter(x_g[32,0,:], y_g[32,0,:],c=np.log10(btot[32,0,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    #ax.scatter(x_g[-10,34,:], y_g[-10,34,:],c=np.log10(btot[-10,34,:]),cmap='viridis')
    #ax.scatter(x_g[20,-34,:], y_g[20,-34,:],c=np.log10(btot[20,-34,:]),cmap='viridis')
    #ax.scatter(x_g[-30,-34,:], y_g[-30,-34,:],c=np.log10(btot[-30,-34,:]),cmap='viridis')
    plt.colorbar(a)

    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(x_g[0,0,:], y_g[0,0,:],c=psf_all[0,0,:],cmap='viridis')
    ax.scatter(x_g[-1,0,:], y_g[-1,0,:],c=psf_all[-1,0,:],cmap='viridis')
    ax.scatter(x_g[0,-1,:], y_g[0,-1,:],c=psf_all[0,-1,:],cmap='viridis')
    ax.scatter(x_g[-1,-1,:], y_g[-1,-1,:],c=psf_all[-1,-1,:],cmap='viridis')

    ax.scatter(x_g[10,10,:], y_g[10,10,:],c=psf_all[10,10,:],cmap='viridis')
    ax.scatter(x_g[-10,10,:], y_g[-10,10,:],c=psf_all[-10,10,:],cmap='viridis')
    ax.scatter(x_g[10,-10,:], y_g[10,-10,:],c=psf_all[10,-10,:],cmap='viridis')
    ax.scatter(x_g[-10,-10,:], y_g[-10,-10,:],c=psf_all[-10,-10,:],cmap='viridis')

    ax.scatter(x_g[20,20,:], y_g[20,20,:],c=psf_all[20,20,:],cmap='viridis')
    ax.scatter(x_g[-20,20,:], y_g[-20,20,:],c=psf_all[-20,20,:],cmap='viridis')
    ax.scatter(x_g[20,-20,:], y_g[20,-20,:],c=psf_all[20,-20,:],cmap='viridis')
    ax.scatter(x_g[-20,-20,:], y_g[-20,-20,:],c=psf_all[-20,-20,:],cmap='viridis')

    ax.scatter(x_g[30,30,:], y_g[30,30,:],c=psf_all[30,30,:],cmap='viridis')
    ax.scatter(x_g[-30,30,:], y_g[-30,30,:],c=psf_all[-30,30,:],cmap='viridis')
    ax.scatter(x_g[30,-30,:], y_g[30,-30,:],c=psf_all[30,-30,:],cmap='viridis')
    ax.scatter(x_g[-30,-30,:], y_g[-30,-30,:],c=psf_all[-30,-30,:],cmap='viridis')

    a = ax.scatter(x_g[32,0,:], y_g[32,0,:],c=psf_all[32,0,:],cmap='viridis')
    #ax.scatter(x_g[-10,34,:], y_g[-10,34,:],c=psf_all[-10,34,:],cmap='viridis')
    #ax.scatter(x_g[20,-34,:], y_g[20,-34,:],c=psf_all[20,-34,:],cmap='viridis')
    #ax.scatter(x_g[-30,-34,:], y_g[-30,-34,:],c=psf_all[-30,-34,:],cmap='viridis')
    plt.colorbar(a)
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot()

    vmin,vmax = np.min(np.log10(btot*psf_all)),np.max(np.log10(btot*psf_all))

    ax.scatter(x_g[0,0,:], y_g[0,0,:],c=np.log10(btot[0,0,:]*psf_all[0,0,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-1,0,:], y_g[-1,0,:],c=np.log10(btot[-1,0,:]*psf_all[-1,0,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[0,-1,:], y_g[0,-1,:],c=np.log10(btot[0,-1,:]*psf_all[0,-1,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-1,-1,:], y_g[-1,-1,:],c=np.log10(btot[-1,-1,:]*psf_all[-1,-1,:]),cmap='viridis',vmin=vmin,vmax=vmax)

    ax.scatter(x_g[10,10,:], y_g[10,10,:],c=np.log10(btot[10,10,:]*psf_all[10,10,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-10,10,:], y_g[-10,10,:],c=np.log10(btot[-10,10,:]*psf_all[-10,10,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[10,-10,:], y_g[10,-10,:],c=np.log10(btot[10,-10,:]*psf_all[10,-10,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-10,-10,:], y_g[-10,-10,:],c=np.log10(btot[-10,-10,:]*psf_all[-10,-10,:]),cmap='viridis',vmin=vmin,vmax=vmax)

    ax.scatter(x_g[20,20,:], y_g[20,20,:],c=np.log10(btot[20,20,:]*psf_all[20,20,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-20,20,:], y_g[-20,20,:],c=np.log10(btot[-20,20,:]*psf_all[-20,20,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[20,-20,:], y_g[20,-20,:],c=np.log10(btot[20,-20,:]*psf_all[20,-20,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-20,-20,:], y_g[-20,-20,:],c=np.log10(btot[-20,-20,:]*psf_all[-20,-20,:]),cmap='viridis',vmin=vmin,vmax=vmax)

    ax.scatter(x_g[30,30,:], y_g[30,30,:],c=np.log10(btot[30,30,:]*psf_all[30,30,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-30,30,:], y_g[-30,30,:],c=np.log10(btot[-30,30,:]*psf_all[-30,30,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[30,-30,:], y_g[30,-30,:],c=np.log10(btot[30,-30,:]*psf_all[30,-30,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[-30,-30,:], y_g[-30,-30,:],c=np.log10(btot[-30,-30,:]*psf_all[-30,-30,:]),cmap='viridis',vmin=vmin,vmax=vmax)

    ax.scatter(x_g[32,0,:], y_g[32,0,:],c=np.log10(btot[32,0,:]*psf_all[32,0,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(x_g[32,20,:], y_g[32,20,:],c=np.log10(btot[32,20,:]*psf_all[32,20,:]),cmap='viridis',vmin=vmin,vmax=vmax)
    #ax.scatter(x_g[-10,34,:], y_g[-10,34,:],c=psf_all[-10,34,:],cmap='viridis')
    #ax.scatter(x_g[20,-34,:], y_g[20,-34,:],c=psf_all[20,-34,:],cmap='viridis')
    #ax.scatter(x_g[-30,-34,:], y_g[-30,-34,:],c=psf_all[-30,-34,:],cmap='viridis')

    fig.show()

    plt.show()