from numba import cuda,float64,float32,int16
import numpy as np 
from math import isnan
from math import pi as M_PI
from math import acos,atan2,cos,sin,exp,log,log10,sqrt


'''Units are:
            - M_sun/kpc^2 for surface densities
            - km/s for velocities and dispersions
G = 4.299e-6
'''

#---------------------------------------------------------------------------
                    #GENERAL FUNCTIONS
#---------------------------------------------------------------------------
@cuda.jit('float32(float32,float32)',
          device=True,
          inline=True,
          fastmath=True)        # it is OK to call it, but it has no effect, it depends only on kernel fastmath Trueon
def radius(x,y)  :
    '''
    DEVICE FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Return radius in polar coordinate given cartesian coordinate.

        Parameters:
                x(float32): x cartesian coordinate
                y(float32): y cartesian coordinate

        Returns:
                out(float32): polar radius 
    '''
    return sqrt(x*x+y*y)


@cuda.jit('float32(float32)',
          device=True,
          #debug=True,
          fastmath=True)
def my_i0e(x):
    '''
    DEVICE FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    0th order modified Bessel function of the first kind exponentially scaled.
    For asymptotic expansion see Abramowitz and Stegun.

        Parameters:
                x(float32): float number 

        Returns: 
                ans(float32): 0th order modified Bessel function of the first kind exponentially scaled 
    '''
    if ( x < 3.75): 
        y=x/3.75
        y=y*y
        ans=exp(-x)*(1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2))))))
    else: 
        y=3.75/x
        ans=(1/sqrt(x))*(0.39894228+y*(0.1328592e-1+y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2+y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1+y*0.392377e-2))))))))
        
    return ans



@cuda.jit('float32(float32,float32)',
          device=True,
          fastmath=True)
def my_k0e(x,i0e):
    '''
    DEVICE FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    0th order modified Bessel function of the second kind exponentially scaled.
    For asymptotic expansion see Abramowitz and Stegun. 

        Parameters:
                x(float32)   : float number 
                i0e(float32) : output of my_i0e

        Returns: 
                ans(float32): 0th order modified Bessel function of the second kind exponentially scaled 
    '''
    if (x <= 2.0) :
        y=x*x/4.0
        ans=exp(x)*((-log(x/2.0)*exp(x)*my_i0e(x))+(-0.57721566+y*(0.42278420+y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2+y*(0.10750e-3+y*0.74e-5)))))))
    else:
        y=2.0/x
        ans=(1./sqrt(x))*(1.25331414+y*(-0.7832358e-1+y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2+y*(-0.251540e-2+y*0.53208e-3))))))
        
    return ans


@cuda.jit('float32(float32)',
          device=True,
          fastmath=True)
def my_i1e(x):
    '''
    DEVICE FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    1st order modified Bessel function of the first kind exponentially scaled.
    For asymptotic expansion see Abramowitz and Stegun. 

        Parameters:
                x(float32): float number 

        Returns: 
                ans(float32): 1st order modified Bessel function of the first kind exponentially scaled 
    '''
    if (x < 3.75):
        y=x/3.75
        y*=y
        ans=exp(-x)*x*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934+y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))))
    else :
        y=3.75/x
        ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1-y*0.420059e-2))
        ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2+y*(0.163801e-2+y*(-0.1031555e-1+y*ans))))
        ans *= (1/sqrt(x))

    return ans

@cuda.jit('float32(float32,float32)',
          device=True,
          fastmath=True)
def my_k1e(x,i1e):
    '''
    DEVICE FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    1st order modified Bessel function of the second kind exponentially scaled.
    For asymptotic expansion see Abramowitz and Stegun. 
    
        Parameters:
                x(float32)   : float number 
                i1e(float32) : output of my_i1e

        Returns: 
                ans(float32): 1st order modified Bessel function of the second kind exponentially scaled 
    '''
    if (x <= 2.0):
        y=x*x/4.0
        ans=exp(x)*((log(x/2.0)*exp(x)*my_i1e(x))+(1.0/x)*(1.0+y*(0.15443144+y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1+y*(-0.110404e-2+y*(-0.4686e-4))))))))
    else:
        y=2.0/x
        ans=(1/sqrt(x))*(1.25331414+y*(0.23498619+y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2+y*(0.325614e-2+y*(-0.68245e-3)))))))
    
    return ans 


#---------------------------------------------------------------------------
                    #TOTAL QUANTITIES
#---------------------------------------------------------------------------

@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
          device = True,
          inline = True,
          fastmath = True)
def v_avg (peso1,   # disc1 surface density
           peso2,   # disc2 surface density
           peso3,   # bulge surface density
           mu1,     # disc1 mean velocity
           mu2,     # disc2 mean velocity
           mu3) :   # bulge mean velocity (zero)
    ''' 
    DEVICE FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Return average velocity of 3 components.

        Parameters:
                peso1(float32) : weight of the first component
                peso2(float32) : weight of the second component
                peso3(float32) : weight of the third component
                mu1(float32)   : velocity of the first component
                mu2(float32)   : velocity of the second component
                mu3(float32)   : velocity of the third component

        Returns:
                out(float32) : weighted average of the 3 components
    '''

    return (peso1*mu1+peso2*mu2+peso3*mu3)/(peso1+peso2+peso3)


@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
          device = True,
          fastmath=True)
def sigma_avg (peso1,            # disc surface density
               peso2,            # bulge surface density
               peso3,            # bulge surface density
               sigma1_square,    # disc mean velocity dispersion squared
               sigma2_square,    # disc mean velocity dispersion squared
               sigma3_square) :  # bulge mean velocity dispersion squared
    
    ''' 
    DEVICE FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Return average velocity dispersion of 3 components.

        Parameters:
                peso1(float32) : weight of the first component
                peso2(float32) : weight of the second component
                peso3(float32) : weight of the third component
                sigma1_square(float32)   : dispersion squared of the first component
                sigma2_square(float32)   : dispersion squared of the second component
                sigma3_square(float32)   : dispersion squared of the third component

        Returns:
                out(float32) : weighted average of the 3 components

    Notes:
        v_avg and sigma_avg do exactly the same things, they can be reduced to a common func
    '''


    return (peso1*sigma1_square+peso2*sigma2_square+peso3*sigma3_square)/(peso1+peso2+peso3)

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
                    #MODEL
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

@cuda.jit('void(float32[::1],float32[::1],float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def coordinate_transformation(x,                    # x position refined grid è N X J X K
                              y,                    # y position refined grid
                              x0,                  # x center position
                              y0,                  # y center position
                              theta,                # P.A. angle
                              incl,                 # inclination angle
                              r_proj,               # (N,J,K) projected radius (for Herquinst)
                              r_true,               # (N,J,K) real radius (for velocity and exp disc)
                              phi):                 # (N,J,K) azimuthal angle
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute extrinsic parameter transformations: 
                                                - translation
                                                - rotation
                                                - de-projection

        Parameters:
                x(1D arr float32) : x coordinate of the refined grid (N X J X K)
                y(1D arr float32) : y coordinate of the refined grid (N X J X K)
                x0(float32)       : x position of the center 
                y0(float32)       : y position of the center 
                theta(float32)    : P.A. angle
                incl(float32)     : inclination angle

        Returns:
                r_proj(1D arr float32) : projected radius (R in the paper)
                r_true(1D arr float32) : 3D distance (r in the paper)
                phi(1D arr float32)    : azimuthal angle in the plane of the disc 

    Notes:
        Refined grid starts from the data grid (N X J) adding K new point for PSF convolution
    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    c_theta = cos(theta)
    s_theta = sin(theta)
    c_incl = cos(incl)

    if (i < x.shape[0]):
        xtr = x[i]-x0
        ytr = y[i]-y0 
        xr = xtr*c_theta+ytr*s_theta
        yr = ytr*c_theta-xtr*s_theta
        yd = yr/c_incl                             

        r_proj[i] = radius(xtr,ytr)   
        r_true[i] = radius(xr,yd)
        phi[i]    = atan2(yd,xr) 



@cuda.jit('void(float32,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def Xfunction(Rb,r_proj,X)  :
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute X(s) function useful for Hernquist/Jaffe surface density and 
    dispersion (see Hernquist 1990) 
                                             
        Parameters:
                Rb(float32)              : bulge scale radius
                r_proj(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)

        Returns:
                X(1D arr float32)   : X(s) function 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r_proj.shape[0]):
        s = r_proj[i]
        s = s/Rb
        one_minus_s2 = 1.0-s*s
        if s == 0:
            s = 1e-8
            
        if s < 1.0:
            tmp = log((1.0 + sqrt(one_minus_s2)) / s) / sqrt(one_minus_s2)
            X[i] = tmp 
        elif s ==  1.0:
            X[i] = 1.0
        else:
            tmp = acos(1.0/s)/sqrt(-one_minus_s2)
            X[i] = tmp 
    else:
        pass

@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def X1dot(Rb,r_proj,X,X1_dot):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                Rb(float32)              : bulge scale radius
                r_proj(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                X (1D arr float32)       : output of kernel X

        Returns:
                X1_dot(1D arr float32)   : first derivative of X 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r_proj.shape[0]):
        s = r_proj[i]
        s = s/Rb

        if 0.999<=s<=1.001:
            s_minus_one1 = s-1.0
            s_minus_one2 = s_minus_one1*s_minus_one1
            s_minus_one3 = s_minus_one2*s_minus_one1
            s_minus_one4 = s_minus_one2*s_minus_one2

            tmp = -2./3. + 14.*s_minus_one1/15. - 36./35.*s_minus_one2 + 332./315.*s_minus_one3 -730./693.*s_minus_one4
            X1_dot[i] = tmp
        else:
            X_tmp = X[i]
            X1_dot = (1./s - s*X_tmp)/(s*s-1)


@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def X2dot(Rb,r_proj,X,X2_dot):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                Rb(float32)              : bulge scale radius
                r_proj(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                X (1D arr float32)       : output of kernel X

        Returns:
                X2_dot(1D arr float32)   : second derivative of X 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r_proj.shape[0]):
        s = r_proj[i]
        s = s/Rb

        if 0.999<=s<=1.001:
            s_minus_one1 = s-1.0
            s_minus_one2 = s_minus_one1*s_minus_one1
            s_minus_one3 = s_minus_one2*s_minus_one1
            s_minus_one4 = s_minus_one2*s_minus_one2

            tmp = 14./15. - 72.*s_minus_one1/35. + 332./105.*s_minus_one2 - 2920./693.*s_minus_one3 + 5230./1001.*s_minus_one4
            X2_dot[i] = tmp
        else:
            X_tmp = X[i]
            s2    = s*s
            s2_minus_one1 = s2-1.0

            tmp = (-4.0+1.0/s2*(1.+2.*s2)*X_tmp)/(s2_minus_one1*s2_minus_one1)
            X2_dot[i] = tmp


@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def X3dot(Rb,r_proj,X,X3_dot):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                Rb(float32)              : bulge scale radius
                r_proj(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                X (1D arr float32)       : output of kernel X

        Returns:
                X2_dot(1D arr float32)   : second derivative of X 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r_proj.shape[0]):
        s = r_proj[i]
        s = s/Rb

        if 0.999<=s<=1.001:
            s_minus_one1 = s-1.0
            s_minus_one2 = s_minus_one1*s_minus_one1
            s_minus_one3 = s_minus_one2*s_minus_one1
            s_minus_one4 = s_minus_one2*s_minus_one2

            tmp = -72./35. + 664.*s_minus_one1/105. - 2920./231.*s_minus_one2 + 20920./1001.*s_minus_one3 - 13328./429.*s_minus_one4
            X3_dot[i] = tmp
        else:
            X_tmp = X[i]
            s2    = s*s
            s3    = s*s2
            s2_minus_one1 = s2-1.0

            tmp = (18.*s-5./s+2./s3-3*s*(3.+2.*s2)*X_tmp)/(s2_minus_one1*s2_minus_one1*s2_minus_one1)
            X3_dot[i] = tmp


@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def all_Xdot(Rb,r_proj,X,X1_dot,X2_dot,X3_dot):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                Rb(float32)              : bulge scale radius
                r_proj(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                X (1D arr float32)       : output of kernel X

        Returns:
                X1_dot(1D arr float32)   : first derivative of X 
                X2_dot(1D arr float32)   : second derivative of X 
                X3_dot(1D arr float32)   : third derivative of X 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r_proj.shape[0]):
        s = r_proj[i]
        s = s/Rb

        if 0.93<=s<=1.07:
            s_minus_one1 = s-1.0
            s_minus_one2 = s_minus_one1*s_minus_one1
            s_minus_one3 = s_minus_one2*s_minus_one1
            s_minus_one4 = s_minus_one2*s_minus_one2

            X1_dot[i] = -2./3. + 14.*s_minus_one1/15. - 36./35.*s_minus_one2 + 332./315.*s_minus_one3 -730./693.*s_minus_one4
            X2_dot[i] = 14./15. - 72.*s_minus_one1/35. + 332./105.*s_minus_one2 - 2920./693.*s_minus_one3 + 5230./1001.*s_minus_one4
            X3_dot[i] = -72./35. + 664.*s_minus_one1/105. - 2920./231.*s_minus_one2 + 20920./1001.*s_minus_one3 - 13328./429.*s_minus_one4

        else:
            if s >= 7.5:
                pass
            else:
                X_tmp = X[i]
                s2    = s*s
                s3    = s*s2
                s2_minus_one1 = s2-1.0
                X1_dot[i] = (1./s - s*X_tmp)/(s*s-1.)
                X2_dot[i] = (-4.0+1.0/s2+(1.+2.*s2)*X_tmp)/(s2_minus_one1*s2_minus_one1)
                X3_dot[i] = (18.*s-5./s+2./s3-3.*s*(3.+2.*s2)*X_tmp)/(s2_minus_one1*s2_minus_one1*s2_minus_one1)


@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def Y12(Rb,r_proj,X,Y_12):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                Rb(float32)              : bulge scale radius
                r_proj(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                X (1D arr float32)       : output of kernel X

        Returns:
                Y_12(1D arr float32)   : recurrence relation for BH_herq sigma

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r_proj.shape[0]):
        s = r_proj[i]
        s = s/Rb
        s2 = s*s
        X_tmp = X[i]

        tmp = ( 1/2.*s - 1/M_PI - 1./(8.*s) + 1./(12.*M_PI*s2) + (3.-4.*s2)/(4.*M_PI)*X_tmp)
        Y_12[i] = tmp
    else:
        pass


@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def Y1(Rb,r_proj,X,X1_dot,Y_1):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                Rb(float32)              : bulge scale radius
                r_proj(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                X (1D arr float32)       : output of kernel X

        Returns:
                Y_1(1D arr float32)   : recurrence relation for BH_herq sigma

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r_proj.shape[0]):
        s = r_proj[i]
        s = s/Rb
        if s >= 7.5:
            pass
        else:
            s2 = s*s
            X_tmp = X[i]
            X1dot_tmp = X1_dot[i]

            tmp = (M_PI + 16.*s - 12.*M_PI*s2 + 4.*s*(8.*s2-3.)*X_tmp + (8.*s2*s2-6.*s2)*X1dot_tmp)/(8.*M_PI*s)
            Y_1[i] = tmp
    else:
        pass


@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def Y32(Rb,r_proj,X,X1_dot,X2_dot,Y_32):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                Rb(float32)              : bulge scale radius
                r_proj(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                X (1D arr float32)       : output of kernel X
                X1_dot (1D arr float32)       : first derivative of X
                X2_dot (1D arr float32)       : second derivative of X

        Returns:
                Y_32(1D arr float32)   : recurrence relation for BH_herq sigma

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r_proj.shape[0]):
        s = r_proj[i]
        s = s/Rb
        s2 = s*s
        X_tmp = X[i]
        X1dot_tmp = X1_dot[i]
        X2dot_tmp = X2_dot[i]

        tmp = (-8.+12.*M_PI*s + (6.-48.*s2)*X_tmp - 4.*s*(8.*s2-3.)*X1dot_tmp + (3.*s2 - 4.*s2*s2)*X2dot_tmp)/(8.*M_PI)
        Y_32[i] = tmp
    else:
        pass


@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def Y2(Rb,r_proj,X,X1_dot,X2_dot,X3_dot,Y_2):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                Rb(float32)              : bulge scale radius
                r_proj(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                X (1D arr float32)       : output of kernel X
                X1_dot (1D arr float32)       : first derivative of X
                X2_dot (1D arr float32)       : second derivative of X
                X2_dot (1D arr float32)       : third derivative of X

        Returns:
                Y_32(1D arr float32)   : recurrence relation for BH_herq sigma

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r_proj.shape[0]):
        s = r_proj[i]
        s = s/Rb
        if s >= 7.5:
            pass
        else:
            s2 = s*s
            X_tmp = X[i]
            X1dot_tmp = X1_dot[i]
            X2dot_tmp = X2_dot[i]
            X3dot_tmp = X3_dot[i]

            tmp = (-12.*M_PI+96.*s*X_tmp + 18.*(8.*s2-1.)*X1dot_tmp+(-18.*s+48.*s2*s)*X2dot_tmp+(-3.*s2+4.*s2*s2)*X3dot_tmp)*s/(24.*M_PI)
            Y_2[i] = tmp
    else:
        pass

@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def rho_D(Md,Rd,incl,r,all_rhoD) :
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute exponential disc projected surface density.  
                                             
        Parameters:
                Md(float32)        : disc mass
                Rd(float32)        : disc scale radius
                incl(float32)      : inclination angle
                r(1D arr float32)  : 3D distance (see r_true in coordinate_transformation)

        Returns:
                all_rhoD(1D arr float32)   : exp disc projected density 
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if (i < r.shape[0]):
        r_tmp = r[i]
        if r_tmp/Rd > 80:
            all_rhoD[i] = 1e-10
        else:
            rho_tmp = (Md / (2.0 * M_PI * Rd*Rd) ) * exp(-r_tmp/Rd) 
            all_rhoD[i] = rho_tmp/cos(incl)
    else:
        pass



@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def herquinst_rho(M,Rb,r,rhoB,Xs):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Hernquist projected surface density.  
                                             
        Parameters:
                M(float32)          : bulge mass
                Rb(float32)         : bulge scale radius
                r(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                Xs(1D arr float32)  : output of X1 
        Returns:
                rhoB(1D arr float32)   : Hernquist surface density 

    '''


    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r.shape[0]):

        Rb2 = Rb*Rb
        s = r[i]
        s = s/Rb

        if s == 0:
            s = 1e-8

        if s >= 0.7 and s <=1.3:
            s_minus_one = s-1.0
            square_s_minus_one = s_minus_one*s_minus_one
            cube_s_minus_one = square_s_minus_one*s_minus_one
            fourth_s_minus_one = square_s_minus_one*square_s_minus_one
            fifth_s_minus_one = cube_s_minus_one*square_s_minus_one
            rho_tmp = M / (2.0*M_PI*Rb2) *( 4.0/15.0 - 16.0*s_minus_one/35.0 + (8.0* square_s_minus_one)/15.0 - (368.0* cube_s_minus_one)/693.0 + (1468.0* fourth_s_minus_one)/3003.0 - (928.0* fifth_s_minus_one)/2145.0)
        else:
            X = Xs[i]
            s2 = s*s
            one_minus_s2 =1.0-s2
            one_minus_s2_2 = one_minus_s2*one_minus_s2
            rho_tmp =  (M / (2.0 * M_PI * Rb2 * one_minus_s2_2)) * ((2.0 + s2) * X - 3.0) 
        rhoB[i] = rho_tmp
    else:
        pass



@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def herquinst_sigma(M,Rb,r,sigmaB2,Xs):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Hernquist projected velocity dispersion.  
                                             
        Parameters:
                M(float32)          : bulge mass
                Rb(float32)         : bulge scale radius
                r(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                Xs(1D arr float32)  : output of X1 
        Returns:
                sigmaB2(1D arr float32)   : Hernquist projected velocity dispersion squared 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x


    if (i < r.shape[0]) :            
        Rb2 = Rb*Rb
        s = r[i]
        s = s/Rb
        M_PI2 = M_PI*M_PI
        M_PI3 = M_PI2*M_PI
        G = 4.299e-6

        if s == 0:
            s = 1e-8

        if s >= 0.7 and s <=1.3:
            s_minus_one = s-1.0
            square_s_minus_one = s_minus_one*s_minus_one
            sigma_tmp = (G*M/Rb) * ( (332.0-105.0*M_PI)/28.0 + (18784.0-5985.0*M_PI)*s_minus_one/588.0 + (707800.0-225225.0*M_PI)*square_s_minus_one/22638.0 )

        else:
            X = Xs[i]
            s2 = s*s
            one_minus_s2 = (1.0 - s2)
            one_minus_s2_2 = one_minus_s2*one_minus_s2
            if s >= 12.5:
                s3 = s*s2
                A =  (G * M * 8.0) / (15.0 * s * M_PI * Rb)
                B =  (8.0/M_PI - 75.0*M_PI/64.0) / s
                C =  (64.0/(M_PI2) - 297.0/56.0) / s2
                D =  (512.0/(M_PI3) - 1199.0/(21.0*M_PI) - 75.0*M_PI/512.0) / s3
                sigma_tmp = A * (1 + B + C + D) 
            else:
                s4 = s2*s2 
                s6 = s4*s2
                rho = (M / (2.0 * M_PI * Rb2 * one_minus_s2_2)) * ((2.0 + s2) * X - 3.0) 
                A = (G * M*M ) / (12.0 * M_PI * Rb2 * Rb * rho)
                B = 1.0 /  (2.0*one_minus_s2_2*one_minus_s2)
                C = (-3.0 * s2) * X * (8.0*s6 - 28.0*s4 + 35.0*s2 - 20.0) - 24.0*s6 + 68.0*s4 - 65.0*s2 + 6.0
                D = 6.0*M_PI*s
                
                sigma_tmp =  A*(B*C-D)
        
        sigmaB2[i] = sigma_tmp
    else:
        pass



@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def herquinst_sigma_BH(Mbh,Mb,Rb,r,Y_1,Y_2,rhoB,sigmaB2):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Hernquist projected velocity dispersion considering central BH.  
                                             
        Parameters:
                Mbh(float32)        : BH mass
                Mb(float32)         : bulge scale mass
                Rb(float32)         : bulge scale radius
                r(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                Y_1(1D arr float32)  : output of Y1
                Y_2(1D arr float32)  : output of Y2
                rhoB(1D arr float32) : herquinst surface brightness
        Returns:
                sigmaB2(1D arr float32)   : Hernquist projected velocity dispersion squared with BH 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x


    if (i < r.shape[0]) : 
        G = 4.299e-6
        s = r[i]/Rb
        if s >= 7.5:
            s2 = s*s 
            s3 = s*s2
            M_PI2 = M_PI*M_PI
            sigma_exp3 = (-157.*G*Mb/(105.*M_PI*Rb) + 8.*G*Mb*(-2.5 + 64./M_PI2)/(15.*M_PI*Rb) - 47.*G*Mbh/(21.*M_PI*Rb) + \
                         8.*G*Mbh*(-2.5 + 64./M_PI2)/(15.*M_PI*Rb))/s3 + (-5.*G*Mb/(8.*Rb) + 64.*G*Mb/(15.*M_PI2*Rb) - \
                         3.*G*Mbh/(8.*Rb) + 64.*G*Mbh/(15.*M_PI2*Rb))/s2 + (8.*G*Mb/(15.*M_PI*Rb) + 8.*G*Mbh/(15.*M_PI*Rb))/s
            sigmaB2[i] = sigma_exp3
        else:
            Rb3   = Rb*Rb*Rb 
            norm1 = G*Mb*Mb/Rb3
            norm2 = G*Mb/Rb3

            sigmaB2[i] = (norm1*Y_2[i]+norm2*Mbh*2*Y_1[i])/rhoB[i]


@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def jaffe_rho(Mb,Rb,r,rhoB,Xs):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Jaffe projected surface density.  
                                             
        Parameters:
                M(float32)          : bulge mass
                Rb(float32)         : bulge scale radius
                r(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                Xs(1D arr float32)  : output of X1 
        Returns:
                rhoB(1D arr float32)   : Jaffe surface density 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r.shape[0]): 
        s = r[i]
        s = s/Rb
        if 0.99<s<1.01:
            rho_tmp = 0.25-2.0/3.0*M_PI
        else:
            if s > 300.0:
                s2 = s*s
                s3 = s2*s
                s4 = s2*s2
                rho_tmp = 1.0/(s3*8.0)-2.0/(3*M_PI*s4)
            else:
                s2 = s*s
                X = Xs[i]
                rho_tmp = 1.0/(4.0*s) + (1.0-(2.0-s2)*X)/(2.0*M_PI*(1.0-s2)) 
        
        rhoB[i] = Mb/(Rb*Rb)*rho_tmp 


@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def jaffe_sigma(Mb,Rb,r,sigmaB2,Xs):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Jaffe projected velocity dispersion.  
                                             
        Parameters:
                M(float32)          : bulge mass
                Rb(float32)         : bulge scale radius
                r(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                Xs(1D arr float32)  : output of X1 
        Returns:
                sigmaB2(1D arr float32)   : Jaffe projected velocity dispersion squared 

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x


    if (i < r.shape[0]) : 
        s = r[i]
        s = s/Rb
        G = 4.299e-6

        if  0.99<s<1.01:
            rho_tmp = 0.25-2.0/3.0*M_PI
            sigma_tmp = (G*Mb/(4.0*M_PI*Rb*rho_tmp))*(-0.15372602*s+0.20832732)
        else:
            if s >=4.0:
                M_PI2 = M_PI*M_PI
                s2 = s*s
                s3 = s2*s
                sigma_tmp = G*Mb/(8.0*Rb*M_PI)*( 64.0/(15.0*s) + (1024.0/(45.0*M_PI)-3.0*M_PI)/s2 - 128.0*(-896.0+81.0*M_PI2)/(945.0*M_PI2*s3) ) 
            else:
                s2 = s*s
                s3 = s2*s
                s4 = s2*s2
                X = Xs[i]

                rho_tmp = 1.0/(4.0*s) + (1.0-(2.0-s2)*X)/(2.0*M_PI*(1.0-s2)) 
                A = (M_PI/(2*s)+11.0-6.5*M_PI*s-12.0*s2+6.0*M_PI*s3-X*(6.0-19.0*s2+12.0*s4))/(1.0-s2)
                B = G*Mb/(4*M_PI*Rb*rho_tmp)
            
                sigma_tmp = A*B
        
        sigmaB2[i] = sigma_tmp


@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def v_D(Md,         # disc mass
        Rd,         # disc mass
        r,          # true radius
        v_exp):     # exponential disc velocity squared
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute exp disc circular velocity squared.
                                             
        Parameters:
                Md(float32)         : disc mass
                Rd(float32)         : disc scale radius
                r(1D arr float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                v_exp(1D arr float32)   : exp disc circular velocity squared.

    '''
    
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    G = 4.299e-6
    v_c2 = 0.

    if i < r.shape[0]:
        r_ns = r[i]
        r_tmp = r_ns/(2.0*Rd)
        if r_tmp == 0.:
            r_tmp = 1e-8
            r_ns = 1e-8
        if r_tmp <= 1000.:
            i0e = my_i0e(r_tmp)
            i1e = my_i1e(r_tmp)
            v_c2 =  ((G * Md * r_ns) / (Rd*Rd)) * r_tmp * (i0e*my_k0e(r_tmp,i0e) - i1e*my_k1e(r_tmp,i1e))
        else:
            r2 = r_ns*r_ns 
            r3 = r2*r_ns
            r5 = r3*r2 
            v_c2 = ((G * Md * r_ns) / (Rd*Rd)) * r_tmp * (1.0/(4.0*r3) + 9.0/(32.0*r5))

        v_exp[i] = v_c2
    else:
        pass


@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def v_H(Mh,        # bulge mass
        Rh,        # halo mass
        r,         # true radius
        v_H):      # halo velocity
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Hernquist circular velocity squared.
                                             
        Parameters:
                Mh(float32)         : bulge/halo mass
                Rh(float32)         : bulge/halo scale radius
                r(1D arr float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                v_H(1D arr float32)   : Hernquist circular velocity squared. 

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if i < r.shape[0] :
        G = 4.299e-6
        r_tmp = r[i]
        r_tmp_H = r_tmp/Rh
        one_plus_r_H = 1.0+r_tmp_H
        v_c2_H = G * Mh * r_tmp_H / (Rh*one_plus_r_H*one_plus_r_H)
        v_H[i] = v_c2_H
    else:
        pass

@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def v_J(Mb,        # bulge mass
        Rb,        # bulge radius
        r,         # true radius
        v_B):       # bulge velocity
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Jaffe(bulge)+Hernquist(halo) circular velocity squared.
                                             
        Parameters:
                Mb(float32)         : bulge mass
                Rb(float32)         : bulge scale radius
                Mh(float32)         : halo mass
                Rh(float32)         : halo scale radius
                r(1D arr float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                v_B(1D arr float32)   : Jaffe(bulge) circular velocity squared. 
                v_H(1D arr float32)   : Hernquist(Halo) circular velocity squared. 

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if i < r.shape[0] :
        G = 4.299e-6
        r_tmp = r[i]
        v_c2_B = G*Mb/(r_tmp+Rb)
        v_B[i] = v_c2_B
    else:
        pass


@cuda.jit('void(float32,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def v_BH(Mbh,        # bulge mass
         r,         # true radius
         vBH):       # bulge velocity
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute BH circular velocity squared.
                                             
        Parameters:
                Mbh(float32)        : BH mass
                r(1D arr float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                vBH(1D arr float32)  : BH velocities

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if i < r.shape[0] :
        G = 4.299e-6
        r_tmp = r[i]
        v_c2_BH = G*Mbh/(r_tmp)
        vBH[i] = v_c2_BH
    else:
        pass

@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
          fastmath = True,
          #max_registers=10,
          debug = False)
def v_tot(v_exp1,        # disc velocity squared
          v_exp2,
          v_bulge,      # bulge velocity squared 
          v_halo,       # halo velocity squared
          incl,         # sin(incl)
          phi,          # phi azimuthal angle
          all_v,
          all_v2_abs):       # velocity
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute total circular velocity squared and line of sight velocity assuming (B+D1+D2+H).
                                             
        Parameters:
                v_exp1(1D arr float32)    : circ vel squared disc1 (see v_exp in v_D kernel)
                v_exp2(1D arr float32)    : circ vel squared disc2 (see v_exp in v_D kernel)
                v_bulge(1D arr float32)   : circ vel squared bulge (see v_H/v_B in v_H/v_Jaffe_H kernel)
                v_halo(1D arr float32)    : circ vel squared halo (see v_H/v_H in v_H/v_Jaffe_H kernel)
                incl(float32)             : inclination angle
                phi(1D arr float32)       : azimuthal angle in the disc plane (see phi in coordinate_transformation)
        Returns:
                all_v(1D arr float32)       : LOS velocity
                all_v2_abs(1D arr float32)  : total circular velocity 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < v_exp1.shape[0]) :  
        phi_tmp = phi[i]
        v_tmp1 = v_exp1[i]+v_bulge[i]+v_halo[i]+v_exp2[i]
        v_tmp = sin(incl)*cos(phi_tmp)*sqrt(v_tmp1)
        all_v[i]      = v_tmp 
        all_v2_abs[i] = v_tmp1
    else:
        pass
 
@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
          fastmath = True,
          #max_registers=10,
          debug = False)
def v_tot_BD(v_exp1,        # disc velocity squared
          v_bulge,      # bulge velocity squared 
          v_halo,       # halo velocity squared
          incl,         # sin(incl)
          phi,          # phi azimuthal angle
          all_v,
          all_v2_abs):       # velocity
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute total circular velocity squared and line of sight velocity assuming (B+D1+D2+H).
                                             
        Parameters:
                v_exp1(1D arr float32)    : circ vel squared disc1 (see v_exp in v_D kernel)
                v_exp2(1D arr float32)    : circ vel squared disc2 (see v_exp in v_D kernel)
                v_bulge(1D arr float32)   : circ vel squared bulge (see v_H/v_B in v_H/v_Jaffe_H kernel)
                v_halo(1D arr float32)    : circ vel squared halo (see v_H/v_H in v_H/v_Jaffe_H kernel)
                incl(float32)             : inclination angle
                phi(1D arr float32)       : azimuthal angle in the disc plane (see phi in coordinate_transformation)
        Returns:
                all_v(1D arr float32)       : LOS velocity
                all_v2_abs(1D arr float32)  : total circular velocity 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < v_exp1.shape[0]) :  
        phi_tmp = phi[i]
        v_tmp1 = v_exp1[i]+v_bulge[i]+v_halo[i]
        v_tmp = sin(incl)*cos(phi_tmp)*sqrt(v_tmp1)
        all_v[i]      = v_tmp 
        all_v2_abs[i] = v_tmp1
    else:
        pass

@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
          fastmath = True,
          #max_registers=10,
          debug = False)
def v_tot_BH(v_exp1,        # disc velocity squared
            v_exp2,
            v_bulge,      # bulge velocity squared 
            v_halo,       # halo velocity squared
            v_BH,       # BH velocity squared
            incl,         # sin(incl)
            phi,          # phi azimuthal angle
            all_v,
            all_v2_abs):       # velocity
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute total circular velocity squared and line of sight velocity assuming (BH+B+D1+D2+H).
                                             
        Parameters:
                v_exp1(1D arr float32)    : circ vel squared disc1 (see v_exp in v_D kernel)
                v_exp2(1D arr float32)    : circ vel squared disc2 (see v_exp in v_D kernel)
                v_bulge(1D arr float32)   : circ vel squared bulge (see v_H/v_B in v_H/v_Jaffe_H kernel)
                v_halo(1D arr float32)    : circ vel squared halo (see v_H/v_H in v_H/v_Jaffe_H kernel)
                incl(float32)             : inclination angle
                phi(1D arr float32)       : azimuthal angle in the disc plane (see phi in coordinate_transformation)
        Returns:
                all_v(1D arr float32)       : LOS velocity
                all_v2_abs(1D arr float32)  : total circular velocity 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < v_exp1.shape[0]) :  
        phi_tmp = phi[i]
        v_tmp1 = v_exp1[i]+v_bulge[i]+v_halo[i]+v_exp2[i]+v_BH[i]
        v_tmp = sin(incl)*cos(phi_tmp)*sqrt(v_tmp1)
        all_v[i]      = v_tmp 
        all_v2_abs[i] = v_tmp1
    else:
        pass

@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
          fastmath = True,
          #max_registers=10,
          debug = False)
def v_tot_BD_BH(v_exp1,        # disc velocity squared
            v_bulge,      # bulge velocity squared 
            v_halo,       # halo velocity squared
            v_BH,       # BH velocity squared
            incl,         # sin(incl)
            phi,          # phi azimuthal angle
            all_v,
            all_v2_abs):       # velocity
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute total circular velocity squared and line of sight velocity assuming (BH+B+D1+H).
                                             
        Parameters:
                v_exp1(1D arr float32)    : circ vel squared disc1 (see v_exp in v_D kernel)
                v_exp2(1D arr float32)    : circ vel squared disc2 (see v_exp in v_D kernel)
                v_bulge(1D arr float32)   : circ vel squared bulge (see v_H/v_B in v_H/v_Jaffe_H kernel)
                v_halo(1D arr float32)    : circ vel squared halo (see v_H/v_H in v_H/v_Jaffe_H kernel)
                incl(float32)             : inclination angle
                phi(1D arr float32)       : azimuthal angle in the disc plane (see phi in coordinate_transformation)
        Returns:
                all_v(1D arr float32)       : LOS velocity
                all_v2_abs(1D arr float32)  : total circular velocity 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < v_exp1.shape[0]) :  
        phi_tmp = phi[i]
        v_tmp1 = v_exp1[i]+v_bulge[i]+v_halo[i]+v_BH[i]
        v_tmp = sin(incl)*cos(phi_tmp)*sqrt(v_tmp1)
        all_v[i]      = v_tmp 
        all_v2_abs[i] = v_tmp1
    else:
        pass
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
                    #MODEL PSF B+D1+D2+H
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_rho_v(all_rhoB,         # (N,J,K) contains all bulge surface density
              all_rhoD1,        # (N,J,K) contains all disc surface density
              all_rhoD2,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              all_sigma2,
              psf,              # (N,J,K) psf weights
              L_to_M_b,
              L_to_M_d1,
              L_to_M_d2,
              k_D1,
              k_D2,
              rho_AVG,
              v_AVG,
              sigma_AVG):


    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - total surface brightness
            - mean line of sight velocity
            - mean line of sight velocity dispersion
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe sigma)
                psf(1D arr float32)         : psf weight (of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                L_to_M_d2(float32)          : light to mass disc 2
                k_D1(float32)               : kinematic decomposition disc 1
                k_D2(float32)               : kinematic decomposition disc 2
        Returns:
                rho_AVG(1D arr float32)     : total surface brightness
                v_AVG(1D arr float32)       : average line of sight velocity
                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        K = psf.shape[0]
        
        psf_sum = 0.

        rhoB_sum = 0.
        rhoD1_sum = 0.
        rhoD2_sum = 0.
        rhotot_sum = 0.

        vD1_sum = 0.
        vD2_sum = 0.
        vD1_2_sum   = 0.
        vD2_2_sum   = 0.
        vB_2_sum    = 0.
        
        for k in range(i*K,(i+1)*K):
            psf_tmp    = psf[int(k-i*K)]

            rhoB_tmp    = L_to_M_b*all_rhoB[k]
            rhoD1_tmp   = L_to_M_d1*all_rhoD1[k]
            rhoD2_tmp   = L_to_M_d2*all_rhoD2[k]
            
            v_tmp      = all_v[k]
            v2_tmp     = all_v2[k] 
            sigma2_tmp = all_sigma2[k] 
            
            
            psf_sum    = psf_sum+psf_tmp
            
            rhoB_sum    = rhoB_sum+rhoB_tmp*psf_tmp
            rhoD1_sum   = rhoD1_sum+rhoD1_tmp*psf_tmp 
            rhoD2_sum   = rhoD2_sum+rhoD2_tmp*psf_tmp 
            rhotot_sum  = rhotot_sum + (rhoB_tmp+rhoD1_tmp+rhoD2_tmp)*psf_tmp 
            
            vD1_sum   = vD1_sum+psf_tmp*rhoD1_tmp*v_tmp
            vD2_sum   = vD2_sum+psf_tmp*rhoD2_tmp*v_tmp
            
            vD1_2_sum   = vD1_2_sum+psf_tmp*rhoD1_tmp*v2_tmp
            vD2_2_sum   = vD2_2_sum+psf_tmp*rhoD2_tmp*v2_tmp
            vB_2_sum    = vB_2_sum+psf_tmp*rhoB_tmp*sigma2_tmp

        rhoB_mean  = rhoB_sum/psf_sum
        rhoD1_mean = rhoD1_sum/psf_sum
        rhoD2_mean = rhoD2_sum/psf_sum
        rho_AVG[i] = rhotot_sum/psf_sum
        vD1_mean        = k_D1*vD1_sum/rhoD1_sum
        vD2_mean        = k_D2*vD2_sum/rhoD2_sum

        sigmaD1_2_mean = (1.0-k_D1*k_D1)*vD1_2_sum/rhoD1_sum
        sigmaD2_2_mean = (1.0-k_D2*k_D2)*vD2_2_sum/rhoD2_sum
        sigmaB2_mean   = vB_2_sum/rhoB_sum
        

        v_AVG[i]     = v_avg(rhoD1_mean,rhoD2_mean,rhoB_mean,vD1_mean,vD2_mean,0)#vB_mean)
        sigma_AVG[i] = sqrt(sigma_avg(rhoD1_mean,rhoD2_mean,rhoB_mean,sigmaD1_2_mean/3.0,sigmaD2_2_mean/3.0,sigmaB2_mean))

    else:
        pass



@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1])',
          fastmath=True,
          debug = False)
def avg_LM(all_rhoB,        # (N,J,K) contains all bulge surface density
           all_rhoD1,        # (N,J,K) contains all disc surface density
           all_rhoD2,        # (N,J,K) contains all disc surface density
           psf,             # (N,J,K) psf weights
           L_to_M_bulge,
           L_to_M_disc1,
           L_to_M_disc2,
           LM_AVG):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute avg light to mass (DECIDE NOTATION)
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
                psf(1D arr float32)         : psf weight (of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                L_to_M_d2(float32)          : light to mass disc 2
        Returns:
                LM_AVG(1D arr float32)   : average light to mass ratio
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < LM_AVG.shape[0]):
        K = psf.shape[0]

        rho_tot = 0.
        LM_sum = 0.
        for k in range(i*K,(i+1)*K):
            rhoD1_tmp = all_rhoD1[k]
            rhoD2_tmp = all_rhoD2[k]
            rhoB_tmp  = all_rhoB[k]
            psf_tmp   = psf[int(k-i*K)]
            rho_tot   = rho_tot+psf_tmp*(rhoD1_tmp+rhoB_tmp+rhoD2_tmp)
            LM_sum    = LM_sum+psf_tmp*(rhoD2_tmp*L_to_M_disc2+rhoB_tmp*L_to_M_bulge+rhoD1_tmp*L_to_M_disc1)

        LM_AVG[i] =  LM_sum/rho_tot

    else:
        pass

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
                    #MODEL PSF 2 vis components +H
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_rho_v_2_vis_components(all_rhoB,         # (N,J,K) contains all bulge surface density
              all_rhoD1,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              all_sigma2,
              psf,              # (N,J,K) psf weights
              L_to_M_b,
              L_to_M_d1,
              k_D1,
              rho_AVG,
              v_AVG,
              sigma_AVG):


    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute (for model with 2 visible components only):
            - total surface brightness
            - mean line of sight velocity
            - mean line of sight velocity dispersion
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe sigma)
                psf(1D arr float32)         : psf weight (of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                k_D1(float32)               : kinematic decomposition disc 1
        Returns:
                rho_AVG(1D arr float32)     : total surface brightness
                v_AVG(1D arr float32)       : average line of sight velocity
                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        K = psf.shape[0]
        
        psf_sum = 0.

        rhoB_sum = 0.
        rhoD1_sum = 0.
        rhotot_sum = 0.

        vD1_sum = 0.
        vD1_2_sum   = 0.
        vB_2_sum    = 0.
        
        for k in range(i*K,(i+1)*K):
            psf_tmp    = psf[int(k-i*K)]

            rhoB_tmp    = L_to_M_b*all_rhoB[k]
            rhoD1_tmp   = L_to_M_d1*all_rhoD1[k]
            
            v_tmp      = all_v[k]
            v2_tmp     = all_v2[k] 
            sigma2_tmp = all_sigma2[k] 
            
            
            psf_sum    = psf_sum+psf_tmp
            
            rhoB_sum    = rhoB_sum+rhoB_tmp*psf_tmp
            rhoD1_sum   = rhoD1_sum+rhoD1_tmp*psf_tmp 
            rhotot_sum  = rhotot_sum + (rhoB_tmp+rhoD1_tmp)*psf_tmp 
            
            vD1_sum   = vD1_sum+psf_tmp*rhoD1_tmp*v_tmp
            
            vD1_2_sum   = vD1_2_sum+psf_tmp*rhoD1_tmp*v2_tmp
            vB_2_sum    = vB_2_sum+psf_tmp*rhoB_tmp*sigma2_tmp

        rhoB_mean  = rhoB_sum/psf_sum
        rhoD1_mean = rhoD1_sum/psf_sum
        rho_AVG[i] = rhotot_sum/psf_sum
        vD1_mean        = k_D1*vD1_sum/rhoD1_sum

        sigmaD1_2_mean = (1.0-k_D1*k_D1)*vD1_2_sum/rhoD1_sum
        sigmaB2_mean   = vB_2_sum/rhoB_sum
        

        v_AVG[i]     = rhoD1_mean*vD1_mean/(rhoD1_mean+rhoB_mean) 
        sigma_AVG[i] = sqrt((rhoD1_mean*sigmaD1_2_mean/3.0 + rhoB_mean*sigmaB2_mean)/(rhoB_mean+rhoD1_mean))

    else:
        pass



@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32,float32[::1])',
          fastmath=True,
          debug = False)
def avg_LM_2_vis_components(all_rhoB,        # (N,J,K) contains all bulge surface density
                          all_rhoD1,        # (N,J,K) contains all disc surface density
                          psf,             # (N,J,K) psf weights
                          L_to_M_bulge,
                          L_to_M_disc1,
                          LM_AVG):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute avg light to mass in the case of 2 visible components (DECIDE NOTATION)
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                psf(1D arr float32)         : psf weight (of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
        Returns:
                LM_AVG(1D arr float32)   : average light to mass ratio
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < LM_AVG.shape[0]):
        K = psf.shape[0]

        rho_tot = 0.
        LM_sum = 0.
        for k in range(i*K,(i+1)*K):
            rhoD1_tmp = all_rhoD1[k]
            rhoB_tmp  = all_rhoB[k]
            psf_tmp   = psf[int(k-i*K)]
            rho_tot   = rho_tot+psf_tmp*(rhoD1_tmp+rhoB_tmp)
            LM_sum    = LM_sum+psf_tmp*(rhoB_tmp*L_to_M_bulge+rhoD1_tmp*L_to_M_disc1)

        LM_AVG[i] =  LM_sum/rho_tot

    else:
        pass

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
        #MODEL PSF only BULGE+"HALO" which up to now does not do anything
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_rho_v_only_bulge(all_rhoB,         # (N,J,K) contains all bulge surface density
                        all_sigma2,
                        psf,              # (N,J,K) psf weights
                        L_to_M_b,
                        rho_AVG,
                        v_AVG,
                        sigma_AVG):


    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute (for model with 2 visible components only):
            - total surface brightness
            - mean line of sight velocity
            - mean line of sight velocity dispersion
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe sigma)
                psf(1D arr float32)         : psf weight (of lenght K)
                L_to_M_b(float32)           : light to mass bulge
        Returns:
                rho_AVG(1D arr float32)     : total surface brightness
                v_AVG(1D arr float32)       : average line of sight velocity
                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        K = psf.shape[0]
        
        psf_sum = 0.

        rhoB_sum = 0.
        vB_2_sum = 0.
        
        for k in range(i*K,(i+1)*K):
            psf_tmp    = psf[int(k-i*K)]

            rhoB_tmp    = L_to_M_b*all_rhoB[k]
            
            sigma2_tmp = all_sigma2[k] 
            
            
            psf_sum    = psf_sum+psf_tmp
            
            rhoB_sum    = rhoB_sum+rhoB_tmp*psf_tmp
            
            vB_2_sum    = vB_2_sum+psf_tmp*rhoB_tmp*sigma2_tmp

        rhoB_mean  = rhoB_sum/psf_sum
        rho_AVG[i] = rhoB_mean

        sigmaB2_mean   = vB_2_sum/rhoB_sum
        

        v_AVG[i]     = 0.
        sigma_AVG[i] = sqrt(sigmaB2_mean)

    else:
        pass



@cuda.jit('void(float32[::1],float32[::1],float32,float32[::1])',
          fastmath=True,
          debug = False)
def avg_LM_only_bulge(all_rhoB,        # (N,J,K) contains all bulge surface density
                      psf,             # (N,J,K) psf weights
                      L_to_M_bulge,
                      LM_AVG):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute avg light to mass in the case of 2 visible components (DECIDE NOTATION)
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                psf(1D arr float32)         : psf weight (of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
        Returns:
                LM_AVG(1D arr float32)   : average light to mass ratio
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < LM_AVG.shape[0]):
        LM_AVG[i] =  L_to_M_bulge

    else:
        pass

@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1])',
          fastmath = True,
          debug = False)
def sum_likelihood(rho_model,
                   v_model,
                   sigma_model,
                   LM_model,
                   ibrightness_data,
                   v_data,
                   sigma_data,
                   ML_data,
                   ibrightness_err,
                   v_err,
                   sigma_err,
                   ML_err,
                   sys_rho,
                   likelihood):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute likelihood
                                             
        Parameters:
                rho_model(1D arr float32)        : model brightness (see avg_rho_v)
                v_model(1D arr float32)          : model los velocity (see avg_rho_v)  
                sigma_model(1D arr float32)      : model los dispersion (see avg_rho_v)
                LM_model(1D arr float32)         : model avg LM ratio (see avg_LM)
                ibrightness_data(1D arr float32) : data brightness
                v_data(1D arr float32)           : velocity data
                sigma_data(1D arr float32)       : sigma data
                ML_data(1D arr float32)          : light to mass data
                ibrightness_err(1D arr float32)  : brightness errror
                v_err(1D arr float32)            : velocity error
                sigma_err(1D arr float32)        : velocity dispersion error
                ML_err(1D arr float32)           : light to mass error
                sys_rho(float32)                 : systematic error for surface brightness (it is added as error on logarithm)
        Returns:
                likelihood(1D arr float32)       : likelihood (it is a vector of one element)            
    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    likelihood[0] = 0.
    if i < rho_model.shape[0]:
        if isnan(ibrightness_data[i]) == True:
            rho_lk = 0.
        else:
            rho_err_tmp  = ibrightness_err[i]
            rho_err_tmp  = rho_err_tmp*rho_err_tmp + (sys_rho/10**ibrightness_data[i]/log(10))*(sys_rho/10**ibrightness_data[i]/log(10))
            delta_rho    = ibrightness_data[i]-log10(rho_model[i])
            rho_lk       = (delta_rho*delta_rho)/rho_err_tmp+log(2.0*M_PI*rho_err_tmp)
        
        if isnan(v_data[i]) == True:
            v_lk  = 0.
        else:
            v_err_tmp = v_err[i]
            delta_v   = v_data[i]-v_model[i]
            v_err_tmp = v_err_tmp*v_err_tmp
            v_lk      = delta_v*delta_v/v_err_tmp+log(2.0*M_PI*v_err_tmp)

        if isnan(sigma_data[i]) == True:
            sigma_lk = 0.
        else:
            sigma_err_tmp = sigma_err[i]
            delta_sigma = sigma_data[i]-sigma_model[i]
            sigma_err_tmp = sigma_err_tmp*sigma_err_tmp
            sigma_lk = delta_sigma*delta_sigma/sigma_err_tmp+log(2.0*M_PI*sigma_err_tmp)

        if isnan(ML_data[i]) == True:
            ML_lk = 0.
        else:
            ML_err_tmp = ML_err[i]
            ML_err_tmp = ML_err_tmp*ML_err_tmp
            delta_LM = ML_data[i]-(1./LM_model[i])
            ML_lk = delta_LM*delta_LM/ML_err_tmp+log(2*M_PI*ML_err_tmp)

        tmp = -0.5*(rho_lk+v_lk+sigma_lk+ML_lk)
        cuda.atomic.add(likelihood,0,tmp)
    else:
        pass

def kernels_call_no_smooth(threadsperblock_1,
                           threadsperblock_2,
                           threadsperblock_3,
                           blockpergrid_1,
                           blockpergrid_2,
                           blockpergrid_3,
                           r_proj_device,
                           r_true_device,
                           phi_device,
                           all_rhoB_device,
                           all_rhoD1_device,
                           all_rhoD2_device,
                           all_v_device,
                           all_v2_device,
                           all_sigma2_device,
                           Xs_device,
                           v_bulge_device,
                           v_halo_device,
                           v_exp1_device,
                           v_exp2_device,
                           rho_avg_device,
                           v_avg_device,
                           sigma_avg_device,
                           LM_avg_device,
                           x_device,
                           y_device,
                           ibrightness_data_device,
                           v_data_device,
                           sigma_data_device,
                           logML_data_device,
                           ibrightness_error_device,
                           v_error_device,
                           sigma_error_device,
                           logML_error_device,
                           psf_device,
                           Mb,
                           Rb,
                           Md1,
                           Rd1,
                           Md2,
                           Rd2,
                           Mh,
                           Rh,
                           xcm,
                           ycm,
                           theta,
                           incl,
                           L_to_M_b,
                           L_to_M_d1,
                           L_to_M_d2,
                           k_D1,
                           k_D2,
                           sys_rho,
                           lk_device):
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Main function of the code, it calls all the kernels serially and return the likelihood
        
        Arrays allocated on device(GPU):  
                        r_proj_device(1D arr float32)            : projected radius
                        r_true_device(1D arr float32)            : 3D radius
                        phi_device(1D arr float32)               : azimuthal angle on disc plane
                        all_rhoB_device(1D arr float32)          : bulge surface densities
                        all_rhoD1_device(1D arr float32)         : disc1 surface densities
                        all_rhoD2_device(1D arr float32)         : disc2 surface densities
                        all_v_device(1D arr float32)             : line of sight velocities
                        all_v2_device(1D arr float32)            : total circular velocity
                        all_sigma2_device(1D arr float32)        : bulge dispersion
                        Xs_device(1D arr float32)                : X(s) Herquinst/Jaffe function
                        v_bulge_device(1D arr float32)           : bulge circular velocity squared
                        v_halo_device(1D arr float32)            : halo circular velocity squared
                        v_exp1_device(1D arr float32)            : disc1 circular velocity squared
                        v_exp2_device(1D arr float32)            : disc2 circular velocity squared
                        rho_avg_device(1D arr float32)           : total surface brightness
                        v_avg_device(1D arr float32)             : average los velocity
                        sigma_avg_device(1D arr float32)         : average los dispersion
                        LM_avg_device(1D arr float32)            : average Light to mass ratio
                        x_device(1D arr float32)                 : x refiuned grid (kpc)
                        y_device(1D arr float32)                 : y refined grid (kpc)
                        ibrightness_data_device(1D arr float32)  : surface brightness data (in log)
                        v_data_device(1D arr float32)            : velocity data
                        sigma_data_device(1D arr float32)        : dispersion data
                        logML_data_device(1D arr float32)           : light to mass data
                        ibrightness_error_device(1D arr float32) : log brightness error
                        v_error_device(1D arr float32)           : velocity error
                        sigma_error_device(1D arr float32)       : dispersion error
                        logML_error_device(1D arr float32)          : light to mass error
                        psf_device(1D arr float32)               : psf weight

        Performance tuning parameters:
                    threadsperblock_1(int16) : threads per block in N X J X K vectors
                    threadsperblock_2(int16) : threads per block in N X J X K vectors
                    threadsperblock_3(int16) : threads per block in N X J vectors
                    blockpergrid_1(int16)    : blocks per grid  in N X J X K vectors
                    blockpergrid_2(int16)    : blocks per grid  in N X J X K vectors
                    blockpergrid_3(int16)    : blocks per grid  in N X J  vectors

        Free Parameters:
                    Mb(float32)        : bulge mass in log10
                    Rb(float32)        : bulge scale radius in log10
                    Md1(float32)       : disc1 mass in log10
                    Rd1(float32)       : disc1 scale radius in log10
                    Md2(float32)       : disc2 mass in log10
                    Rd2(float32)       : disc2 scale radius in log10
                    Mh(float32)        : halo mass in log10
                    Rh(float32)        : halo scale radius in log10
                    x0(float32)        : x center position in kpc
                    y0(float32)        : y center position in kpc
                    theta(float32)     : P.A. angle in rad
                    incl(float32)      : incl angle in rad
                    L_to_M_b(float32)  : light to mass bulge 
                    L_to_M_d1(float32) : light to mass disc1
                    L_to_M_d2(float32) : light to mass disc2
                    k_D1(float32)      : kinematic decoposition params for disc1
                    k_D2(float32)      : kinematic decoposition params for disc2
                    sys_rho(float32)   : systematic error for surface brightness (should be optional)
                
        Returns:
                lk_device(float32)     : likelihood           
    '''
    
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
    coordinate_transformation[blockpergrid_2,threadsperblock_2](x_device,
                                                                y_device,
                                                                xcm,
                                                                ycm,
                                                                theta,
                                                                incl,
                                                                r_proj_device,
                                                                r_true_device,
                                                                phi_device)


    Xfunction[blockpergrid_2,threadsperblock_2](Rb,r_proj_device,Xs_device)
    ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
    herquinst_rho[blockpergrid_1,threadsperblock_1](Mb,
                                       Rb,
                                       r_proj_device,
                                       all_rhoB_device,
                                       Xs_device)
    # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
    herquinst_sigma[blockpergrid_2,threadsperblock_2](Mb,
                                                      Rb,
                                                      r_proj_device,
                                                      all_sigma2_device,
                                                      Xs_device)

    ## DISC SURFACE DENSITY
    rho_D[blockpergrid_2,threadsperblock_2](Md1,
                                            Rd1,
                                            incl,
                                            r_true_device,
                                            all_rhoD1_device)

    rho_D[blockpergrid_2,threadsperblock_2](Md2,
                                            Rd2,
                                            incl,
                                            r_true_device,
                                            all_rhoD2_device)
    
    ## DISC VELOCITY 
    v_H[blockpergrid_1,threadsperblock_1](Mh,
                                          Rh,
                                          r_true_device,
                                          v_halo_device)
    v_H[blockpergrid_1,threadsperblock_1](Mb,
                                          Rb,
                                          r_true_device,
                                          v_bulge_device)
    v_D[blockpergrid_1,threadsperblock_1](Md1,
                                          Rd1,
                                          r_true_device,
                                          v_exp1_device)
    v_D[blockpergrid_1,threadsperblock_1](Md2,
                                          Rd2,
                                          r_true_device,
                                          v_exp2_device)
    v_tot[blockpergrid_1,threadsperblock_1](v_exp1_device,
                                            v_exp2_device,
                                            v_bulge_device,
                                            v_halo_device,
                                            incl,
                                            phi_device,
                                            all_v_device,
                                            all_v2_device)

    #  RHO, V E SIGMA with psf
    avg_rho_v[blockpergrid_3,threadsperblock_3](all_rhoB_device,
                                                all_rhoD1_device,
                                                all_rhoD2_device,
                                                all_v_device,
                                                all_v2_device,
                                                all_sigma2_device,
                                                psf_device,
                                                L_to_M_b,
                                                L_to_M_d1,
                                                L_to_M_d2,
                                                #k_B,
                                                k_D1,
                                                k_D2,
                                                rho_avg_device,
                                                v_avg_device,
                                                sigma_avg_device)
    


    avg_LM[blockpergrid_3,threadsperblock_3](all_rhoB_device,
                                                all_rhoD1_device,
                                                all_rhoD2_device,
                                                psf_device,
                                                L_to_M_b,
                                                L_to_M_d1,
                                                L_to_M_d2,
                                                LM_avg_device)


    sum_likelihood[blockpergrid_3,threadsperblock_3](rho_avg_device,
                                                     v_avg_device,
                                                     sigma_avg_device,
                                                     LM_avg_device,
                                                     ibrightness_data_device,
                                                     v_data_device,
                                                     sigma_data_device,
                                                     logML_data_device,
                                                     ibrightness_error_device,
                                                     v_error_device,
                                                     sigma_error_device,
                                                     logML_error_device,
                                                     sys_rho,
                                                     lk_device)
    
    lk = lk_device.copy_to_host()

    return lk[0]


def model_evaluation(threadsperblock_1,
                     threadsperblock_2,
                     threadsperblock_3,
                     blockpergrid_1,
                     blockpergrid_2,
                     blockpergrid_3,
                     r_proj_device,
                     r_true_device,
                     phi_device,
                     all_rhoB_device,
                     all_rhoD1_device,
                     all_rhoD2_device,
                     all_v_device,
                     all_v2_device,
                     all_sigma2_device,
                     Xs_device,
                     v_bulge_device,
                     v_halo_device,
                     v_exp1_device,
                     v_exp2_device,
                     rho_avg_device,
                     v_avg_device,
                     sigma_avg_device,
                     LM_avg_device,
                     x_device,
                     y_device,
                     psf_device,
                     Mb,
                     Rb,
                     Md1,
                     Rd1,
                     Md2,
                     Rd2,
                     Mh,
                     Rh,
                     xcm,
                     ycm,
                     theta,
                     incl,
                     L_to_M_b,
                     L_to_M_d1,
                     L_to_M_d2,
                     k_D1,
                     k_D2):
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Main function of the code, it calls all the kernels serially and return the likelihood
        
        Arrays allocated on device(GPU):  
                        r_proj_device(1D arr float32)            : projected radius
                        r_true_device(1D arr float32)            : 3D radius
                        phi_device(1D arr float32)               : azimuthal angle on disc plane
                        all_rhoB_device(1D arr float32)          : bulge surface densities
                        all_rhoD1_device(1D arr float32)         : disc1 surface densities
                        all_rhoD2_device(1D arr float32)         : disc2 surface densities
                        all_v_device(1D arr float32)             : line of sight velocities
                        all_v2_device(1D arr float32)            : total circular velocity
                        all_sigma2_device(1D arr float32)        : bulge dispersion
                        Xs_device(1D arr float32)                : X(s) Herquinst/Jaffe function
                        v_bulge_device(1D arr float32)           : bulge circular velocity squared
                        v_halo_device(1D arr float32)            : halo circular velocity squared
                        v_exp1_device(1D arr float32)            : disc1 circular velocity squared
                        v_exp2_device(1D arr float32)            : disc2 circular velocity squared
                        rho_avg_device(1D arr float32)           : total surface brightness
                        v_avg_device(1D arr float32)             : average los velocity
                        sigma_avg_device(1D arr float32)         : average los dispersion
                        LM_avg_device(1D arr float32)            : average Light to mass ratio
                        x_device(1D arr float32)                 : x refiuned grid (kpc)
                        y_device(1D arr float32)                 : y refined grid (kpc)
                        ibrightness_data_device(1D arr float32)  : surface brightness data (in log)
                        v_data_device(1D arr float32)            : velocity data
                        sigma_data_device(1D arr float32)        : dispersion data
                        logML_data_device(1D arr float32)           : light to mass data
                        ibrightness_error_device(1D arr float32) : log brightness error
                        v_error_device(1D arr float32)           : velocity error
                        sigma_error_device(1D arr float32)       : dispersion error
                        logML_error_device(1D arr float32)          : light to mass error
                        psf_device(1D arr float32)               : psf weight

        Performance tuning parameters:
                    threadsperblock_1(int16) : threads per block in N X J X K vectors
                    threadsperblock_2(int16) : threads per block in N X J X K vectors
                    threadsperblock_3(int16) : threads per block in N X J vectors
                    blockpergrid_1(int16)    : blocks per grid  in N X J X K vectors
                    blockpergrid_2(int16)    : blocks per grid  in N X J X K vectors
                    blockpergrid_3(int16)    : blocks per grid  in N X J  vectors

        Free Parameters:
                    Mb(float32)        : bulge mass in log10
                    Rb(float32)        : bulge scale radius in log10
                    Md1(float32)       : disc1 mass in log10
                    Rd1(float32)       : disc1 scale radius in log10
                    Md2(float32)       : disc2 mass in log10
                    Rd2(float32)       : disc2 scale radius in log10
                    Mh(float32)        : halo mass in log10
                    Rh(float32)        : halo scale radius in log10
                    x0(float32)        : x center position in kpc
                    y0(float32)        : y center position in kpc
                    theta(float32)     : P.A. angle in rad
                    incl(float32)      : incl angle in rad
                    L_to_M_b(float32)  : light to mass bulge 
                    L_to_M_d1(float32) : light to mass disc1
                    L_to_M_d2(float32) : light to mass disc2
                    k_D1(float32)      : kinematic decoposition params for disc1
                    k_D2(float32)      : kinematic decoposition params for disc2
                
        Returns:
                rho(1D arr float32)     : surface brightness
                v(1D arr float32)       : los velocity
                sigma(1D arr float32)   : los velocity dispersion
                LM(1D arr float32)      : avg light to mass           
    '''
    
    Mb           = 10**Mb
    Md1          = 10**Md1    
    Md2          = 10**Md2    
    Mh           = 10**Mh    
    Rb           = 10**Rb    
    Rd1          = 10**Rd1    
    Rd2          = 10**Rd2    
    Rh           = 10**Rh
    # GRID DEFINITION
    coordinate_transformation[blockpergrid_2,threadsperblock_2](x_device,
                                                                y_device,
                                                                xcm,
                                                                ycm,
                                                                theta,
                                                                incl,
                                                                r_proj_device,
                                                                r_true_device,
                                                                phi_device)


    Xfunction[blockpergrid_2,threadsperblock_2](Rb,r_proj_device,Xs_device)
    ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
    herquinst_rho[blockpergrid_1,threadsperblock_1](Mb,
                                                    Rb,
                                                    r_proj_device,
                                                    all_rhoB_device,
                                                    Xs_device)
    # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
    herquinst_sigma[blockpergrid_2,threadsperblock_2](Mb,
                                                      Rb,
                                                      r_proj_device,
                                                      all_sigma2_device,
                                                      Xs_device)

    ## DISC SURFACE DENSITY
    rho_D[blockpergrid_2,threadsperblock_2](Md1,
                                            Rd1,
                                            incl,
                                            r_true_device,
                                            all_rhoD1_device)

    rho_D[blockpergrid_2,threadsperblock_2](Md2,
                                            Rd2,
                                            incl,
                                            r_true_device,
                                            all_rhoD2_device)
    
    ## DISC VELOCITY 
    v_H[blockpergrid_1,threadsperblock_1](Mh,
                                          Rh,
                                          r_true_device,
                                          v_halo_device)
    v_H[blockpergrid_1,threadsperblock_1](Mb,
                                          Rb,
                                          r_true_device,
                                          v_bulge_device)
    v_D[blockpergrid_1,threadsperblock_1](Md1,
                                          Rd1,
                                          r_true_device,
                                          v_exp1_device)
    v_D[blockpergrid_1,threadsperblock_1](Md2,
                                          Rd2,
                                          r_true_device,
                                          v_exp2_device)
    v_tot[blockpergrid_1,threadsperblock_1](v_exp1_device,
                                            v_exp2_device,
                                            v_bulge_device,
                                            v_halo_device,
                                            incl,
                                            phi_device,
                                            all_v_device,
                                            all_v2_device)

    #  RHO, V E SIGMA with psf
    avg_rho_v[blockpergrid_3,threadsperblock_3](all_rhoB_device,
                                                all_rhoD1_device,
                                                all_rhoD2_device,
                                                all_v_device,
                                                all_v2_device,
                                                all_sigma2_device,
                                                psf_device,
                                                L_to_M_b,
                                                L_to_M_d1,
                                                L_to_M_d2,
                                                #k_B,
                                                k_D1,
                                                k_D2,
                                                rho_avg_device,
                                                v_avg_device,
                                                sigma_avg_device)
    


    avg_LM [blockpergrid_3,threadsperblock_3](all_rhoB_device,
                                                all_rhoD1_device,
                                                all_rhoD2_device,
                                                psf_device,
                                                L_to_M_b,
                                                L_to_M_d1,
                                                L_to_M_d2,
                                                LM_avg_device)

    rho   = rho_avg_device.copy_to_host()
    v     = v_avg_device.copy_to_host()
    sigma = sigma_avg_device.copy_to_host()
    LM    = LM_avg_device.copy_to_host()

    return rho,v,sigma,LM



def kernels_call_no_smooth_BH(threadsperblock_1,
                           threadsperblock_2,
                           threadsperblock_3,
                           blockpergrid_1,
                           blockpergrid_2,
                           blockpergrid_3,
                           r_proj_device,
                           r_true_device,
                           phi_device,
                           all_rhoB_device,
                           all_rhoD1_device,
                           all_rhoD2_device,
                           all_v_device,
                           all_v2_device,
                           all_sigma2_device,
                           Xs_device,
                           X1_dot_device,
                           X2_dot_device,
                           X3_dot_device,
                           Y_1_device,
                           Y_2_device,
                           v_BH_device,
                           v_bulge_device,
                           v_halo_device,
                           v_exp1_device,
                           v_exp2_device,
                           rho_avg_device,
                           v_avg_device,
                           sigma_avg_device,
                           LM_avg_device,
                           x_device,
                           y_device,
                           ibrightness_data_device,
                           v_data_device,
                           sigma_data_device,
                           logML_data_device,
                           ibrightness_error_device,
                           v_error_device,
                           sigma_error_device,
                           logML_error_device,
                           psf_device,
                           Mbh,
                           Mb,
                           Rb,
                           Md1,
                           Rd1,
                           Md2,
                           Rd2,
                           Mh,
                           Rh,
                           xcm,
                           ycm,
                           theta,
                           incl,
                           L_to_M_b,
                           L_to_M_d1,
                           L_to_M_d2,
                           k_D1,
                           k_D2,
                           sys_rho,
                           lk_device):
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Main function of the code, it calls all the kernels serially and return the likelihood
        
        Arrays allocated on device(GPU):  
                        r_proj_device(1D arr float32)            : projected radius
                        r_true_device(1D arr float32)            : 3D radius
                        phi_device(1D arr float32)               : azimuthal angle on disc plane
                        all_rhoB_device(1D arr float32)          : bulge surface densities
                        all_rhoD1_device(1D arr float32)         : disc1 surface densities
                        all_rhoD2_device(1D arr float32)         : disc2 surface densities
                        all_v_device(1D arr float32)             : line of sight velocities
                        all_v2_device(1D arr float32)            : total circular velocity
                        all_sigma2_device(1D arr float32)        : bulge dispersion
                        Xs_device(1D arr float32)                : X(s) Herquinst/Jaffe function
                        X1_dot_device(1D arr float32)            : first derivative of X(s) function
                        X2_dot_device(1D arr float32)            : second derivative of X(s) function
                        X3_dot_device(1D arr float32)            : third derivative of X(s) function
                        Y_1_device(1D arr float32)               : auxiliar function for BH+bulge sigma
                        Y_2_device(1D arr float32)               : auxiliar function for BH+bulge sigma
                        v_BH_device(1D arr float32)              : BH velocity squared
                        v_bulge_device(1D arr float32)           : bulge circular velocity squared
                        v_halo_device(1D arr float32)            : halo circular velocity squared
                        v_exp1_device(1D arr float32)            : disc1 circular velocity squared
                        v_exp2_device(1D arr float32)            : disc2 circular velocity squared
                        rho_avg_device(1D arr float32)           : total surface brightness
                        v_avg_device(1D arr float32)             : average los velocity
                        sigma_avg_device(1D arr float32)         : average los dispersion
                        LM_avg_device(1D arr float32)            : average Light to mass ratio
                        x_device(1D arr float32)                 : x refiuned grid (kpc)
                        y_device(1D arr float32)                 : y refined grid (kpc)
                        ibrightness_data_device(1D arr float32)  : surface brightness data (in log)
                        v_data_device(1D arr float32)            : velocity data
                        sigma_data_device(1D arr float32)        : dispersion data
                        logML_data_device(1D arr float32)           : light to mass data
                        ibrightness_error_device(1D arr float32) : log brightness error
                        v_error_device(1D arr float32)           : velocity error
                        sigma_error_device(1D arr float32)       : dispersion error
                        logML_error_device(1D arr float32)          : light to mass error
                        psf_device(1D arr float32)               : psf weight

        Performance tuning parameters:
                    threadsperblock_1(int16) : threads per block in N X J X K vectors
                    threadsperblock_2(int16) : threads per block in N X J X K vectors
                    threadsperblock_3(int16) : threads per block in N X J vectors
                    blockpergrid_1(int16)    : blocks per grid  in N X J X K vectors
                    blockpergrid_2(int16)    : blocks per grid  in N X J X K vectors
                    blockpergrid_3(int16)    : blocks per grid  in N X J  vectors

        Free Parameters:
                    Mbh(float32)       : BH mass in log10
                    Mb(float32)        : bulge mass in log10
                    Rb(float32)        : bulge scale radius in log10
                    Md1(float32)       : disc1 mass in log10
                    Rd1(float32)       : disc1 scale radius in log10
                    Md2(float32)       : disc2 mass in log10
                    Rd2(float32)       : disc2 scale radius in log10
                    Mh(float32)        : halo mass in log10
                    Rh(float32)        : halo scale radius in log10
                    x0(float32)        : x center position in kpc
                    y0(float32)        : y center position in kpc
                    theta(float32)     : P.A. angle in rad
                    incl(float32)      : incl angle in rad
                    L_to_M_b(float32)  : light to mass bulge 
                    L_to_M_d1(float32) : light to mass disc1
                    L_to_M_d2(float32) : light to mass disc2
                    k_D1(float32)      : kinematic decoposition params for disc1
                    k_D2(float32)      : kinematic decoposition params for disc2
                    sys_rho(float32)   : systematic error for surface brightness (should be optional)
                
        Returns:
                lk_device(float32)     : likelihood           
    '''
    
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
    coordinate_transformation[blockpergrid_2,threadsperblock_2](x_device,
                                                                y_device,
                                                                xcm,
                                                                ycm,
                                                                theta,
                                                                incl,
                                                                r_proj_device,
                                                                r_true_device,
                                                                phi_device)


    Xfunction[blockpergrid_2,threadsperblock_2](Rb,r_proj_device,Xs_device)
    all_Xdot[blockpergrid_2,threadsperblock_2](Rb,
                                               r_proj_device,
                                               Xs_device,
                                               X1_dot_device,
                                               X2_dot_device,
                                               X3_dot_device)
    Y1[blockpergrid_2,threadsperblock_2](Rb,
                                         r_proj_device,
                                         Xs_device,
                                         X1_dot_device,
                                         Y_1_device)
    Y2[blockpergrid_2,threadsperblock_2](Rb,
                                         r_proj_device,
                                         Xs_device,
                                         X1_dot_device,
                                         X2_dot_device,
                                         X3_dot_device,
                                         Y_2_device)
    ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
    herquinst_rho[blockpergrid_1,threadsperblock_1](Mb,
                                       Rb,
                                       r_proj_device,
                                       all_rhoB_device,
                                       Xs_device)
    # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
    herquinst_sigma_BH[blockpergrid_2,threadsperblock_2](Mbh,
                                                         Mb,
                                                         Rb,
                                                         r_proj_device,
                                                         Y_1_device,
                                                         Y_2_device,
                                                         all_rhoB_device,
                                                         all_sigma2_device)

    ## DISC SURFACE DENSITY
    rho_D[blockpergrid_2,threadsperblock_2](Md1,
                                            Rd1,
                                            incl,
                                            r_true_device,
                                            all_rhoD1_device)

    rho_D[blockpergrid_2,threadsperblock_2](Md2,
                                            Rd2,
                                            incl,
                                            r_true_device,
                                            all_rhoD2_device)
    
    ## DISC VELOCITY 
    v_H[blockpergrid_1,threadsperblock_1](Mh,
                                          Rh,
                                          r_true_device,
                                          v_halo_device)
    v_H[blockpergrid_1,threadsperblock_1](Mb,
                                          Rb,
                                          r_true_device,
                                          v_bulge_device)
    v_D[blockpergrid_1,threadsperblock_1](Md1,
                                          Rd1,
                                          r_true_device,
                                          v_exp1_device)
    v_D[blockpergrid_1,threadsperblock_1](Md2,
                                          Rd2,
                                          r_true_device,
                                          v_exp2_device)
    v_BH[blockpergrid_1,threadsperblock_1](Mbh,
                                          r_true_device,
                                          v_BH_device)
    v_tot_BH[blockpergrid_1,threadsperblock_1](v_exp1_device,
                                            v_exp2_device,
                                            v_bulge_device,
                                            v_halo_device,
                                            v_BH_device,
                                            incl,
                                            phi_device,
                                            all_v_device,
                                            all_v2_device)

    #  RHO, V E SIGMA with psf
    avg_rho_v[blockpergrid_3,threadsperblock_3](all_rhoB_device,
                                                all_rhoD1_device,
                                                all_rhoD2_device,
                                                all_v_device,
                                                all_v2_device,
                                                all_sigma2_device,
                                                psf_device,
                                                L_to_M_b,
                                                L_to_M_d1,
                                                L_to_M_d2,
                                                #k_B,
                                                k_D1,
                                                k_D2,
                                                rho_avg_device,
                                                v_avg_device,
                                                sigma_avg_device)
    


    avg_LM[blockpergrid_3,threadsperblock_3](all_rhoB_device,
                                                all_rhoD1_device,
                                                all_rhoD2_device,
                                                psf_device,
                                                L_to_M_b,
                                                L_to_M_d1,
                                                L_to_M_d2,
                                                LM_avg_device)


    sum_likelihood[blockpergrid_3,threadsperblock_3](rho_avg_device,
                                                     v_avg_device,
                                                     sigma_avg_device,
                                                     LM_avg_device,
                                                     ibrightness_data_device,
                                                     v_data_device,
                                                     sigma_data_device,
                                                     logML_data_device,
                                                     ibrightness_error_device,
                                                     v_error_device,
                                                     sigma_error_device,
                                                     logML_error_device,
                                                     sys_rho,
                                                     lk_device)
    
    lk = lk_device.copy_to_host()

    return lk[0]


def model_evaluation_BH(threadsperblock_1,
                     threadsperblock_2,
                     threadsperblock_3,
                     blockpergrid_1,
                     blockpergrid_2,
                     blockpergrid_3,
                     r_proj_device,
                     r_true_device,
                     phi_device,
                     all_rhoB_device,
                     all_rhoD1_device,
                     all_rhoD2_device,
                     all_v_device,
                     all_v2_device,
                     all_sigma2_device,
                     Xs_device,
                     X1_dot_device,
                     X2_dot_device,
                     X3_dot_device,
                     Y_1_device,
                     Y_2_device,
                     v_BH_device,
                     v_bulge_device,
                     v_halo_device,
                     v_exp1_device,
                     v_exp2_device,
                     rho_avg_device,
                     v_avg_device,
                     sigma_avg_device,
                     LM_avg_device,
                     x_device,
                     y_device,
                     psf_device,
                     Mbh,
                     Mb,
                     Rb,
                     Md1,
                     Rd1,
                     Md2,
                     Rd2,
                     Mh,
                     Rh,
                     xcm,
                     ycm,
                     theta,
                     incl,
                     L_to_M_b,
                     L_to_M_d1,
                     L_to_M_d2,
                     k_D1,
                     k_D2):
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Main function of the code, it calls all the kernels serially and return the likelihood
        
        Arrays allocated on device(GPU):  
                        r_proj_device(1D arr float32)            : projected radius
                        r_true_device(1D arr float32)            : 3D radius
                        phi_device(1D arr float32)               : azimuthal angle on disc plane
                        all_rhoB_device(1D arr float32)          : bulge surface densities
                        all_rhoD1_device(1D arr float32)         : disc1 surface densities
                        all_rhoD2_device(1D arr float32)         : disc2 surface densities
                        all_v_device(1D arr float32)             : line of sight velocities
                        all_v2_device(1D arr float32)            : total circular velocity
                        all_sigma2_device(1D arr float32)        : bulge dispersion
                        Xs_device(1D arr float32)                : X(s) Herquinst/Jaffe function
                        X1_dot_device(1D arr float32)            : first derivative of X(s) function
                        X2_dot_device(1D arr float32)            : second derivative of X(s) function
                        X3_dot_device(1D arr float32)            : third derivative of X(s) function
                        Y_1_device(1D arr float32)               : auxiliar function for BH+bulge sigma
                        Y_2_device(1D arr float32)               : auxiliar function for BH+bulge sigma
                        v_BH_device(1D arr float32)              : BH velocity squared
                        v_bulge_device(1D arr float32)           : bulge circular velocity squared
                        v_halo_device(1D arr float32)            : halo circular velocity squared
                        v_exp1_device(1D arr float32)            : disc1 circular velocity squared
                        v_exp2_device(1D arr float32)            : disc2 circular velocity squared
                        rho_avg_device(1D arr float32)           : total surface brightness
                        v_avg_device(1D arr float32)             : average los velocity
                        sigma_avg_device(1D arr float32)         : average los dispersion
                        LM_avg_device(1D arr float32)            : average Light to mass ratio
                        x_device(1D arr float32)                 : x refiuned grid (kpc)
                        y_device(1D arr float32)                 : y refined grid (kpc)
                        ibrightness_data_device(1D arr float32)  : surface brightness data (in log)
                        v_data_device(1D arr float32)            : velocity data
                        sigma_data_device(1D arr float32)        : dispersion data
                        logML_data_device(1D arr float32)           : light to mass data
                        ibrightness_error_device(1D arr float32) : log brightness error
                        v_error_device(1D arr float32)           : velocity error
                        sigma_error_device(1D arr float32)       : dispersion error
                        logML_error_device(1D arr float32)          : light to mass error
                        psf_device(1D arr float32)               : psf weight

        Performance tuning parameters:
                    threadsperblock_1(int16) : threads per block in N X J X K vectors
                    threadsperblock_2(int16) : threads per block in N X J X K vectors
                    threadsperblock_3(int16) : threads per block in N X J vectors
                    blockpergrid_1(int16)    : blocks per grid  in N X J X K vectors
                    blockpergrid_2(int16)    : blocks per grid  in N X J X K vectors
                    blockpergrid_3(int16)    : blocks per grid  in N X J  vectors

        Free Parameters:
                    Mbh(float32)       : BH mass in log10
                    Mb(float32)        : bulge mass in log10
                    Rb(float32)        : bulge scale radius in log10
                    Md1(float32)       : disc1 mass in log10
                    Rd1(float32)       : disc1 scale radius in log10
                    Md2(float32)       : disc2 mass in log10
                    Rd2(float32)       : disc2 scale radius in log10
                    Mh(float32)        : halo mass in log10
                    Rh(float32)        : halo scale radius in log10
                    x0(float32)        : x center position in kpc
                    y0(float32)        : y center position in kpc
                    theta(float32)     : P.A. angle in rad
                    incl(float32)      : incl angle in rad
                    L_to_M_b(float32)  : light to mass bulge 
                    L_to_M_d1(float32) : light to mass disc1
                    L_to_M_d2(float32) : light to mass disc2
                    k_D1(float32)      : kinematic decoposition params for disc1
                    k_D2(float32)      : kinematic decoposition params for disc2
                
        Returns:
                rho(1D arr float32)     : surface brightness
                v(1D arr float32)       : los velocity
                sigma(1D arr float32)   : los velocity dispersion
                LM(1D arr float32)      : avg light to mass           
    '''
    
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
    coordinate_transformation[blockpergrid_2,threadsperblock_2](x_device,
                                                                y_device,
                                                                xcm,
                                                                ycm,
                                                                theta,
                                                                incl,
                                                                r_proj_device,
                                                                r_true_device,
                                                                phi_device)


    Xfunction[blockpergrid_2,threadsperblock_2](Rb,r_proj_device,Xs_device)
    all_Xdot[blockpergrid_2,threadsperblock_2](Rb,
                                               r_proj_device,
                                               Xs_device,
                                               X1_dot_device,
                                               X2_dot_device,
                                               X3_dot_device)
    Y1[blockpergrid_2,threadsperblock_2](Rb,
                                         r_proj_device,
                                         Xs_device,
                                         X1_dot_device,
                                         Y_2_device)
    Y2[blockpergrid_2,threadsperblock_2](Rb,
                                         r_proj_device,
                                         Xs_device,
                                         X1_dot_device,
                                         X2_dot_device,
                                         X3_dot_device,
                                         Y_2_device)
    ## BULGE SURFACE DENSITY AND VELOCITY DISPERSION
    herquinst_rho[blockpergrid_1,threadsperblock_1](Mb,
                                                    Rb,
                                                    r_proj_device,
                                                    all_rhoB_device,
                                                    Xs_device)
    # BULGE SURFACE DENSITY AND VELOCITY DISPERSION
    herquinst_sigma_BH[blockpergrid_2,threadsperblock_2](Mbh,
                                                         Mb,
                                                         Rb,
                                                         r_proj_device,
                                                         Y_1_device,
                                                         Y_2_device,
                                                         all_rhoB_device,
                                                         all_sigma2_device)

    ## DISC SURFACE DENSITY
    rho_D[blockpergrid_2,threadsperblock_2](Md1,
                                            Rd1,
                                            incl,
                                            r_true_device,
                                            all_rhoD1_device)

    rho_D[blockpergrid_2,threadsperblock_2](Md2,
                                            Rd2,
                                            incl,
                                            r_true_device,
                                            all_rhoD2_device)
    
    ## DISC VELOCITY 
    v_H[blockpergrid_1,threadsperblock_1](Mh,
                                          Rh,
                                          r_true_device,
                                          v_halo_device)
    v_H[blockpergrid_1,threadsperblock_1](Mb,
                                          Rb,
                                          r_true_device,
                                          v_bulge_device)
    v_D[blockpergrid_1,threadsperblock_1](Md1,
                                          Rd1,
                                          r_true_device,
                                          v_exp1_device)
    v_D[blockpergrid_1,threadsperblock_1](Md2,
                                          Rd2,
                                          r_true_device,
                                          v_exp2_device)
    v_BH[blockpergrid_1,threadsperblock_1](Mbh,
                                          r_true_device,
                                          v_BH_device)
    v_tot_BH[blockpergrid_1,threadsperblock_1](v_exp1_device,
                                            v_exp2_device,
                                            v_bulge_device,
                                            v_halo_device,
                                            v_BH_device,
                                            incl,
                                            phi_device,
                                            all_v_device,
                                            all_v2_device)

    #  RHO, V E SIGMA with psf
    avg_rho_v[blockpergrid_3,threadsperblock_3](all_rhoB_device,
                                                all_rhoD1_device,
                                                all_rhoD2_device,
                                                all_v_device,
                                                all_v2_device,
                                                all_sigma2_device,
                                                psf_device,
                                                L_to_M_b,
                                                L_to_M_d1,
                                                L_to_M_d2,
                                                #k_B,
                                                k_D1,
                                                k_D2,
                                                rho_avg_device,
                                                v_avg_device,
                                                sigma_avg_device)
    


    avg_LM [blockpergrid_3,threadsperblock_3](all_rhoB_device,
                                                all_rhoD1_device,
                                                all_rhoD2_device,
                                                psf_device,
                                                L_to_M_b,
                                                L_to_M_d1,
                                                L_to_M_d2,
                                                LM_avg_device)

    rho   = rho_avg_device.copy_to_host()
    v     = v_avg_device.copy_to_host()
    sigma = sigma_avg_device.copy_to_host()
    LM    = LM_avg_device.copy_to_host()

    return rho,v,sigma,LM


if __name__ == "__main__":

    pass