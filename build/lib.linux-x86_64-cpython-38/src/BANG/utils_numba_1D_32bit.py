from numba import cuda,float64,float32,int16
import numpy as np 
from math import isnan
from math import pi as M_PI
from numba.cuda.libdevice import acosf as acos
from numba.cuda.libdevice import atan2f as atan2
from numba.cuda.libdevice import fast_cosf as cos
from numba.cuda.libdevice import fast_sinf as sin
from numba.cuda.libdevice import fast_expf as exp
from numba.cuda.libdevice import fast_logf as log
from numba.cuda.libdevice import fast_log10f as log10
from numba.cuda.libdevice import sqrtf as sqrt
from numba.cuda.libdevice import fast_exp10f
from numba.cuda.libdevice import fadd_rn,fast_fdividef,fsub_rn,hypotf,fmul_rn,fmaf_rn,frcp_rn,fast_powf


'''
It contains all the relevant function of the code written for GPU usage.
Approximation with fastmast operations.
'''


'''
Units are:
            - M_sun/kpc^2 for surface densities
            - km/s for velocities and dispersions

'''

#---------------------------------------------------------------------------
                    #GENERAL FUNCTIONS
#---------------------------------------------------------------------------
@cuda.jit('float32(float32,float32,float32,float32,float32,\
                   float32,float32,float32,float32,float32,)',
          device=True,
          inline=True,
          fastmath=True) 
def my_bilinear(x1,x2,y1,y2,f11,f12,f21,f22,x,y):
    '''
    DEVICE FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Bilinear interpolation.

        Parameters:
                x1(float32): x cartesian coordinate first point
                x2(float32): x cartesian coordinate second point
                y1(float32): y cartesian coordinate first point
                y2(float32): y cartesian coordinate second point
                f11(float32): function value at (x1,y1)
                f12(float32): function value at (x1,y2)
                f21(float32): function value at (x2,y1)
                f22(float32): function value at (x2,y2)
        Returns:
                out(float32): bilinear interpolation 
    '''
    den = (x2-x1)*(y2-y1)
    num = f11*(x2-x)*(y2-y)+f21*(x-x1)*(y2-y)+\
          f12*(x2-x)*(y-y1)+f22*(x-x1)*(y-y1)
    return num/den



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
        y=fast_fdividef(x,3.75)
        y=fmul_rn(y,y)
        ans= fmul_rn(exp(-x),fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,0.45813e-2,0.360768e-1),0.2659732),1.2067492),3.0899424),3.5156229),1.0))
    else: 
        y=fast_fdividef(3.75,x)
        ans= fast_fdividef(fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,0.392377e-2,-0.1647633e-1),0.2635537e-1),-0.2057706e-1),0.916281e-2),-0.157565e-2),0.225319e-2),0.1328592e-1),0.39894228),sqrt(x))
        
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
        y=fast_fdividef(fmul_rn(x,x),4.0)
        ans= fmul_rn(exp(x),fadd_rn(fmul_rn(-log(fast_fdividef(x,2.0)),fmul_rn(exp(x),i0e)),fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,0.74e-5,0.10750e-3),0.262698e-2),0.3488590e-1),0.23069756),0.42278420),-0.57721566)))
    else :
        y=fast_fdividef(2.0,x)
        ans= fast_fdividef(fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,0.53208e-3,-0.251540e-2),0.587872e-2),-0.1062446e-1),0.2189568e-1),-0.7832358e-1),1.25331414),sqrt(x))
        
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
        y=fast_fdividef(x,3.75)
        y=fmul_rn(y,y)
        ans= fmul_rn(exp(-x),fmul_rn(x,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,0.32411e-3,0.301532e-2),0.2658733e-1),0.15084934),0.51498869),0.87890594),0.5)))
    else :
        y=fast_fdividef(3.75,x)
        ans=fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,-0.420059e-2,0.1787654e-1),-0.2895312e-1),0.2282967e-1)
        ans= fast_fdividef(fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(ans,y,-0.1031555e-1),0.163801e-2),-0.362018e-2),-0.3988024e-1),0.39894228),sqrt(x))

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
        y = fast_fdividef(fmul_rn(x,x),4.0)
        ans= fmul_rn(exp(x),fadd_rn(fmul_rn(log(fast_fdividef(x,2.0)),fmul_rn(exp(x),i1e)),fast_fdividef(fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,-0.4686e-4,-0.110404e-2),-0.1919402e-1),-0.18156897),-0.67278579),0.15443144),1.0),x)))
    else:
        y=2.0/x
        y = fast_fdividef(2.0,x)
        ans=fast_fdividef(fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,fmaf_rn(y,-0.68245e-3,0.325614e-2),-0.780353e-2),0.1504268e-1),-0.3655620e-1),0.23498619),1.25331414),sqrt(x))
    
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

    return fast_fdividef(fadd_rn(fadd_rn(fmul_rn(peso1,mu1),fmul_rn(peso2,mu2)),fmul_rn(peso3,mu3)),fadd_rn(fadd_rn(peso1,peso2),peso3))


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

    return fast_fdividef(fadd_rn(fadd_rn(fmul_rn(peso1,sigma1_square),fmul_rn(peso2,sigma2_square)),fmul_rn(peso3,sigma3_square)),fadd_rn(fadd_rn(peso1,peso2),peso3))

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
        Refined grid starts from the data grid (N X J) adding K new point for PSF convolution. Look at
        data.py for more information about how the convolution is done.
    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    c_theta = cos(theta)
    s_theta = sin(theta)
    c_incl = cos(incl)
    if (i < x.shape[0]): 
        xtr = fsub_rn(x[i],x0) 
        ytr = fsub_rn(y[i],y0)                              
        xr = fadd_rn(fmul_rn(xtr,c_theta),fmul_rn(ytr,s_theta)) 
        yr = fsub_rn(fmul_rn(ytr,c_theta),fmul_rn(xtr,s_theta))
        yd = fast_fdividef(yr,c_incl)

        r_proj[i] = radius(xtr,ytr)    
        r_true[i] = radius(xr,yd)
        phi[i]    = atan2(yd,xr) 



@cuda.jit('void(float32,float32[::1],float32[::1],float32,int16)',
          fastmath=True,
          #max_registers=10,
          debug = False)
def Xfunction(Rb,r_proj,X,central_integration_radius,central_index):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute X(s) function useful for Hernquist/Jaffe surface density and 
    dispersion (see Hernquist 1990). 
                                             
        Parameters:
                Rb(float32)              : bulge scale radius
                r_proj(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)

        Returns:
                X(1D arr float32)   : X(s) function 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r_proj.shape[0]):
        if i != central_index:
            s = r_proj[i]
            s = fast_fdividef(s,Rb)
        else:
            s = central_integration_radius/Rb
            
        one_minus_s2 = fsub_rn(1.0,fmul_rn(s,s))
        if s < 1.0:
            tmp = fast_fdividef(log(fast_fdividef(fadd_rn(1.0,sqrt(one_minus_s2)),s)),sqrt(one_minus_s2))
            X[i] = tmp 
        elif s ==  1.0:
            X[i] = 1.0
        else:
            tmp = fast_fdividef(acos(frcp_rn(s)),sqrt(-one_minus_s2))
            X[i] = tmp
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
            rho_tmp = fast_fdividef(fmul_rn(fast_fdividef(Md,( fmul_rn(2.0,fmul_rn(M_PI,fmul_rn(Rd,Rd))))),exp(-r_tmp/Rd)),cos(incl))
            all_rhoD[i] = rho_tmp
    else:
        pass



@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1],float32,int16)',
          fastmath=True,
          #max_registers=10,
          debug = False)
def herquinst_rho(M,Rb,r,rhoB,Xs,central_integration_radius,central_index):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Hernquist projected surface density.  
                                             
        Parameters:
                M(float32)          : bulge mass
                Rb(float32)         : bulge scale radius
                r(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                Xs(1D arr float32)  : output of Xfunction 
        Returns:
                rhoB(1D arr float32)   : Hernquist surface density 

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r.shape[0]):            
        Rb2 = fmul_rn(Rb,Rb) 
        if i != central_index:
            s = r[i]
            s = fast_fdividef(s,Rb)
            if s >= 0.7 and s <=1.3:
                s_minus_one = fsub_rn(s,1.0)
                square_s_minus_one = fmul_rn(s_minus_one,s_minus_one)
                cube_s_minus_one = fmul_rn(square_s_minus_one,s_minus_one)
                fourth_s_minus_one = fmul_rn(square_s_minus_one,square_s_minus_one)
                fifth_s_minus_one = fmul_rn(cube_s_minus_one,square_s_minus_one)
                rho_tmp = fmul_rn(fast_fdividef(M,fmul_rn(fmul_rn(2.0,M_PI),Rb2)),(fadd_rn(fadd_rn(fsub_rn(fast_fdividef(4.0,15.0),fast_fdividef(fmul_rn(16.0,s_minus_one),35.0)),fsub_rn(fast_fdividef(fmul_rn(8.0,square_s_minus_one),15.0),fast_fdividef(fmul_rn(368.0,cube_s_minus_one),693.0))),fsub_rn(fast_fdividef(fmul_rn(1468.0,fourth_s_minus_one),3003.0),fast_fdividef(fmul_rn(928.0,fifth_s_minus_one),2145.0)))))
            else:
                X = Xs[i]
                s2 = fmul_rn(s,s)
                one_minus_s2 = fsub_rn(1.0,s2)
                one_minus_s2_2 = fmul_rn(one_minus_s2,one_minus_s2)
                rho_tmp =  fmul_rn(fast_fdividef(M,fmul_rn(2.0,fmul_rn(M_PI,fmul_rn(Rb2,one_minus_s2_2)))),fsub_rn(fmul_rn(fadd_rn(2.0,s2),X),3.0))
        else:
            sz = central_integration_radius/Rb
            rho_tmp = M / (M_PI*Rb2) /(1.0-sz*sz) * (Xs[i]-1.0)

        rhoB[i] = rho_tmp
    else:
        pass


@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1],float32,int16)',
          fastmath=True,
          #max_registers=10,
          debug = False)
def herquinst_sigma(M,Rb,r,sigmaB2,Xs,central_integration_radius,central_index):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Hernquist projected velocity dispersion.  
                                             
        Parameters:
                M(float32)          : bulge mass
                Rb(float32)         : bulge scale radius
                r(1D arr float32)   : projected radius (see r_proj in coordinate_transformation)
                Xs(1D arr float32)  : output of Xfunction 
        Returns:
                sigmaB2(1D arr float32)   : Hernquist projected velocity dispersion squared 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x


    if (i < r.shape[0]) :
        Rb2 = fmul_rn(Rb,Rb) 
        M_PI2 = fmul_rn(M_PI,M_PI)
        M_PI3 = fmul_rn(M_PI2,M_PI)
        G = 4.299e-6
        if i != central_index:            
            s = r[i]
            s = fast_fdividef(s,Rb)
            X = Xs[i]

            if s >= 0.7 and s <=1.3:
                s_minus_one = s-1.0
                square_s_minus_one = s_minus_one*s_minus_one
                sigma_tmp = (G*M/Rb) * ( (332.0-105.0*M_PI)/28.0 + (18784.0-5985.0*M_PI)*s_minus_one/588.0 + (707800.0-225225.0*M_PI)*square_s_minus_one/22638.0 )

            else:

                s2 = s*s
                one_minus_s2 = (1.0 - s2)
                one_minus_s2_2 = one_minus_s2*one_minus_s2
                if s >= 12.5:
                    s3 = s*s2
                    A =  fast_fdividef(fmul_rn(G,fmul_rn(M,8.0)),fmul_rn(15.0,fmul_rn(s,fmul_rn(M_PI,Rb))))
                    B =  fast_fdividef(fsub_rn(fast_fdividef(8.0,M_PI),fast_fdividef(fmul_rn(75.0,M_PI),64.0)),s)
                    C =  fast_fdividef(fsub_rn(fast_fdividef(64.0,M_PI2),fast_fdividef(297.0,56.0)),s2)
                    D =  fast_fdividef(fsub_rn(fsub_rn(fast_fdividef(512.0,M_PI3),fast_fdividef(1199.0,fmul_rn(21.0,M_PI))),fast_fdividef(fmul_rn(75.0,M_PI),512.0)),s3)
                    sigma_tmp = fmul_rn(A,fadd_rn(1.0,fadd_rn(B,fadd_rn(C,D))))   
                else:
                    s4 = s2*s2 
                    s6 = s4*s2
                    rhoB_tmp=   fast_fdividef(fsub_rn(fmul_rn(fadd_rn(2.0,s2),X),3.0),fmul_rn(fmul_rn(2.0,M_PI),fmul_rn(Rb2,one_minus_s2_2)))
                    A = fast_fdividef(fmul_rn(G,M),fmul_rn(fmul_rn(12.0,fmul_rn(M_PI,Rb2)),fmul_rn(Rb,rhoB_tmp)))
                    B = (1.0 /  (2.0*one_minus_s2_2*one_minus_s2))
                    C = (-3.0 * s2) * X * (8.0*s6 - 28.0*s4 + 35.0*s2 - 20.0) - 24.0*s6 + 68.0*s4 - 65.0*s2 + 6.0
                    D = 6.0*M_PI*s

                    sigma_tmp =  A*(B*C-D)
        else:
            X = Xs[i]
            s = central_integration_radius/Rb
            s2 = s*s
            s3 = s2*s 
            s4 = s2*s2 
            s5 = s3*s2
            s6 = s3*s3
            s_sqr_minus_one = 1.-s2
            A = 1.0/2./s_sqr_minus_one
            B = 3.-4.*M_PI*s-14.*s2+8.*M_PI*s3+8.*s4-4.*M_PI*s5+X*(15.*s2-20.*s4+8.*s6)
            sigma_tmp = G*M/(6.*Rb) * A*B / (X-1.0)
            sigmaB2[i] = sigma_tmp
    else:
        pass


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
                Xs(1D arr float32)  : output of Xfunction 
        Returns:
                rhoB(1D arr float32)   : Jaffe surface density 

    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r.shape[0]): 
        s = r[i]
        s = fast_fdividef(s,Rb)
        if 0.99<s<1.01:
            rho_tmp = fsub_rn(0.25,fast_fdividef(2.0,fmul_rn(3.0,M_PI)))
        else:
            if s > 300:
                s2 = fmul_rn(s,s)
                s3 = fmul_rn(s2,s)
                s4 = fmul_rn(s2,s2)
                rho_tmp = fsub_rn(fast_fdividef(1.0,fmul_rn(s3,8.0)),fast_fdividef(2.0,fmul_rn(fmul_rn(3,M_PI),s4))) 
            else:
                s2 = fmul_rn(s,s)
                X = Xs[i]
                rho_tmp = fadd_rn(fast_fdividef(1.0,fmul_rn(4.0,s)),fast_fdividef(fsub_rn(1.0,fmul_rn(fsub_rn(2.0,s2),X)),fmul_rn(2.0,fmul_rn(M_PI,fsub_rn(1.0,s2)))))
        
        rhoB[i] = fmul_rn(fast_fdividef(Mb,fmul_rn(Rb,Rb)),rho_tmp)


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
                Xs(1D arr float32)  : output of Xfunction 
        Returns:
                sigmaB2(1D arr float32)   : Jaffe projected velocity dispersion squared 

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x


    if (i < r.shape[0]) : 
        s = r[i]
        s = fast_fdividef(s,Rb)
        G = 4.299e-6

        if  0.99<s<1.01:
            rho_tmp = fsub_rn(0.25,fast_fdividef(2.0,fmul_rn(3.0,M_PI)))
            sigma_tmp = (G*Mb/(4.0*M_PI*Rb*rho_tmp))*(-0.15372602*s+0.20832732)
            sigma_tmp = fmul_rn(fast_fdividef(fmul_rn(G,Mb),fmul_rn(4.0,fmul_rn(M_PI,fmul_rn(Rb,rho_tmp)))),fadd_rn(fmul_rn(-0.15372602,s),0.20832732))
        else:

            if s >=4.0:
                M_PI2 = fmul_rn(M_PI,M_PI)
                s2 = fmul_rn(s,s)
                s3 = fmul_rn(s2,s)
                sigma_tmp = fmul_rn(fast_fdividef(fmul_rn(G,Mb),fmul_rn(fmul_rn(8.0,Rb),M_PI)),fsub_rn(fadd_rn(fast_fdividef(64.0,fmul_rn(15.0,s)),fast_fdividef(fsub_rn(fast_fdividef(1024.0,fmul_rn(45.0,M_PI)),fmul_rn(3.0,M_PI)),s2)),fmul_rn(128.0,fast_fdividef(fadd_rn(-896.0,fmul_rn(81.0,M_PI2)),fmul_rn(fmul_rn(945.0,M_PI2),s3)))))
            else:
                s2 = fmul_rn(s,s)
                s3 = fmul_rn(s2,s)
                s4 = fmul_rn(s3,s)
                X = Xs[i]

                rho_tmp = fadd_rn(fast_fdividef(1.0,fmul_rn(4.0,s)),fast_fdividef(fsub_rn(1.0,fmul_rn(fsub_rn(2.0,s2),X)),fmul_rn(2.0,fmul_rn(M_PI,fsub_rn(1.0,s2)))))
                A = fast_fdividef(fsub_rn(fadd_rn(fsub_rn(fsub_rn(fadd_rn(fast_fdividef(M_PI,fmul_rn(2.0,s)),11.0),fmul_rn(6.5,fmul_rn(M_PI,s))),fmul_rn(12.0,s2)),fmul_rn(6.0,fmul_rn(M_PI,s3))),fmul_rn(X,fadd_rn(fsub_rn(6.0,fmul_rn(19.0,s2)),fmul_rn(12.0,s4)))),fsub_rn(1.0,s2))
                B = fast_fdividef(fmul_rn(G,Mb),fmul_rn(fmul_rn(fmul_rn(4.0,M_PI),Rb),rho_tmp))
            
                sigma_tmp = fmul_rn(A,B)
        
        sigmaB2[i] = sigma_tmp


@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1],float32[::1],\
                float32[::1],float32[::1],float32[::1],float32[::1],\
                float32,int16,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def dehnen(Mb,Rb,gamma,r,s_grid,gamma_grid,rho_grid,sigma_grid,rhoB,sigmaB2,
           central_integration_radius,central_index,central_rho,central_sigma):
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Dehnen (see Dehnen 1993 or Tremaine 1993) projected surface density and velocity isotropic
    dispersion.  
                                             
        Parameters:
                Mb(float32)                         : bulge mass
                Rb(float32)                         : bulge scale radius
                gamma(float32)                      : inner slope of the density profile 
                r(1D arr float32)                   : projected radius (see r_proj in coordinate_transformation)
                s_grid(1D arr float32)              : R/Rb of the grid used for interpolation (see dehnen/s_grid.npy) 
                gamma_grid(1D arr float32)          : gamma of the grid used for interpolation (see dehnen/gamma_grid.npy) 
                rho_grid(1D arr float32)            : brightness of the grid used for interpolation (see dehnen/dehnen_brightness.npy) 
                sigma_grid(1D arr float32)          : velocity dispersion of the grid used for interpolation (see dehnen/dehnen_sigma2.npy) 
                central_integration_radius(float32) : radius of the central pixel for computing the average of brightness and dispersion
                central_index(int16)                : integer of the central index
                central_rho(1D arr float32)         : cumulative brightness of the grid used for interpolation (see dehnen/dehnen_luminosity.npy) 
                central_sigma(1D arr float32)       : cumulative dispersion of the grid used for interpolation (see dehnen/dehnen_avg_sigma2.npy)
        Returns:
                rhoB(1D arr float32)      : Dehnen projected surface density 
                sigmaB2(1D arr float32)   : Dehnen projected velocity dispersion 

    Notes:
        The brightness and sigma are computed by interpolation. 
        Since the central pixel can cause problems we approximate the brightness and the dispersion in the central point
        as the average on the spatial extension of the pixel (i.e. the surface brightness is the luminosity/area of the pixel).

    Improvements:
        Knowing the spatial extension of the grid in (s;gamma) we can compute directly their value reducing the number of memory accesses.
        The ID of the central pixel is computed using np.argmin on the projected radius but in principles it can be computed "a priori".
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if (i < r.shape[0]):
        G = 4.299e-6
        logs_min=-3
        gamma_min=0.
        dlogs=0.06060606060606055
        dgamma=0.17142857142857143
        N=15   
        m = int((gamma-gamma_min)/dgamma)  
        y1,y2 = gamma_grid[m],gamma_grid[m+1]     
        if i != central_index:
            s = r[i]
            s = fast_fdividef(s,Rb)
            Rb2 = Rb*Rb
            if s>1e3:
                s2 = s*s
                s3 = s2*s
                s4 = s2*s2
                s5 = s3*s2
                rhoB_tmp = Mb/Rb2*(3-gamma)/M_PI* ( M_PI/8./s3 + (gamma-4.)/3./s4 + \
                                                    3.*M_PI*(gamma-5.)*(gamma-4.)/(32.*s5) )
                sigma_tmp = G*Mb/Rb * ( 0.16877/s + (0.14410*(gamma-4.)-0.125*(7.-2.*gamma))/s2 +\
                                       (4.-gamma)/s3*(-0.12232*(gamma-4.)+0.02782*(3.*gamma-14.)) ) 
            else:
                l = int((log10(s)-logs_min)/dlogs)
                
                k,k1 = l*N+m,(l+1)*N+m
                x1,x2, = s_grid[l],s_grid[l+1]

                rhoB_tmp   = 10**my_bilinear(x1,x2,y1,y2,\
                                       rho_grid[k],rho_grid[k+1],rho_grid[k1],rho_grid[k1+1],s,gamma)
                sigmaB_tmp = 10**my_bilinear(x1,x2,y1,y2,\
                                       sigma_grid[k],sigma_grid[k+1],sigma_grid[k1],sigma_grid[k1+1],s,gamma)

                # occhio che rhoB_tmp deve essere adimensionale quando lo uso per calcolare 
                # sigmaB_tmp
                sigmaB_tmp = G*Mb*(3.-gamma)/(1+s)**(gamma-2)/Rb*sigmaB_tmp/rhoB_tmp    
                rhoB_tmp   = Mb*(3.-gamma)/(1+s)**(3.5-gamma)/Rb2*rhoB_tmp    
        else:
            s = central_integration_radius/Rb
            if s > 1e3:
                rhoB_tmp = Mb*(1.-(3.-gamma)*( M_PI/(4.*s) + (gamma-4.)/(3.*s*s) +\
                                              M_PI*(gamma-4.)*(gamma-5.)/(32.*s*s*s) ))
                #sigmaB_tmp = G*Mb*Mb*(3.-gamma)/Rb*( 3./(10.*s*s) - (3.*M_PI-32.)*(2.*gamma-7.)/(96.*s*s*s) \
                #                                 + 27.*(28.-15.*gamma+2.*gamma*gamma)/(140.*s*s*s*s) )
                sigmaB_tmp = 0.
                rhoB_tmp   = rhoB_tmp/(M_PI*central_integration_radius*central_integration_radius)
            else:
                l = int((log10(s)-logs_min)/dlogs)

                k,k1 = l*N+m,(l+1)*N+m
                x1,x2 = s_grid[l],s_grid[l+1]

                rhoB_tmp   = Mb*(1.0-2.0*(3.0-gamma)*(1+s)**(gamma-2.5)*10**my_bilinear(x1,x2,y1,y2,\
                                       central_rho[k],central_rho[k+1],central_rho[k1],central_rho[k1+1],s,gamma))

                sigmaB_tmp = 10**my_bilinear(x1,x2,y1,y2,\
                                       central_sigma[k],central_sigma[k+1],central_sigma[k1],central_sigma[k1+1],s,gamma)
                

                sigmaB_tmp = G*Mb*Mb/Rb*(1.-4.*(1.+s)**(2.*gamma-4.5)*(3.-gamma)*(5.-2.*gamma)*sigmaB_tmp)/(6.*(5.-2.*gamma))
                sigmaB_tmp = sigmaB_tmp/rhoB_tmp
                
                rhoB_tmp   = rhoB_tmp/(M_PI*central_integration_radius*central_integration_radius)

        rhoB[i] = rhoB_tmp
        sigmaB2[i] = sigmaB_tmp
    else:
        pass


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
        r_tmp = fast_fdividef(r_ns,fmul_rn(2.0,Rd))
        if r_tmp == 0.:
            r_tmp = 1e-8
            r_ns = 1e-8
        if r_tmp <= 1000.:
            i0e = my_i0e(r_tmp)
            i1e = my_i1e(r_tmp)
            v_c2 =  fmul_rn(fast_fdividef(fmul_rn(G,fmul_rn(Md,r_ns)),fmul_rn(Rd,Rd)),fmul_rn(r_tmp,fsub_rn(fmul_rn(i0e,my_k0e(r_tmp,i0e)),fmul_rn(i1e,my_k1e(r_tmp,i1e)))))
        else:
            r2 = fmul_rn(r_ns,r_ns) 
            r3 = fmul_rn(r2,r_ns)
            r5 = fmul_rn(r3,r2) 
            v_c2 =   fmul_rn(fast_fdividef(fmul_rn(G,fmul_rn(Md,r_ns)),fmul_rn(Rd,Rd)),fmul_rn(r_tmp,fadd_rn(fast_fdividef(1.0,fmul_rn(4.0,r3)),fast_fdividef(9.0,fmul_rn(32.0,r5)))))

        v_exp[i] = v_c2
    else:
        pass


@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def v_H(Mh,        # bulge/halo mass
        Rh,        # bulge/halo mass
        r,         # true radius
        v_H):      # bulge/halo velocity
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
        r_tmp_H = fast_fdividef(r_tmp,Rh)
        one_plus_r_H = fadd_rn(1.0,r_tmp_H)
        v_c2_H = fast_fdividef(fmul_rn(fmul_rn(G,Mh),r_tmp_H),fmul_rn(fmul_rn(Rh,one_plus_r_H),one_plus_r_H))
        v_H[i] = v_c2_H
    else:
        pass

@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def v_Halo(Mh,        # bulge/halo mass
        Rh,        # bulge/halo mass
        rho_crit,
        r,         # true radius
        v_H):      # bulge/halo velocity
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
        r_tmp_H = fast_fdividef(r_tmp,Rh)
        one_plus_r_H = fadd_rn(1.0,r_tmp_H)
        v_c2_H = fast_fdividef(fmul_rn(fmul_rn(G,Mh),r_tmp_H),fmul_rn(fmul_rn(Rh,one_plus_r_H),one_plus_r_H))
        v_H[i] = v_c2_H
    else:
        pass

@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def v_NFW(M_halo,        # bulge/halo mass
        c,        # bulge/halo mass
        rho_crit,
        r,         # true radius
        v_H):      # bulge/halo velocity
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute NFW circular velocity squared.
                                             
        Parameters:
                Mh(float32)         : halo mass
                c(float32)          : concentration r200/R_halo
                rho_crit(float32)   : 3*H(z)^2/(8piG) critical density at that redshift
                r(1D arr float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                v_H(1D arr float32)   : Halo(NFW) circular velocity squared. 
    
    Notes: 
        The mass of the NFW halo diverges so we assume the following:
        M_halo = M_200   and    M(r=r200) = 200*Mcrit = 200*(4/3pi*r200^3)rho_crit

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if i < r.shape[0] :
        G = 4.299e-6                          # kpc*km^2/M_sun/s^2
        r_tmp = r[i]
        a = ((3.*M_halo/(800.*M_PI*rho_crit))**(1./3.))/c
        r_tmp_H = fast_fdividef(r_tmp,a)
        one_plus_r_H = fadd_rn(1.0,r_tmp_H)
        v_c2_H = log(one_plus_r_H)-r_tmp_H/one_plus_r_H 
        v_H[i] = (v_c2_H*G*M_halo/r_tmp)/(log(1+c)-c/(1+c))
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

@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
          fastmath=True,
          #max_registers=10,
          debug = False)
def v_dehnen(Mb,        # bulge mass
             Rb,        # bulge radius
             gamma,
             r,         # true radius
             v_B):       # bulge velocity
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Denhen circular velocity squared.
                                             
        Parameters:
                Mb(float32)         : bulge mass
                Rb(float32)         : bulge scale radius
                gamma(float32)      : inner slope of the bulge profile
                r(1D arr float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                v_B(1D arr float32)   : Dehnen circular velocity squared. 

    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if i < r.shape[0] :
        G = 4.299e-6
        r_tmp = r[i]
        v_c2_B = G*Mb*r_tmp**(2.-gamma)/(r_tmp+Rb)**(3.-gamma)
        v_B[i] = v_c2_B
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
        v_tmp1 = fadd_rn(fadd_rn(fadd_rn(v_exp1[i],v_bulge[i]),v_halo[i]),v_exp2[i])
        v_tmp = fmul_rn(fmul_rn(sin(incl),cos(phi_tmp)),sqrt(v_tmp1))
        all_v[i]      = v_tmp 
        all_v2_abs[i] = v_tmp1
    else:
        pass
 

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#                         MODEL  B+D1+D2
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
                float32,float32,float32,float32,float32,\
                float32[::1],float32[::1],float32[::1],float32[::1],int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_rho_v(all_rhoB,         # (N,J,K) contains all bulge surface density
              all_rhoD1,        # (N,J,K) contains all disc surface density
              all_rhoD2,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              all_sigma2,
              kin_psf,              # (N,J,K) psf weights
              pho_psf,              # (N,J,K) psf weights
              L_to_M_b,
              L_to_M_d1,
              L_to_M_d2,
              k_D1,
              k_D2,
              rho_AVG,
              v_AVG,
              sigma_AVG,
              LM_AVG,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - total surface brightness
            - mean line of sight velocity
            - mean L/M ratio
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe/dehnen sigma)
                kin_psf(1D arr float32)     : psf weight for the kinematics (of lenght K)
                pho_psf(1D arr float32)     : psf weight for the photometry (of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                L_to_M_d2(float32)          : light to mass disc 2
                k_D1(float32)               : kinematic decomposition disc 1
                k_D2(float32)               : kinematic decomposition disc 2
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                rho_AVG(1D arr float32)     : total surface brightness
                v_AVG(1D arr float32)       : average line of sight velocity
                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion (NOT USED/COMPUTED IN THIS KERNEL)
                LM_AVG(1D arr float32)      : average light to mass ratio


    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]

            kinpsf_sum = 0.
            phopsf_sum = 0.
            Btot_sum_kin = 0.
            
            rhotot_sum = 0.
            Btot_sum = 0.

            v_sum = 0.
            #v_2_sum   = 0.

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                phopsf_tmp    = pho_psf[int(k-i*K)]
                ind         = indexing[k]
                rhoB_tmp    = all_rhoB[ind]
                rhoD1_tmp   = all_rhoD1[ind]
                rhoD2_tmp   = all_rhoD2[ind]
                B_B_tmp    = fmul_rn(L_to_M_b,rhoB_tmp)
                B_D1_tmp   = fmul_rn(L_to_M_d1,rhoD1_tmp)
                B_D2_tmp   = fmul_rn(L_to_M_d2,rhoD2_tmp)

                v_tmp      = all_v[ind]
                #v2_tmp     = all_v2[ind] 
                #sigma2_tmp = all_sigma2[ind] 


                phopsf_sum    = fadd_rn(phopsf_sum,phopsf_tmp)

                Btot_sum    = fadd_rn(Btot_sum,fmul_rn(fadd_rn(fadd_rn(B_B_tmp,B_D1_tmp),B_D2_tmp),phopsf_tmp)) 
                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(fadd_rn(B_B_tmp,B_D1_tmp),B_D2_tmp),kinpsf_tmp)) 
                #Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(fadd_rn(rhoB_tmp,rhoD1_tmp),rhoD2_tmp),kinpsf_tmp)) 
                rhotot_sum  = fadd_rn(rhotot_sum,fmul_rn(phopsf_tmp,fadd_rn(rhoD1_tmp,fadd_rn(rhoB_tmp,rhoD2_tmp))))

                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1+B_D2_tmp*v_tmp*k_D2)
                #v_sum     += kinpsf_tmp*(rhoD1_tmp*v_tmp*k_D1+rhoD2_tmp*v_tmp*k_D2)


            rho_AVG[i] = fast_fdividef(Btot_sum,phopsf_sum)

            v_AVG[i]     = fast_fdividef(v_sum,Btot_sum_kin)
            #sigma_AVG[i] = sqrt(v_2_sum/Btot_sum_kin)
            LM_AVG[i]    =  fast_fdividef(Btot_sum,rhotot_sum)
        
        else:
            pass

    else:
        pass




@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
                float32,float32,float32,float32,float32,\
                float32[::1],float32[::1],int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_sigma(all_rhoB,         # (N,J,K) contains all bulge surface density
              all_rhoD1,        # (N,J,K) contains all disc surface density
              all_rhoD2,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              all_sigma2,
              kin_psf,              # (N,J,K) psf weights
              L_to_M_b,
              L_to_M_d1,
              L_to_M_d2,
              k_D1,
              k_D2,
              v_AVG,
              sigma_AVG,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - mean line of sight velocity dispersion
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe sigma)
                kin_psf(1D arr float32)     : psf weight kinematics(of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                L_to_M_d2(float32)          : light to mass disc 2
                k_D1(float32)               : kinematic decomposition disc 1
                k_D2(float32)               : kinematic decomposition disc 2
                v_AGV(1D arr float32)       : average l.o.s velocity as computed in avg_rho_v
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion
    
    Notes:
        Dispersion accounts for beam smearing.
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]

            Btot_sum_kin = 0.

            v_2_sum   = 0.
            v_mean_tmp = v_AVG[i] 

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                ind         = indexing[k]
                B_B_tmp    = fmul_rn(L_to_M_b,all_rhoB[ind])
                B_D1_tmp   = fmul_rn(L_to_M_d1,all_rhoD1[ind])
                B_D2_tmp   = fmul_rn(L_to_M_d2,all_rhoD2[ind])
                #B_B_tmp    = all_rhoB[ind]
                #B_D1_tmp   = all_rhoD1[ind]
                #B_D2_tmp   = all_rhoD2[ind]

                v_tmp      = all_v[ind]
                v2_tmp     = all_v2[ind] 
                sigma2_tmp = all_sigma2[ind] 


                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(fadd_rn(B_B_tmp,B_D1_tmp),B_D2_tmp),kinpsf_tmp)) 

                v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0 +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp) + \
                                         B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp) + \
                                         sigma2_tmp*B_B_tmp)


            sigma_AVG[i] = sqrt(v_2_sum/Btot_sum_kin)
        
        else:
            pass

    else:
        pass


@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
                float32,float32,float32,float32,float32,\
                float32[::1],float32[::1],float32[::1],float32[::1],\
                float32[::1],float32[::1],float32[::1],float32[::1],\
                float32[::1],float32[::1],float32[::1],float32[::1],\
                int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_rho_v_all_output(all_rhoB,         # (N,J,K) contains all bulge surface density
              all_rhoD1,        # (N,J,K) contains all disc surface density
              all_rhoD2,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              all_sigma2,
              kin_psf,              # (N,J,K) psf weights
              pho_psf,              # (N,J,K) psf weights
              L_to_M_b,
              L_to_M_d1,
              L_to_M_d2,
              k_D1,
              k_D2,
              rho_AVG,
              rho_AVG_B,
              rho_AVG_D1,
              rho_AVG_D2,
              v_AVG,
              v_AVG_B,
              v_AVG_D1,
              v_AVG_D2,
              LM_AVG,
              LM_AVG_B,
              LM_AVG_D1,
              LM_AVG_D2,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total,B,D1,D2) 
            - mean line of sight velocity (total,B,D1,D2)
            - mean L/M ratio (total,B,D1,D2)
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe/dehnen sigma)
                kin_psf(1D arr float32)     : psf weight for the kinematics (of lenght K)
                pho_psf(1D arr float32)     : psf weight for the photometry (of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                L_to_M_d2(float32)          : light to mass disc 2
                k_D1(float32)               : kinematic decomposition disc 1
                k_D2(float32)               : kinematic decomposition disc 2
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                rho_AVG(1D arr float32)     : total surface brightness
                rho_AVG_B(1D arr float32)   : bulge surface brightness
                rho_AVG_D1(1D arr float32)  : disc1 surface brightness
                rho_AVG_D2(1D arr float32)  : disc2 surface brightness
                v_AVG(1D arr float32)       : average line of sight velocity
                v_AVG_B(1D arr float32)     : bulge line of sight velocity
                v_AVG_D1(1D arr float32)    : disc1 line of sight velocity
                v_AVG_D2(1D arr float32)    : disc2 line of sight velocity
                LM_AVG(1D arr float32)      : average L/M ratio
                LM_AVG_B(1D arr float32)    : bulge L/M ratio
                LM_AVG_D1(1D arr float32)   : disc1 L/M ratio
                LM_AVG_D2(1D arr float32)   : disc2 L/M ratio


    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]

            kinpsf_sum = 0.
            phopsf_sum = 0.
            Btot_sum_kin = 0.
            
            rhotot_sum = 0.
            rho_B_sum = 0.
            rho_D1_sum = 0.
            rho_D2_sum = 0.

            Btot_sum = 0.
            B_B_sum = 0.
            B_D1_sum = 0.
            B_D2_sum = 0.

            B_d1_sum_kin = 0.
            B_d2_sum_kin = 0.

            v_sum = 0.
            v_sum_D1 = 0.
            v_sum_D2 = 0.
            #v_2_sum   = 0.

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                phopsf_tmp    = pho_psf[int(k-i*K)]
                ind         = indexing[k]
                rhoB_tmp    = all_rhoB[ind]
                rhoD1_tmp   = all_rhoD1[ind]
                rhoD2_tmp   = all_rhoD2[ind]
                B_B_tmp    = fmul_rn(L_to_M_b,rhoB_tmp)
                B_D1_tmp   = fmul_rn(L_to_M_d1,rhoD1_tmp)
                B_D2_tmp   = fmul_rn(L_to_M_d2,rhoD2_tmp)

                v_tmp      = all_v[ind]
                #v2_tmp     = all_v2[ind] 
                #sigma2_tmp = all_sigma2[ind] 


                phopsf_sum    = fadd_rn(phopsf_sum,phopsf_tmp)

                Btot_sum    = fadd_rn(Btot_sum,fmul_rn(fadd_rn(fadd_rn(B_B_tmp,B_D1_tmp),B_D2_tmp),phopsf_tmp)) 
                B_B_sum     = fadd_rn(B_B_sum,fmul_rn(B_B_tmp,phopsf_tmp)) 
                B_D1_sum    = fadd_rn(B_D1_sum,fmul_rn(B_D1_tmp,phopsf_tmp)) 
                B_D2_sum    = fadd_rn(B_D2_sum,fmul_rn(B_D2_tmp,phopsf_tmp)) 

                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(fadd_rn(B_B_tmp,B_D1_tmp),B_D2_tmp),kinpsf_tmp)) 
                #B_d1_sum_kin = fadd_rn(B_d1_sum_kin,fmul_rn(B_D1_tmp,kinpsf_tmp)) 
                #B_d2_sum_kin = fadd_rn(B_d2_sum_kin,fmul_rn(B_D2_tmp,kinpsf_tmp)) 

                rhotot_sum  = fadd_rn(rhotot_sum,fmul_rn(phopsf_tmp,fadd_rn(rhoD1_tmp,fadd_rn(rhoB_tmp,rhoD2_tmp))))
                #rho_B_sum   = fadd_rn(rho_B_sum,fmul_rn(phopsf_tmp,rhoB_tmp))
                #rho_D1_sum  = fadd_rn(rho_D1_sum,fmul_rn(phopsf_tmp,rhoD1_tmp))
                #rho_D2_sum  = fadd_rn(rho_D2_sum,fmul_rn(phopsf_tmp,rhoD2_tmp))

                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1+B_D2_tmp*v_tmp*k_D2)
                v_sum_D1     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1)
                v_sum_D2     += kinpsf_tmp*(B_D2_tmp*v_tmp*k_D2)
            #    v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0 + sigma2_tmp*B_B_tmp)


            rho_AVG[i] = fast_fdividef(Btot_sum,phopsf_sum)
            rho_AVG_B[i] = fast_fdividef(B_B_sum,phopsf_sum)
            rho_AVG_D1[i] = fast_fdividef(B_D1_sum,phopsf_sum)
            rho_AVG_D2[i] = fast_fdividef(B_D2_sum,phopsf_sum)

            v_AVG[i]     = fast_fdividef(v_sum,Btot_sum_kin)
            v_AVG_B[i]     = 0.
            v_AVG_D1[i]     = fast_fdividef(v_sum_D1,Btot_sum_kin)
            v_AVG_D2[i]     = fast_fdividef(v_sum_D2,Btot_sum_kin)
            #sigma_AVG[i] = sqrt(v_2_sum/Btot_sum_kin)

            LM_AVG[i]       =  fast_fdividef(Btot_sum,rhotot_sum)
            LM_AVG_B[i]     =  fast_fdividef(Btot_sum,B_B_sum/L_to_M_b)
            LM_AVG_D1[i]    =  fast_fdividef(Btot_sum,B_D1_sum/L_to_M_d1)
            LM_AVG_D2[i]    =  fast_fdividef(Btot_sum,B_D2_sum/L_to_M_d2)
        
        else:
            pass

    else:
        pass




@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
                float32,float32,float32,float32,float32,\
                float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
                int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_sigma_all_output(all_rhoB,         # (N,J,K) contains all bulge surface density
              all_rhoD1,        # (N,J,K) contains all disc surface density
              all_rhoD2,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              all_sigma2,
              kin_psf,              # (N,J,K) psf weights
              L_to_M_b,
              L_to_M_d1,
              L_to_M_d2,
              k_D1,
              k_D2,
              v_AVG,
              sigma_AVG,
              sigma_AVG_B,
              sigma_AVG_D1,
              sigma_AVG_D2,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - mean line of sight velocity dispersion
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe sigma)
                kin_psf(1D arr float32)     : psf weight kinematics(of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                L_to_M_d2(float32)          : light to mass disc 2
                k_D1(float32)               : kinematic decomposition disc 1
                k_D2(float32)               : kinematic decomposition disc 2
                v_AGV(1D arr float32)       : average l.o.s velocity as computed in avg_rho_v
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                sigma_AVG(1D arr float32)    : average line of sight velocity dispersion
                sigma_AVG_B(1D arr float32)  : bulge line of sight velocity dispersion
                sigma_AVG_D1(1D arr float32) : disc1 line of sight velocity dispersion
                sigma_AVG_D2(1D arr float32) : disc2 line of sight velocity dispersion
    
    Notes:
        Dispersion accounts for beam smearing.
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]

            Btot_sum_kin = 0.

            v_2_sum   = 0.
            v_2_sum_B   = 0.
            v_2_sum_D1   = 0.
            v_2_sum_D2   = 0.
            v_mean_tmp = v_AVG[i] 

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                ind         = indexing[k]
                B_B_tmp    = fmul_rn(L_to_M_b,all_rhoB[ind])
                B_D1_tmp   = fmul_rn(L_to_M_d1,all_rhoD1[ind])
                B_D2_tmp   = fmul_rn(L_to_M_d2,all_rhoD2[ind])

                v_tmp      = all_v[ind]
                v2_tmp     = all_v2[ind] 
                sigma2_tmp = all_sigma2[ind] 


                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(fadd_rn(B_B_tmp,B_D1_tmp),B_D2_tmp),kinpsf_tmp)) 

                v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0 +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp) + \
                                         B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp) + \
                                         sigma2_tmp*B_B_tmp)
                v_2_sum_B += kinpsf_tmp*(sigma2_tmp*B_B_tmp)           
                v_2_sum_D1 += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0+B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp))        
                v_2_sum_D2 += kinpsf_tmp*(B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0+B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp))          

            sigma_AVG[i] = sqrt(v_2_sum/Btot_sum_kin)
            sigma_AVG_B[i] = sqrt(v_2_sum_B/Btot_sum_kin)
            sigma_AVG_D1[i] = sqrt(v_2_sum_D1/Btot_sum_kin)
            sigma_AVG_D2[i] = sqrt(v_2_sum_D2/Btot_sum_kin)
        
        else:
            pass

    else:
        pass


#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
#                float32,float32,float32,float32,float32,\
#                float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
#                int16[::1],float32[::1])',
#          fastmath=True,
#          debug = False)
#def avg_rho_v(all_rhoB,         # (N,J,K) contains all bulge surface density
#              all_rhoD1,        # (N,J,K) contains all disc surface density
#              all_rhoD2,        # (N,J,K) contains all disc surface density
#              all_v,
#              kin_psf,              # (N,J,K) psf weights
#              pho_psf,              # (N,J,K) psf weights
#              L_to_M_b,
#              L_to_M_d1,
#              L_to_M_d2,
#              k_D1,
#              k_D2,
#              rho_AVG,
#              v_AVG,
#              vD1_mean,
#              vD2_mean,
#              LM_AVG,
#              indexing,
#              goodpix):
#
#    ''' 
#    KERNEL
#    ----------------------------------------------------------------------
#    ----------------------------------------------------------------------
#    Compute:
#            - total surface brightness
#            - mean line of sight velocity
#            - mean line of sight velocity dispersion
#                                             
#        Parameters:
#                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
#                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
#                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
#                all_v(1D arr float32)       : line of sight velocity (see v_tot)
#                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
#                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe sigma)
#                psf(1D arr float32)         : psf weight (of lenght K)
#                L_to_M_b(float32)           : light to mass bulge
#                L_to_M_d1(float32)          : light to mass disc 1
#                L_to_M_d2(float32)          : light to mass disc 2
#                k_D1(float32)               : kinematic decomposition disc 1
#                k_D2(float32)               : kinematic decomposition disc 2
#        Returns:
#                rho_AVG(1D arr float32)     : total surface brightness
#                v_AVG(1D arr float32)       : average line of sight velocity
#                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion
#
#    '''
#
#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#    if (i < v_AVG.shape[0]):
#        if goodpix[i] == 1.:
#            K = kin_psf.shape[0]
#
#            kinpsf_sum = 0.
#            phopsf_sum = 0.
#
#            rhoB_sum = 0.
#            rhoD1_sum = 0.
#            rhoD2_sum = 0.
#            rhotot_sum = 0.
#            Btot_sum = 0.
#
#            vD1_sum = 0.
#            vD2_sum = 0.
#
#            for k in range(i*K,(i+1)*K):
#                kinpsf_tmp    = kin_psf[int(k-i*K)]
#                phopsf_tmp    = pho_psf[int(k-i*K)]
#                ind         = indexing[k]
#                rhoB_tmp    = all_rhoB[ind]
#                rhoD1_tmp   = all_rhoD1[ind]
#                rhoD2_tmp   = all_rhoD2[ind]
#                B_B_tmp    = fmul_rn(L_to_M_b,rhoB_tmp)
#                B_D1_tmp   = fmul_rn(L_to_M_d1,rhoD1_tmp)
#                B_D2_tmp   = fmul_rn(L_to_M_d2,rhoD2_tmp)
#
#                v_tmp      = all_v[ind]
#
#                kinpsf_sum    = fadd_rn(kinpsf_sum,kinpsf_tmp)
#                phopsf_sum    = fadd_rn(phopsf_sum,phopsf_tmp)
#
#                rhoB_sum    = fadd_rn(rhoB_sum,fmul_rn(B_B_tmp,kinpsf_tmp))
#                rhoD1_sum   = fadd_rn(rhoD1_sum,fmul_rn(B_D1_tmp,kinpsf_tmp)) 
#                rhoD2_sum   = fadd_rn(rhoD2_sum,fmul_rn(B_D2_tmp,kinpsf_tmp)) 
#                Btot_sum    = fadd_rn(Btot_sum,fmul_rn(fadd_rn(fadd_rn(B_B_tmp,B_D1_tmp),B_D2_tmp),phopsf_tmp)) 
#                rhotot_sum  = fadd_rn(rhotot_sum,fmul_rn(phopsf_tmp,fadd_rn(rhoD1_tmp,fadd_rn(rhoB_tmp,rhoD2_tmp))))
#
#                vD1_sum   = fadd_rn(vD1_sum,fmul_rn(kinpsf_tmp,fmul_rn(B_D1_tmp,v_tmp)))
#                vD2_sum   = fadd_rn(vD2_sum,fmul_rn(kinpsf_tmp,fmul_rn(B_D2_tmp,v_tmp)))
#
#
#
#            rhoB_mean  = fast_fdividef(rhoB_sum,kinpsf_sum)
#            rhoD1_mean = fast_fdividef(rhoD1_sum,kinpsf_sum)
#            rhoD2_mean = fast_fdividef(rhoD2_sum,kinpsf_sum)
#            rho_AVG[i] = fast_fdividef(Btot_sum,phopsf_sum)
#
#            vD1_mean_tmp        = fmul_rn(k_D1,fast_fdividef(vD1_sum,rhoD1_sum))
#            vD2_mean_tmp        = fmul_rn(k_D2,fast_fdividef(vD2_sum,rhoD2_sum))
#
#            vD1_mean[i] = vD1_mean_tmp            
#            vD2_mean[i] = vD2_mean_tmp            
#
#
#            v_AVG[i]     = v_avg(rhoD1_mean,rhoD2_mean,rhoB_mean,vD1_mean_tmp,vD2_mean_tmp,0)
#            LM_AVG[i]    =  fast_fdividef(Btot_sum,rhotot_sum)
#        
#        else:
#            pass
#
#    else:
#        pass
#
#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
#                float32,float32,float32,float32,float32,\
#                float32[::1],float32[::1],float32[::1],float32[::1],int16[::1],float32[::1])',
#          fastmath=True,
#          debug = False)
#def avg_sigma(all_rhoB,         # (N,J,K) contains all bulge surface density
#              all_rhoD1,        # (N,J,K) contains all disc surface density
#              all_rhoD2,        # (N,J,K) contains all disc surface density
#              all_v,
#              all_v2,
#              all_sigma2,
#              kin_psf,              # (N,J,K) psf weights
#              L_to_M_b,
#              L_to_M_d1,
#              L_to_M_d2,
#              k_D1,
#              k_D2,
#              v_AVG,
#              vD1_mean,
#              vD2_mean,
#              sigma_AVG,
#              indexing,
#              goodpix):
#
#    ''' 
#    KERNEL
#    ----------------------------------------------------------------------
#    ----------------------------------------------------------------------
#    Compute:
#            - total surface brightness
#            - mean line of sight velocity
#            - mean line of sight velocity dispersion
#                                             
#        Parameters:
#                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
#                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
#                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
#                all_v(1D arr float32)       : line of sight velocity (see v_tot)
#                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
#                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe sigma)
#                psf(1D arr float32)         : psf weight (of lenght K)
#                L_to_M_b(float32)           : light to mass bulge
#                L_to_M_d1(float32)          : light to mass disc 1
#                L_to_M_d2(float32)          : light to mass disc 2
#                k_D1(float32)               : kinematic decomposition disc 1
#                k_D2(float32)               : kinematic decomposition disc 2
#        Returns:
#                rho_AVG(1D arr float32)     : total surface brightness
#                v_AVG(1D arr float32)       : average line of sight velocity
#                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion
#
#    '''
#
#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#    if (i < v_AVG.shape[0]):
#        if goodpix[i] == 1.:
#            K = kin_psf.shape[0]
#
#
#            rhoB_sum = 0.
#            rhoD1_sum = 0.
#            rhoD2_sum = 0.
#            rhotot_sum = 0.
#            Btot_sum = 0.
#
#            vD1_sum = 0.
#            vD2_sum = 0.
#            vD1_2_sum   = 0.
#            vD2_2_sum   = 0.
#            vB_2_sum    = 0.
#            kinpsf_sum = 0.
#            vD1_mean_tmp = vD1_mean[i]
#            vD2_mean_tmp = vD2_mean[i]
#
#            for k in range(i*K,(i+1)*K):
#                kinpsf_tmp    = kin_psf[int(k-i*K)]
#                ind         = indexing[k]
#                B_B_tmp    = fmul_rn(L_to_M_b,all_rhoB[ind])
#                B_D1_tmp   = fmul_rn(L_to_M_d1,all_rhoD1[ind])
#                B_D2_tmp   = fmul_rn(L_to_M_d2,all_rhoD2[ind])
#
#                v_tmp      = all_v[ind]
#                v2_tmp     = all_v2[ind] 
#                sigma2_tmp = all_sigma2[ind] 
#
#                kinpsf_sum    = fadd_rn(kinpsf_sum,kinpsf_tmp)
#
#                rhoB_sum    = fadd_rn(rhoB_sum,fmul_rn(B_B_tmp,kinpsf_tmp))
#                rhoD1_sum   = fadd_rn(rhoD1_sum,fmul_rn(B_D1_tmp,kinpsf_tmp)) 
#                rhoD2_sum   = fadd_rn(rhoD2_sum,fmul_rn(B_D2_tmp,kinpsf_tmp)) 
#
#                #vD1_2_sum   = fadd_rn(vD1_2_sum,fmul_rn(kinpsf_tmp,fmul_rn(B_D1_tmp,v2_tmp)))
#                #vD2_2_sum   = fadd_rn(vD2_2_sum,fmul_rn(kinpsf_tmp,fmul_rn(B_D2_tmp,v2_tmp)))
#
#                vD1_2_sum   += kinpsf_tmp*B_D1_tmp*(v2_tmp*(1.-k_D1*k_D1)/3.+(v_tmp-vD1_mean_tmp)*(v_tmp-vD1_mean_tmp))
#                vD2_2_sum   += kinpsf_tmp*B_D2_tmp*(v2_tmp*(1.-k_D2*k_D2)/3.+(v_tmp-vD2_mean_tmp)*(v_tmp-vD2_mean_tmp))
#                vB_2_sum    = fadd_rn(vB_2_sum,fmul_rn(kinpsf_tmp,fmul_rn(B_B_tmp,sigma2_tmp)))
#
#
#            rhoB_mean  = fast_fdividef(rhoB_sum,kinpsf_sum)
#            rhoD1_mean = fast_fdividef(rhoD1_sum,kinpsf_sum)
#            rhoD2_mean = fast_fdividef(rhoD2_sum,kinpsf_sum)
#
#
#            sigmaD1_2_mean = fast_fdividef(vD1_2_sum,rhoD1_sum)
#            sigmaD2_2_mean = fast_fdividef(vD2_2_sum,rhoD2_sum)
#            sigmaB2_mean   = fast_fdividef(vB_2_sum,rhoB_sum)
#
#
#            sigma_AVG[i] = sqrt(sigma_avg(rhoD1_mean,rhoD2_mean,rhoB_mean,sigmaD1_2_mean,sigmaD2_2_mean,sigmaB2_mean))
#        
#        else:
#            pass
#
#    else:
#        pass

#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1],float32[::1],int16[::1])',
#          fastmath=True,
#          debug = False)
#def avg_LM(all_rhoB,        # (N,J,K) contains all bulge surface density
#           all_rhoD1,        # (N,J,K) contains all disc surface density
#           all_rhoD2,        # (N,J,K) contains all disc surface density
#           psf,             # (N,J,K) psf weights
#           L_to_M_bulge,
#           L_to_M_disc1,
#           L_to_M_disc2,
#           rho_AVG,
#           LM_AVG,
#           indexing):
#
#    ''' 
#    KERNEL
#    ----------------------------------------------------------------------
#    ----------------------------------------------------------------------
#    Compute avg light to mass (DECIDE NOTATION)
#                                             
#        Parameters:
#                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
#                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
#                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
#                psf(1D arr float32)         : psf weight (of lenght K)
#                L_to_M_b(float32)           : light to mass bulge
#                L_to_M_d1(float32)          : light to mass disc 1
#                L_to_M_d2(float32)          : light to mass disc 2
#        Returns:
#                LM_AVG(1D arr float32)   : average light to mass ratio
#    '''
#
#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#    if (i < LM_AVG.shape[0]):
#        K = psf.shape[0]
#
#        rho_tot = 0.
#        LM_sum = 0.
#        sum_psf = 0.
#        for k in range(i*K,(i+1)*K):
#            ind = indexing[k]
#            rhoD1_tmp = all_rhoD1[ind]
#            rhoD2_tmp = all_rhoD2[ind]
#            rhoB_tmp  = all_rhoB[ind]
#            psf_tmp   = psf[int(k-i*K)]
#            sum_psf  = fadd_rn(sum_psf,psf_tmp)
#            rho_tot   = fadd_rn(rho_tot,fmul_rn(psf_tmp,fadd_rn(rhoD1_tmp,fadd_rn(rhoB_tmp,rhoD2_tmp))))
#            LM_sum    = fadd_rn(LM_sum,fmul_rn(psf_tmp,fadd_rn(fmul_rn(rhoD2_tmp,L_to_M_disc2),fadd_rn(fmul_rn(rhoB_tmp,L_to_M_bulge),fmul_rn(rhoD1_tmp,L_to_M_disc1)))))
#
#        LM_AVG[i] =  fast_fdividef(LM_sum,rho_tot) #fast_fdividef(rho_tot,LM_sum)
#        rho_AVG[i] = fast_fdividef(LM_sum,sum_psf)
#    else:
#        pass


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#                          MODEL B+D
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
           float32,float32,float32,\
           float32[::1],float32[::1],float32[::1],float32[::1],int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_rho_v_2_vis_components(all_rhoB,         # (N,J,K) contains all bulge surface density
              all_rhoD1,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              all_sigma2,
              kin_psf,              # (N,J,K) psf weights
              pho_psf,              # (N,J,K) psf weights
              L_to_M_b,
              L_to_M_d1,
              k_D1,
              rho_AVG,
              v_AVG,
              sigma_AVG,
              LM_AVG,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - total surface brightness
            - mean line of sight velocity
            - mean L/M ratio
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe/dehnen sigma)
                kin_psf(1D arr float32)     : psf weight for the kinematics (of lenght K)
                pho_psf(1D arr float32)     : psf weight for the photometry (of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                k_D1(float32)               : kinematic decomposition disc 1
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                rho_AVG(1D arr float32)     : total surface brightness
                v_AVG(1D arr float32)       : average line of sight velocity
                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion (NOT USED/COMPUTED IN THIS KERNEL)
                LM_AVG(1D arr float32)      : average light to mass ratio


    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]
        
            kinpsf_sum = 0.
            phopsf_sum = 0.

            Btot_sum_kin = 0.
            
            rhotot_sum = 0.
            Btot_sum = 0.

            v_sum = 0.

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                phopsf_tmp    = pho_psf[int(k-i*K)]
                ind         = indexing[k]
                rhoB_tmp    = all_rhoB[ind]
                rhoD1_tmp   = all_rhoD1[ind]
                B_B_tmp    = fmul_rn(L_to_M_b,rhoB_tmp)
                B_D1_tmp   = fmul_rn(L_to_M_d1,rhoD1_tmp)

                v_tmp      = all_v[ind]
                #v2_tmp     = all_v2[ind] 
                #sigma2_tmp = all_sigma2[ind] 


                phopsf_sum    = fadd_rn(phopsf_sum,phopsf_tmp)
               
                Btot_sum    = fadd_rn(Btot_sum,fmul_rn(fadd_rn(B_B_tmp,B_D1_tmp),phopsf_tmp)) 
                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(B_B_tmp,B_D1_tmp),kinpsf_tmp)) 
                rhotot_sum  = fadd_rn(rhotot_sum,fmul_rn(phopsf_tmp,fadd_rn(rhoD1_tmp,rhoB_tmp)))

                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1)
                #v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0 + sigma2_tmp*B_B_tmp)


            rho_AVG[i] = fast_fdividef(Btot_sum,phopsf_sum)

            v_AVG[i]     = fast_fdividef(v_sum,Btot_sum_kin)
            #sigma_AVG[i] = sqrt(v_2_sum/Btot_sum_kin)
            LM_AVG[i]    =  fast_fdividef(Btot_sum,rhotot_sum)
 
        else:
            pass

    else:
        pass


@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
           float32,float32,float32,\
           float32[::1],float32[::1],int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_sigma_2_vis_components(all_rhoB,         # (N,J,K) contains all bulge surface density
              all_rhoD1,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              all_sigma2,
              kin_psf,              # (N,J,K) psf weights
              L_to_M_b,
              L_to_M_d1,
              k_D1,
              v_AVG,
              sigma_AVG,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - mean line of sight velocity dispersion
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe sigma)
                kin_psf(1D arr float32)     : psf weight kinematics(of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                k_D1(float32)               : kinematic decomposition disc 1
                v_AGV(1D arr float32)       : average l.o.s velocity as computed in avg_rho_v
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion
    
    Notes:
        Dispersion accounts for beam smearing.
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]
        
            kinpsf_sum = 0.
            Btot_sum_kin = 0.
            
            v_2_sum   = 0.
            v_mean_tmp = v_AVG[i] 

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                ind         = indexing[k]
                B_B_tmp    = fmul_rn(L_to_M_b,all_rhoB[ind])
                B_D1_tmp   = fmul_rn(L_to_M_d1,all_rhoD1[ind])

                v_tmp      = all_v[ind]
                v2_tmp     = all_v2[ind] 
                sigma2_tmp = all_sigma2[ind] 


               
                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(B_B_tmp,B_D1_tmp),kinpsf_tmp)) 

                v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp) + \
                                         sigma2_tmp*B_B_tmp)


            sigma_AVG[i] = sqrt(v_2_sum/Btot_sum_kin)
        else:
            pass

    else:
        pass


@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
           float32,float32,float32,\
           float32[::1],float32[::1],float32[::1],float32[::1],\
           float32[::1],float32[::1],float32[::1],float32[::1],\
           float32[::1],int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_rho_v_2_vis_components_all_output(all_rhoB,         # (N,J,K) contains all bulge surface density
              all_rhoD1,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              all_sigma2,
              kin_psf,              # (N,J,K) psf weights
              pho_psf,              # (N,J,K) psf weights
              L_to_M_b,
              L_to_M_d1,
              k_D1,
              rho_AVG,
              rho_AVG_B,
              rho_AVG_D1,
              v_AVG,
              v_AVG_B,
              v_AVG_D1,
              LM_AVG,
              LM_AVG_B,
              LM_AVG_D1,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total,B,D1,D2) 
            - mean line of sight velocity (total,B,D1,D2)
            - mean L/M ratio (total,B,D1,D2)
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe/dehnen sigma)
                kin_psf(1D arr float32)     : psf weight for the kinematics (of lenght K)
                pho_psf(1D arr float32)     : psf weight for the photometry (of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                k_D1(float32)               : kinematic decomposition disc 1
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                rho_AVG(1D arr float32)     : total surface brightness
                rho_AVG_B(1D arr float32)   : bulge surface brightness
                rho_AVG_D1(1D arr float32)  : disc1 surface brightness
                v_AVG(1D arr float32)       : average line of sight velocity
                v_AVG_B(1D arr float32)     : bulge line of sight velocity
                v_AVG_D1(1D arr float32)    : disc1 line of sight velocity
                LM_AVG(1D arr float32)      : average L/M ratio
                LM_AVG_B(1D arr float32)    : bulge L/M ratio
                LM_AVG_D1(1D arr float32)   : disc1 L/M ratio


    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]
        
            kinpsf_sum = 0.
            phopsf_sum = 0.
            Btot_sum_kin = 0.
            
            rhotot_sum = 0.
            rho_B_sum = 0.
            rho_D1_sum = 0.

            Btot_sum = 0.
            B_B_sum = 0.
            B_D1_sum = 0.

            B_d1_sum_kin = 0.

            v_sum = 0.
            v_sum_D1 = 0.

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                phopsf_tmp    = pho_psf[int(k-i*K)]
                ind         = indexing[k]
                rhoB_tmp    = all_rhoB[ind]
                rhoD1_tmp   = all_rhoD1[ind]
                B_B_tmp    = fmul_rn(L_to_M_b,rhoB_tmp)
                B_D1_tmp   = fmul_rn(L_to_M_d1,rhoD1_tmp)

                v_tmp      = all_v[ind]
                #v2_tmp     = all_v2[ind] 
                #sigma2_tmp = all_sigma2[ind] 


                phopsf_sum    = fadd_rn(phopsf_sum,phopsf_tmp)
               
                Btot_sum    = fadd_rn(Btot_sum,fmul_rn(fadd_rn(B_B_tmp,B_D1_tmp),phopsf_tmp)) 
                B_B_sum     = fadd_rn(B_B_sum,fmul_rn(B_B_tmp,phopsf_tmp)) 
                B_D1_sum    = fadd_rn(B_D1_sum,fmul_rn(B_D1_tmp,phopsf_tmp))                
                
                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(B_B_tmp,B_D1_tmp),kinpsf_tmp)) 
                rhotot_sum  = fadd_rn(rhotot_sum,fmul_rn(phopsf_tmp,fadd_rn(rhoD1_tmp,rhoB_tmp)))

                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1)
                v_sum_D1     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1)
                #v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0 + sigma2_tmp*B_B_tmp)


            rho_AVG[i] = fast_fdividef(Btot_sum,phopsf_sum)
            rho_AVG_B[i] = fast_fdividef(B_B_sum,phopsf_sum)
            rho_AVG_D1[i] = fast_fdividef(B_D1_sum,phopsf_sum)

            v_AVG[i]     = fast_fdividef(v_sum,Btot_sum_kin)
            v_AVG_B[i]     = 0.
            v_AVG_D1[i]     = fast_fdividef(v_sum_D1,Btot_sum_kin)

            LM_AVG[i]    =  fast_fdividef(Btot_sum,rhotot_sum)
            LM_AVG_B[i]     =  fast_fdividef(Btot_sum,B_B_sum/L_to_M_b)
            LM_AVG_D1[i]    =  fast_fdividef(Btot_sum,B_D1_sum/L_to_M_d1)

        else:
            pass

    else:
        pass


@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
           float32,float32,float32,\
           float32[::1],float32[::1],float32[::1],float32[::1],\
           int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_sigma_2_vis_components_all_output(all_rhoB,         # (N,J,K) contains all bulge surface density
              all_rhoD1,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              all_sigma2,
              kin_psf,              # (N,J,K) psf weights
              L_to_M_b,
              L_to_M_d1,
              k_D1,
              v_AVG,
              sigma_AVG,
              sigma_AVG_B,
              sigma_AVG_D1,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - mean line of sight velocity dispersion
                                             
        Parameters:
                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe sigma)
                kin_psf(1D arr float32)     : psf weight kinematics(of lenght K)
                L_to_M_b(float32)           : light to mass bulge
                L_to_M_d1(float32)          : light to mass disc 1
                k_D1(float32)               : kinematic decomposition disc 1
                v_AGV(1D arr float32)       : average l.o.s velocity as computed in avg_rho_v
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                sigma_AVG(1D arr float32)    : average line of sight velocity dispersion
                sigma_AVG_B(1D arr float32)  : bulge line of sight velocity dispersion
                sigma_AVG_D1(1D arr float32) : disc1 line of sight velocity dispersion
    
    Notes:
        Dispersion accounts for beam smearing.
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]
        
            kinpsf_sum = 0.
            Btot_sum_kin = 0.
            
            v_2_sum   = 0.
            v_2_sum_B   = 0.
            v_2_sum_D1   = 0.
            v_mean_tmp = v_AVG[i] 

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                ind         = indexing[k]
                B_B_tmp    = fmul_rn(L_to_M_b,all_rhoB[ind])
                B_D1_tmp   = fmul_rn(L_to_M_d1,all_rhoD1[ind])

                v_tmp      = all_v[ind]
                v2_tmp     = all_v2[ind] 
                sigma2_tmp = all_sigma2[ind] 


               
                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(B_B_tmp,B_D1_tmp),kinpsf_tmp)) 

                v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp) + \
                                         sigma2_tmp*B_B_tmp)
                v_2_sum_B += kinpsf_tmp*(sigma2_tmp*B_B_tmp)           
                v_2_sum_D1 += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0+B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp)) 
            
            sigma_AVG[i] = sqrt(v_2_sum/Btot_sum_kin)
            sigma_AVG_B[i] = sqrt(v_2_sum_B/Btot_sum_kin)
            sigma_AVG_D1[i] = sqrt(v_2_sum_D1/Btot_sum_kin)
        else:
            pass

    else:
        pass

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#                          MODEL D+D
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
           float32,float32,float32,float32,\
           float32[::1],float32[::1],float32[::1],float32[::1],int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_rho_v_2_vis_components_DD(all_rhoD1,         # (N,J,K) contains all bulge surface density
              all_rhoD2,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              kin_psf,              # (N,J,K) psf weights
              pho_psf,              # (N,J,K) psf weights
              L_to_M_d1,
              L_to_M_d2,
              k_D1,
              k_D2,
              rho_AVG,
              v_AVG,
              sigma_AVG,
              LM_AVG,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - total surface brightness
            - mean line of sight velocity
            - mean L/M ratio
                                             
        Parameters:
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                kin_psf(1D arr float32)     : psf weight for the kinematics (of lenght K)
                pho_psf(1D arr float32)     : psf weight for the photometry (of lenght K)
                L_to_M_d1(float32)          : light to mass disc 1
                L_to_M_d2(float32)          : light to mass disc 2
                k_D1(float32)               : kinematic decomposition disc 1
                k_D2(float32)               : kinematic decomposition disc 2
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                rho_AVG(1D arr float32)     : total surface brightness
                v_AVG(1D arr float32)       : average line of sight velocity
                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion (NOT USED/COMPUTED IN THIS KERNEL)
                LM_AVG(1D arr float32)      : average light to mass ratio


    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]
        
            kinpsf_sum = 0.
            phopsf_sum = 0.

            Btot_sum_kin = 0.
            
            rhotot_sum = 0.
            Btot_sum = 0.

            v_sum = 0.

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                phopsf_tmp    = pho_psf[int(k-i*K)]
                ind         = indexing[k]
                rhoD2_tmp   = all_rhoD2[ind]
                rhoD1_tmp   = all_rhoD1[ind]
                B_D2_tmp   = fmul_rn(L_to_M_d2,rhoD2_tmp)
                B_D1_tmp   = fmul_rn(L_to_M_d1,rhoD1_tmp)

                v_tmp      = all_v[ind]
                #v2_tmp     = all_v2[ind] 
                #sigma2_tmp = all_sigma2[ind] 


                phopsf_sum    = fadd_rn(phopsf_sum,phopsf_tmp)
               
                Btot_sum    = fadd_rn(Btot_sum,fmul_rn(fadd_rn(B_D2_tmp,B_D1_tmp),phopsf_tmp)) 
                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(B_D2_tmp,B_D1_tmp),kinpsf_tmp)) 
                rhotot_sum  = fadd_rn(rhotot_sum,fmul_rn(phopsf_tmp,fadd_rn(rhoD1_tmp,rhoD2_tmp)))

                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1+B_D2_tmp*v_tmp*k_D2)
                #v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0 + sigma2_tmp*B_B_tmp)


            rho_AVG[i] = fast_fdividef(Btot_sum,phopsf_sum)

            v_AVG[i]     = fast_fdividef(v_sum,Btot_sum_kin)
            #sigma_AVG[i] = sqrt(v_2_sum/Btot_sum_kin)
            LM_AVG[i]    =  fast_fdividef(Btot_sum,rhotot_sum)
 
        else:
            pass

    else:
        pass


@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
           float32,float32,float32,float32,\
           float32[::1],float32[::1],int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_sigma_2_vis_components_DD(all_rhoD1,         # (N,J,K) contains all bulge surface density
              all_rhoD2,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              kin_psf,              # (N,J,K) psf weights
              L_to_M_d1,
              L_to_M_d2,
              k_D1,
              k_D2,
              v_AVG,
              sigma_AVG,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - mean line of sight velocity dispersion
                                             
        Parameters:
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                kin_psf(1D arr float32)     : psf weight kinematics(of lenght K)
                L_to_M_d1(float32)          : light to mass disc 1
                L_to_M_d2(float32)          : light to mass disc 2
                k_D1(float32)               : kinematic decomposition disc 1
                k_D2(float32)               : kinematic decomposition disc 2
                v_AGV(1D arr float32)       : average l.o.s velocity as computed in avg_rho_v
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion
    
    Notes:
        Dispersion accounts for beam smearing.
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]
        
            kinpsf_sum = 0.
            Btot_sum_kin = 0.
            
            v_2_sum   = 0.
            v_mean_tmp = v_AVG[i] 

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                ind         = indexing[k]
                B_D2_tmp    = fmul_rn(L_to_M_d2,all_rhoD2[ind])
                B_D1_tmp   = fmul_rn(L_to_M_d1,all_rhoD1[ind])

                v_tmp      = all_v[ind]
                v2_tmp     = all_v2[ind] 

                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(B_D2_tmp,B_D1_tmp),kinpsf_tmp)) 

                v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp) + \
                                         B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0 +\
                                         B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp))


            sigma_AVG[i] = sqrt(v_2_sum/Btot_sum_kin)
        else:
            pass

    else:
        pass


@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
           float32,float32,float32,float32,\
           float32[::1],float32[::1],float32[::1],float32[::1],\
           float32[::1],float32[::1],float32[::1],float32[::1],\
           float32[::1],int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_rho_v_2_vis_components_DD_all_output(all_rhoD1,         # (N,J,K) contains all bulge surface density
              all_rhoD2,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              kin_psf,              # (N,J,K) psf weights
              pho_psf,              # (N,J,K) psf weights
              L_to_M_d1,
              L_to_M_d2,
              k_D1,
              k_D2,
              rho_AVG,
              rho_AVG_D1,
              rho_AVG_D2,
              v_AVG,
              v_AVG_D1,
              v_AVG_D2,
              LM_AVG,
              LM_AVG_D1,
              LM_AVG_D2,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total,B,D1,D2) 
            - mean line of sight velocity (total,B,D1,D2)
            - mean L/M ratio (total,B,D1,D2)
                                             
        Parameters:
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                kin_psf(1D arr float32)     : psf weight for the kinematics (of lenght K)
                pho_psf(1D arr float32)     : psf weight for the photometry (of lenght K)
                L_to_M_d1(float32)          : light to mass disc 1
                L_to_M_d2(float32)          : light to mass disc 2
                k_D1(float32)               : kinematic decomposition disc 1
                k_D2(float32)               : kinematic decomposition disc 2
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                rho_AVG(1D arr float32)     : total surface brightness
                rho_AVG_D1(1D arr float32)  : disc1 surface brightness
                rho_AVG_D2(1D arr float32)  : disc2 surface brightness
                v_AVG(1D arr float32)       : average line of sight velocity
                v_AVG_D1(1D arr float32)    : disc1 line of sight velocity
                v_AVG_D2(1D arr float32)    : disc2 line of sight velocity
                LM_AVG(1D arr float32)      : average L/M ratio
                LM_AVG_D1(1D arr float32)   : disc1 L/M ratio
                LM_AVG_D2(1D arr float32)   : disc2 L/M ratio


    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]
        
            kinpsf_sum = 0.
            phopsf_sum = 0.

            Btot_sum_kin = 0.
            
            rhotot_sum = 0.
            Btot_sum = 0.
            B_D1_sum = 0.
            B_D2_sum = 0.
            
            v_sum_D1 = 0.
            v_sum_D2 = 0.
            v_sum = 0.

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                phopsf_tmp    = pho_psf[int(k-i*K)]
                ind         = indexing[k]
                rhoD2_tmp   = all_rhoD2[ind]
                rhoD1_tmp   = all_rhoD1[ind]
                B_D2_tmp   = fmul_rn(L_to_M_d2,rhoD2_tmp)
                B_D1_tmp   = fmul_rn(L_to_M_d1,rhoD1_tmp)

                v_tmp      = all_v[ind]
                #v2_tmp     = all_v2[ind] 
                #sigma2_tmp = all_sigma2[ind] 


                phopsf_sum    = fadd_rn(phopsf_sum,phopsf_tmp)
               
                Btot_sum    = fadd_rn(Btot_sum,fmul_rn(fadd_rn(B_D2_tmp,B_D1_tmp),phopsf_tmp)) 
                B_D1_sum    = fadd_rn(B_D1_sum,fmul_rn(B_D1_tmp,phopsf_tmp)) 
                B_D2_sum    = fadd_rn(B_D2_sum,fmul_rn(B_D2_tmp,phopsf_tmp)) 

                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(B_D2_tmp,B_D1_tmp),kinpsf_tmp)) 
                
                rhotot_sum  = fadd_rn(rhotot_sum,fmul_rn(phopsf_tmp,fadd_rn(rhoD1_tmp,rhoD2_tmp)))

                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1+B_D2_tmp*v_tmp*k_D2)
                v_sum_D1     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1)
                v_sum_D2     += kinpsf_tmp*(B_D2_tmp*v_tmp*k_D2)                
                #v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0 + sigma2_tmp*B_B_tmp)


            rho_AVG[i] = fast_fdividef(Btot_sum,phopsf_sum)
            rho_AVG_D1[i] = fast_fdividef(B_D1_sum,phopsf_sum)
            rho_AVG_D2[i] = fast_fdividef(B_D2_sum,phopsf_sum)

            v_AVG[i]     = fast_fdividef(v_sum,Btot_sum_kin)
            v_AVG_D1[i]     = fast_fdividef(v_sum_D1,Btot_sum_kin)
            v_AVG_D2[i]     = fast_fdividef(v_sum_D2,Btot_sum_kin)

            LM_AVG[i]    =  fast_fdividef(Btot_sum,rhotot_sum)
            LM_AVG_D1[i]    =  fast_fdividef(Btot_sum,B_D1_sum/L_to_M_d1)
            LM_AVG_D2[i]    =  fast_fdividef(Btot_sum,B_D2_sum/L_to_M_d2)
 
        else:
            pass

    else:
        pass


@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
           float32,float32,float32,float32,\
           float32[::1],float32[::1],float32[::1],float32[::1],int16[::1],float32[::1])',
          fastmath=True,
          debug = False)
def avg_sigma_2_vis_components_DD_all_output(all_rhoD1,         # (N,J,K) contains all bulge surface density
              all_rhoD2,        # (N,J,K) contains all disc surface density
              all_v,
              all_v2,
              kin_psf,              # (N,J,K) psf weights
              L_to_M_d1,
              L_to_M_d2,
              k_D1,
              k_D2,
              v_AVG,
              sigma_AVG,
              sigma_AVG_D1,
              sigma_AVG_D2,
              indexing,
              goodpix):

    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - mean line of sight velocity dispersion
                                             
        Parameters:
                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
                all_rhoD2(1D arr float32)   : disc surf density (see rho_D)
                all_v(1D arr float32)       : line of sight velocity (see v_tot)
                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
                kin_psf(1D arr float32)     : psf weight kinematics(of lenght K)
                L_to_M_d1(float32)          : light to mass disc 1
                L_to_M_d2(float32)          : light to mass disc 2
                k_D1(float32)               : kinematic decomposition disc 1
                k_D2(float32)               : kinematic decomposition disc 2
                v_AGV(1D arr float32)       : average l.o.s velocity as computed in avg_rho_v
                indexing(1D arr int16)      : mapping for the correct convolution
                                                    for example, v_AVG[i] uses points at indexing[k]
                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
        Returns:
                sigma_AVG(1D arr float32)    : average line of sight velocity dispersion
                sigma_AVG_D1(1D arr float32) : disc1 line of sight velocity dispersion
                sigma_AVG_D2(1D arr float32) : disc2 line of sight velocity dispersion
    
    Notes:
        Dispersion accounts for beam smearing.
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (i < v_AVG.shape[0]):
        if goodpix[i] == 1.:
            K = kin_psf.shape[0]
        
            kinpsf_sum = 0.
            Btot_sum_kin = 0.
            
            v_2_sum   = 0.
            v_2_sum_D1   = 0.
            v_2_sum_D2   = 0.
            v_mean_tmp = v_AVG[i] 

            for k in range(i*K,(i+1)*K):
                kinpsf_tmp    = kin_psf[int(k-i*K)]
                ind         = indexing[k]
                B_D2_tmp    = fmul_rn(L_to_M_d2,all_rhoD2[ind])
                B_D1_tmp   = fmul_rn(L_to_M_d1,all_rhoD1[ind])

                v_tmp      = all_v[ind]
                v2_tmp     = all_v2[ind] 

                Btot_sum_kin    = fadd_rn(Btot_sum_kin,fmul_rn(fadd_rn(B_D2_tmp,B_D1_tmp),kinpsf_tmp)) 

                v_2_sum   += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0 +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp) + \
                                         B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0 +\
                                         B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp))
                v_2_sum_D1 += kinpsf_tmp*(B_D1_tmp*v2_tmp*(1.0-k_D1*k_D1)/3.0+B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp))        
                v_2_sum_D2 += kinpsf_tmp*(B_D2_tmp*v2_tmp*(1.0-k_D2*k_D2)/3.0+B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp))          


            sigma_AVG[i] = sqrt(v_2_sum/Btot_sum_kin)
            sigma_AVG_D1[i] = sqrt(v_2_sum_D1/Btot_sum_kin)
            sigma_AVG_D2[i] = sqrt(v_2_sum_D2/Btot_sum_kin)
        else:
            pass

    else:
        pass

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#                     MODEL only bulge
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Not ready to be computed with new psf convolution method

#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
#          fastmath=True,
#          debug = False)
#def avg_rho_v_only_bulge(all_rhoB,         # (N,J,K) contains all bulge surface density
#                         all_sigma2,
#                         psf,              # (N,J,K) psf weights
#                         L_to_M_b,
#                         rho_AVG,
#                         v_AVG,
#                         sigma_AVG):
#
#    ''' 
#    KERNEL
#    ----------------------------------------------------------------------
#    ----------------------------------------------------------------------
#    Compute:
#            - total surface brightness
#            - mean line of sight velocity
#            - mean L/M ratio
#                                             
#        Parameters:
#                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
#                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe/dehnen sigma)
#                kin_psf(1D arr float32)     : psf weight for the kinematics (of lenght K)
#                pho_psf(1D arr float32)     : psf weight for the photometry (of lenght K)
#                L_to_M_b(float32)           : light to mass bulge
#                L_to_M_d1(float32)          : light to mass disc 1
#                k_D1(float32)               : kinematic decomposition disc 1
#                indexing(1D arr int16)      : mapping for the correct convolution
#                                                    for example, v_AVG[i] uses points at indexing[k]
#                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
#        Returns:
#                rho_AVG(1D arr float32)     : total surface brightness
#                v_AVG(1D arr float32)       : average line of sight velocity
#                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion (NOT USED/COMPUTED IN THIS KERNEL)
#                LM_AVG(1D arr float32)      : average light to mass ratio
#
#
#    '''
#
#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#    if (i < v_AVG.shape[0]):
#        K = psf.shape[0]
#        
#        psf_sum = 0.
#
#        rhoB_sum = 0.
#
#        vB_2_sum    = 0.
#        
#        for k in range(i*K,(i+1)*K):
#            psf_tmp    = psf[int(k-i*K)]
#
#            rhoB_tmp    = L_to_M_b*all_rhoB[k]
#            
#            sigma2_tmp = all_sigma2[k] 
#            
#            
#            psf_sum    = fadd_rn(psf_sum,psf_tmp)
#            
#            rhoB_sum    = fadd_rn(rhoB_sum,fmul_rn(rhoB_tmp,psf_tmp))
#            
#            vB_2_sum    = fadd_rn(vB_2_sum,fmul_rn(psf_tmp,fmul_rn(rhoB_tmp,sigma2_tmp)))
#
#        
#        rhoB_mean  = fast_fdividef(rhoB_sum,psf_sum)
#        rho_AVG[i] = rhoB_mean
#
#        sigmaB2_mean   = fast_fdividef(vB_2_sum,rhoB_sum)
#        
#
#        v_AVG[i]     = 0.
#        sigma_AVG[i] = sqrt(sigmaB2_mean)
#
#    else:
#        pass
#
#
#
#@cuda.jit('void(float32[::1],float32[::1],float32,float32[::1])',
#          fastmath=True,
#          debug = False)
#def avg_LM_only_bulge(all_rhoB,        # (N,J,K) contains all bulge surface density
#                      psf,             # (N,J,K) psf weights
#                      L_to_M_bulge,
#                      LM_AVG):
#
#    ''' 
#    KERNEL
#    ----------------------------------------------------------------------
#    ----------------------------------------------------------------------
#    Compute:
#            - mean line of sight velocity dispersion
#                                             
#        Parameters:
#                all_rhoB(1D arr float32)    : bulge surf density (see herquinst_rho or jaffe_rho)
#                all_rhoD1(1D arr float32)   : disc surf density (see rho_D)  
#                all_v(1D arr float32)       : line of sight velocity (see v_tot)
#                all_v2(1D arr float32)      : circular velocity squared (see v_tot)
#                all_sigma2(1D arr float32)  : velocity dispersion squared of the bulge (see hernquist/jaffe sigma)
#                kin_psf(1D arr float32)     : psf weight kinematics(of lenght K)
#                L_to_M_b(float32)           : light to mass bulge
#                L_to_M_d1(float32)          : light to mass disc 1
#                k_D1(float32)               : kinematic decomposition disc 1
#                v_AGV(1D arr float32)       : average l.o.s velocity as computed in avg_rho_v
#                indexing(1D arr int16)      : mapping for the correct convolution
#                                                    for example, v_AVG[i] uses points at indexing[k]
#                goodpix(1D arr float32)     : goodpix=0 means skip that pixel 
#        Returns:
#                sigma_AVG(1D arr float32)   : average line of sight velocity dispersion
#    
#    Notes:
#        Dispersion accounts for beam smearing.
#    '''
#
#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#    if (i < LM_AVG.shape[0]):
#
#        LM_AVG[i] = L_to_M_bulge
#
#    else:
#        pass

@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1])',
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
                   goodpix,
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
                ibrightness_data(1D arr float32) : data brightness (in log10)
                v_data(1D arr float32)           : velocity data
                sigma_data(1D arr float32)       : sigma data
                ML_data(1D arr float32)          : mass to light data
                ibrightness_err(1D arr float32)  : brightness errror (in log10)
                v_err(1D arr float32)            : velocity error
                sigma_err(1D arr float32)        : velocity dispersion error
                ML_err(1D arr float32)           : mass to light  error
                sys_rho(float32)                 : systematic error for surface brightness (it is added as error on logarithm)
        Returns:
                likelihood(1D arr float32)       : likelihood (it is a vector of one element)    +

    Notes:
        comment/uncomment final lines for excluding mass to ligh ratio from the fit.
        This should be added to the configuration file.        
    '''

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    likelihood[0] = 0.
    if i < rho_model.shape[0]:
        if goodpix[i] == 1.0:
            if isnan(ibrightness_data[i]) == True:
                rho_lk = 0.
            else:
                rho_err_tmp  = ibrightness_err[i]
                rho_err_tmp  = fmul_rn(rho_err_tmp,rho_err_tmp) + (sys_rho/10**ibrightness_data[i]/log(10))*(sys_rho/10**ibrightness_data[i]/log(10))
                delta_rho    = fsub_rn(ibrightness_data[i],log10(rho_model[i]))
                rho_lk       = fadd_rn(fast_fdividef(fmul_rn(delta_rho,delta_rho),rho_err_tmp),log(fmul_rn(2,fmul_rn(M_PI,rho_err_tmp))))

            if isnan(v_data[i]) == True:
                v_lk  = 0.
            else:
                v_err_tmp = v_err[i]
                delta_v   = fsub_rn(v_data[i],v_model[i])
                v_err_tmp = fmul_rn(v_err_tmp,v_err_tmp)
                v_lk      = fadd_rn(fast_fdividef(fmul_rn(delta_v,delta_v),v_err_tmp),log(fmul_rn(fmul_rn(2,M_PI),v_err_tmp)))

            if isnan(sigma_data[i]) == True:
                sigma_lk = 0.
            else:
                sigma_err_tmp = sigma_err[i]
                delta_sigma = fsub_rn(sigma_data[i],sigma_model[i])
                sigma_err_tmp = fmul_rn(sigma_err_tmp,sigma_err_tmp)
                sigma_lk = fadd_rn(fast_fdividef(fmul_rn(delta_sigma,delta_sigma),sigma_err_tmp),log(fmul_rn(fmul_rn(2,M_PI),sigma_err_tmp)))

            if isnan(ML_data[i]) == True:
                ML_lk = 0.
            else:
                ML_err_tmp = ML_err[i]
                ML_err_tmp = fmul_rn(ML_err_tmp,ML_err_tmp) 
                delta_ML = fsub_rn(ML_data[i],(1./LM_model[i]))
                ML_lk = fadd_rn(fast_fdividef(fmul_rn(delta_ML,delta_ML),ML_err_tmp),log(fmul_rn(fmul_rn(2,M_PI),ML_err_tmp)))

            tmp = fmul_rn(-0.5,fadd_rn(fadd_rn(fadd_rn(rho_lk,v_lk),sigma_lk),ML_lk))
            cuda.atomic.add(likelihood,0,tmp)
        else:
            pass
    else:
        pass


if __name__=='__main__':
    pass