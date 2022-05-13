#Standard python imports
# cython: profile=True
# cython: infer_types=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from __future__ import division
import  numpy as np
cimport numpy as np
cimport cython
#from scipy.special.cython_special cimport i0e, i1e, k0e, k1e
from libc.math cimport M_PI, sin, cos, exp, log, sqrt, acos, atan2,log10,isnan,nan

#ctypedef np.float64_t my_type
ctypedef double my_type

'''Units are:
            - M_sun/kpc^2 for surface densities
            - km/s for velocities and dispersions
G = 4.299e-6
'''

#---------------------------------------------------------------------------
                    #Generic
#---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type my_bilinear(my_type x1,
                        my_type x2,
                        my_type y1,
                        my_type y2,
                        my_type f11,
                        my_type f12,
                        my_type f21,
                        my_type f22,
                        my_type x,
                        my_type y) nogil:
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
    cdef my_type den = 0.
    cdef my_type num = 0.
    cdef my_type frac = 0.
    den = (x2-x1)*(y2-y1)
    num = f11*(x2-x)*(y2-y)+f21*(x-x1)*(y2-y)+\
          f12*(x2-x)*(y-y1)+f22*(x-x1)*(y-y1)
    frac = num/den
    return frac

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type my_i0e(my_type x) nogil:
    '''
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    0th order modified Bessel function of the first kind exponentially scaled.
    For asymptotic expansion see Abramowitz and Stegun.

        Parameters:
                x(float32): float number 

        Returns: 
                ans(float32): 0th order modified Bessel function of the first kind exponentially scaled 
    '''
    cpdef my_type y = 0.
    cpdef my_type ans = 0.
    if ( x < 3.75): 
        y=x/3.75
        y*=y
        ans=exp(-x)*(1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2))))))
    else: 
        y=3.75/x
        ans=(1/sqrt(x))*(0.39894228+y*(0.1328592e-1+y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2+y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1+y*0.392377e-2))))))))
        
    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type my_k0e(my_type x) nogil:
    '''
    CPU FUNCTION
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
    cdef my_type y = 0.
    cdef my_type ans = 0.
    if (x <= 2.0) :
        y=x*x/4.0
        ans=exp(x)*((-log(x/2.0)*exp(x)*my_i0e(x))+(-0.57721566+y*(0.42278420+y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2+y*(0.10750e-3+y*0.74e-5)))))))
    else :
        y=2.0/x
        ans=(1./sqrt(x))*(1.25331414+y*(-0.7832358e-1+y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2+y*(-0.251540e-2+y*0.53208e-3))))))
        
    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type my_i1e(my_type x) nogil:
    '''
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    1st order modified Bessel function of the first kind exponentially scaled.
    For asymptotic expansion see Abramowitz and Stegun. 

        Parameters:
                x(float32): float number 

        Returns: 
                ans(float32): 1st order modified Bessel function of the first kind exponentially scaled 
    '''
    cdef my_type y = 0.
    cdef my_type ans = 0.
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type my_k1e(my_type x) nogil:
    '''
    CPU FUNCTION
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
    cdef my_type y = 0.
    cdef my_type ans = 0.
    if (x <= 2.0):
        y=x*x/4.0
        ans=exp(x)*((log(x/2.0)*exp(x)*my_i1e(x))+(1.0/x)*(1.0+y*(0.15443144+y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1+y*(-0.110404e-2+y*(-0.4686e-4))))))))
    else:
        y=2.0/x
        ans=(1/sqrt(x))*(1.25331414+y*(0.23498619+y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2+y*(0.325614e-2+y*(-0.68245e-3)))))))
    
    return ans 



#---------------------------------------------------------------------------
                    #DEHNEN
#---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef HERQUINST dehnen_function(my_type Mb,my_type Rb,my_type gamma,my_type r,\
                      my_type[::1] s_grid,my_type[::1] gamma_grid,my_type[::1] rho_grid,my_type[::1] sigma_grid,
                      my_type central_integration_radius,int central_index,\
                      my_type[::1] central_rho,my_type[::1] central_sigma) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Dehnen (see Dehnen 1993 or Tremaine 1993) projected surface density and velocity isotropic
    dispersion.  
                                             
        Parameters:
                Mb(float32)                         : bulge mass
                Rb(float32)                         : bulge scale radius
                gamma(float32)                      : inner slope of the density profile 
                r(float32)                          : projected radius (see r_proj in coordinate_transformation)
                s_grid(1D arr float32)              : R/Rb of the grid used for interpolation (see dehnen/s_grid.npy) 
                gamma_grid(1D arr float32)          : gamma of the grid used for interpolation (see dehnen/gamma_grid.npy) 
                rho_grid(1D arr float32)            : brightness of the grid used for interpolation (see dehnen/dehnen_brightness.npy) 
                sigma_grid(1D arr float32)          : velocity dispersion of the grid used for interpolation (see dehnen/dehnen_sigma2.npy) 
                central_integration_radius(float32) : radius of the central pixel for computing the average of brightness and dispersion
                central_index(int16)                : integer if 1 the pixel is the central one otherwise no
                central_rho(1D arr float32)         : cumulative brightness of the grid used for interpolation (see dehnen/dehnen_luminosity.npy) 
                central_sigma(1D arr float32)       : cumulative dispersion of the grid used for interpolation (see dehnen/dehnen_avg_sigma2.npy)
        Returns:
                rhoB(float32)      : Dehnen projected surface density 
                sigmaB2(float32)   : Dehnen projected velocity dispersion 

    Notes:
        The brightness and sigma are computed by interpolation. 
        Since the central pixel can cause problems we approximate the brightness and the dispersion in the central point
        as the average on the spatial extension of the pixel (i.e. the surface brightness is the luminosity/area of the pixel).

    Improvements:
        Knowing the spatial extension of the grid in (s;gamma) we can compute directly their value reducing the number of memory accesses.
        The ID of the central pixel is computed using np.argmin on the projected radius but in principles it can be computed "a priori".
    '''


    cdef my_type G = 4.299e-6
    cdef my_type logs_min=-3
    cdef my_type gamma_min=0.
    cdef my_type dlogs=0.06060606060606055
    cdef my_type dgamma=0.17142857142857143
    cdef int N=15   
    cdef Py_ssize_t m,l,k1,k2 
    cdef my_type y1 = 0.
    cdef my_type y2 = 0.
    cdef my_type s = 0.
    cdef my_type Rb2 = 0.
    cdef my_type s2 = 0.
    cdef my_type s3 = 0.
    cdef my_type s4 = 0.
    cdef my_type s5 = 0.
    cdef my_type x1 = 0.
    cdef my_type x2 = 0.
    cdef my_type rhoB_tmp = 0.
    cdef my_type sigma_tmp = 0.
    cdef HERQUINST dehnen
    dehnen.rho = 0.
    dehnen.sigma = 0.

    m = int((gamma-gamma_min)/dgamma)  
    y1 = gamma_grid[m]
    y2 = gamma_grid[m+1]
    if central_index == 0:
        s = r/Rb
        Rb2 = Rb*Rb
        if s>1e3:
            s2 = s*s
            s3 = s2*s
            s4 = s2*s2
            s5 = s3*s2
            rhoB_tmp = (((Mb/Rb2)*(3-gamma))/M_PI)* ( (M_PI/8.)/s3 + ((gamma-4.)/3.)/s4 + \
                                                3.*M_PI*(gamma-5.)*(gamma-4.)/(32.*s5) )
            sigma_tmp = (G*Mb/Rb) * ( 0.16877/s + (0.14410*(gamma-4.)-0.125*(7.-2.*gamma))/s2 +\
                                   ((4.-gamma)/s3)*(-0.12232*(gamma-4.)+0.02782*(3.*gamma-14.)) ) 
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
    if central_index == 1:
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

            rhoB_tmp   = Mb*(1.0-2.0*(3.0-gamma)*((1+s)**(gamma-2.5))*(10**my_bilinear(x1,x2,y1,y2,\
                                   central_rho[k],central_rho[k+1],central_rho[k1],central_rho[k1+1],s,gamma)))

            sigmaB_tmp = 10**my_bilinear(x1,x2,y1,y2,\
                                   central_sigma[k],central_sigma[k+1],central_sigma[k1],central_sigma[k1+1],s,gamma)
            
            sigmaB_tmp = (G*Mb*Mb/Rb)*(1.-4.*((1.+s)**(2.*gamma-4.5))*(3.-gamma)*(5.-2.*gamma)*sigmaB_tmp)/(6.*(5.-2.*gamma))
            sigmaB_tmp = sigmaB_tmp/rhoB_tmp
            rhoB_tmp   = rhoB_tmp/(M_PI*central_integration_radius*central_integration_radius)

    dehnen.rho = rhoB_tmp
    dehnen.sigma = sigmaB_tmp

    return dehnen

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type v_dehnen(my_type Mb,        # bulge mass
                      my_type Rb,        # bulge radius
                      my_type gamma,
                      my_type r)nogil:    
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Denhen circular velocity squared.
                                             
        Parameters:
                Mb(float32)         : bulge mass
                Rb(float32)         : bulge scale radius
                gamma(float32)      : inner slope of the bulge profile
                r(float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                v_B(float32)   : Dehnen circular velocity squared. 

    '''


    cdef my_type G = 4.299e-6
    cdef my_type v_B = 0.

    v_B = G*Mb*r**(2.-gamma)/(r+Rb)**(3.-gamma)
    return v_B 

#---------------------------------------------------------------------------
                    #NFW
#---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type v_NFW (my_type M_halo,        # bulge/halo mass
                  my_type c,        # bulge/halo mass
                  my_type rho_crit,
                  my_type r) nogil:         # true radius
    ''' 
    CPU funtionc
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute NFW circular velocity squared.
                                             
        Parameters:
                Mh(float32)         : halo mass
                c(float32)          : concentration r200/R_halo
                rho_crit(float32)   : 3*H(z)^2/(8piG) critical density at that redshift
                r(float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                v_H(float32)   : Halo(NFW) circular velocity squared. 
    
    Notes: 
        The mass of the NFW halo diverges so we assume the following:
        M_halo = M_200   and    M(r=r200) = 200*Mcrit = 200*(4/3pi*r200^3)rho_crit

    '''
    cdef my_type G = 4.299e-6                          # kpc*km^2/M_sun/s^2
    cdef my_type r_tmp_H = 0.
    cdef my_type a = 0.
    cdef my_type v_H = 0.
    cdef my_type v_H2 = 0.
    cdef my_type one_plus_r_H = 0.

    a = ((3.*M_halo/(800.*M_PI*rho_crit))**(1./3.))/c
    r_tmp_H = r/a
    one_plus_r_H = 1.0+r_tmp_H
    v_H2 = log(one_plus_r_H)-(r_tmp_H/one_plus_r_H) 
    v_H = (v_H2*G*M_halo/r)/(log(1+c)-c/(1+c))
    return v_H
#---------------------------------------------------------------------------
                    #HERQUINST
#---------------------------------------------------------------------------
'''
STRUCTURE 
----------------------------------------------------------------------
----------------------------------------------------------------------
Structure to save/compute rho and sigma of bulge togheter
'''
cdef struct HERQUINST:
    my_type rho 
    my_type sigma 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline my_type radius(my_type x, my_type y) nogil :
    '''
    CPU FUNCTION
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type Xfunction(my_type s) nogil :
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute X(s) function useful for Hernquist/Jaffe surface density and 
    dispersion (see Hernquist 1990) 
                                             
        Parameters:
                s (float32)              : R_proj/Rb

        Returns:
                X(float32)   : X(s) function 

    '''
    cdef my_type one_minus_s2 = 1.0-s*s
    if s < 1.0:
        return log((1.0 + sqrt(one_minus_s2)) / s) / sqrt(one_minus_s2)
    elif s ==  1.0:
        return 1.0
    else:
        return acos(1.0/s)/sqrt(-one_minus_s2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type X1dot(my_type s) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                s (float32)              : R_proj/Rb

        Returns:
                X1_dot(float32)   : first derivative of X 

    '''
    cdef my_type X1_dot = 0.
    cdef my_type s_minus_one1 = s-1.0
    cdef my_type s_minus_one2 = s_minus_one1*s_minus_one1
    cdef my_type s_minus_one3 = s_minus_one2*s_minus_one1
    cdef my_type s_minus_one4 = s_minus_one2*s_minus_one2
    if 0.93<=s<=1.07:
        X1_dot = -2./3. + 14.*s_minus_one1/15. - 36./35.*s_minus_one2 + 332./315.*s_minus_one3 -730./693.*s_minus_one4
    else:
        X1_dot = (1./s - s*Xfunction(s))/(s*s-1.)

    return X1_dot

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type X2dot(my_type s) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                s (float32)              : R_proj/Rb

        Returns:
                X2_dot(float32)   : second derivative of X 

    '''
    cdef my_type X2_dot = 0.
    cdef my_type s_minus_one1 = s-1.0
    cdef my_type s_minus_one2 = s_minus_one1*s_minus_one1
    cdef my_type s_minus_one3 = s_minus_one2*s_minus_one1
    cdef my_type s_minus_one4 = s_minus_one2*s_minus_one2
    cdef my_type s2    = s*s
    cdef my_type s2_minus_one1 = s2-1.0
    if 0.93<=s<=1.07:
        X2_dot = 14./15. - 72.*s_minus_one1/35. + 332./105.*s_minus_one2 - 2920./693.*s_minus_one3 + 5230./1001.*s_minus_one4
    else:
        X2_dot = (-4.0+1.0/s2+(1.+2.*s2)*Xfunction(s))/(s2_minus_one1*s2_minus_one1)

    return X2_dot

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type X3dot(my_type s) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                s (float32)              : R_proj/Rb

        Returns:
                X3_dot(float32)   : third derivative of X 

    '''
    cdef my_type X3_dot = 0.
    cdef my_type s_minus_one1 = s-1.0
    cdef my_type s_minus_one2 = s_minus_one1*s_minus_one1
    cdef my_type s_minus_one3 = s_minus_one2*s_minus_one1
    cdef my_type s_minus_one4 = s_minus_one2*s_minus_one2
    cdef my_type s2    = s*s
    cdef my_type s3    = s*s2
    cdef my_type s2_minus_one1 = s2-1.0
    if 0.93<=s<=1.07:
        X3_dot = -72./35. + 664.*s_minus_one1/105. - 2920./231.*s_minus_one2 + 20920./1001.*s_minus_one3 - 13328./429.*s_minus_one4
    else:
        X3_dot = (18.*s-5./s+2./s3-3.*s*(3.+2.*s2)*Xfunction(s))/(s2_minus_one1*s2_minus_one1*s2_minus_one1)

    return X3_dot

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type Y1(my_type s) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                s (float32)              : R_proj/Rb

        Returns:
                Y_1(float32)   : recurrence relation for BH_herq sigma

    '''
    cdef my_type s2 = s*s
    cdef my_type Y_1 = 0.
    Y_1 = (M_PI + 16.*s - 12.*M_PI*s2 + 4.*s*(8.*s2-3.)*Xfunction(s) + (8.*s2*s2-6.*s2)*X1dot(s))/(8.*M_PI*s)

    return Y_1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type Y2(my_type s) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    MISSING DOCUMENTATION                                             
        Parameters:
                s (float32)              : R_proj/Rb

        Returns:
                Y_2(float32)   : recurrence relation for BH_herq sigma

    '''


    cdef my_type s2 = s*s
    cdef my_type Y_2 = 0.
    Y_2 = (-12.*M_PI+96.*s*Xfunction(s) + 18.*(8.*s2-1.)*X1dot(s)+(-18.*s+48.*s2*s)*X2dot(s)+(-3.*s2+4.*s2*s2)*X3dot(s))*s/(24.*M_PI)

    return Y_2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef HERQUINST herquinst_function(my_type M, my_type Rb, my_type r) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Hernquist projected surface density and velocity dispersion squared.  
                                             
        Parameters:
                M(float32)      : bulge mass
                Rb(float32)     : bulge scale radius
                r(float32)      : projected radius 
        Returns:
                HERQUINST  struct(float32;float32) : Struct containing Surface density and dispersione 

    '''
    cdef HERQUINST herquinst
    herquinst.rho = 0.
    herquinst.sigma = 0.
    cdef my_type Rb2 = Rb*Rb 
    cdef my_type s = r/Rb
    cdef my_type s2 = 0.
    cdef my_type s3 = 0.
    cdef my_type s4 = 0.
    cdef my_type s6 = 0.
    cdef my_type one_minus_s2 = 0.
    cdef my_type one_minus_s2_2 = 0.
    cdef my_type A = 0.
    cdef my_type B= 0.
    cdef my_type C = 0.
    cdef my_type D = 0.
    cdef my_type sigma = 0.
    cdef my_type X = 0.0
    cdef my_type rho = 0.
    cdef my_type M_PI2 = M_PI*M_PI
    cdef my_type M_PI3 = M_PI2*M_PI
    cdef my_type s_minus_one = 0.
    cdef my_type square_s_minus_one = 0.
    cdef my_type cube_s_minus_one = 0.
    cdef my_type fourth_s_minus_one = 0.
    cdef my_type fifth_s_minus_one = 0.

    cdef my_type G = 4.299e-6
    if (M == 0.) or (Rb == 0.): 
        return herquinst 
    
    if s >= 0.7 and s <=1.3:
        s_minus_one = s-1.0
        square_s_minus_one = s_minus_one*s_minus_one
        cube_s_minus_one = square_s_minus_one*s_minus_one
        fourth_s_minus_one = square_s_minus_one*square_s_minus_one
        fifth_s_minus_one = cube_s_minus_one*square_s_minus_one
        rho = M / (2.0*M_PI*Rb2) *( 4.0/15.0 - 16.0*s_minus_one/35.0 + (8.0* square_s_minus_one)/15.0 - (368.0* cube_s_minus_one)/693.0 + (1468.0* fourth_s_minus_one)/3003.0 - (928.0* fifth_s_minus_one)/2145.0)
        sigma = (G*M/Rb) * ( (332.0-105.0*M_PI)/28.0 + (18784.0-5985.0*M_PI)*s_minus_one/588.0 + (707800.0-225225.0*M_PI)*square_s_minus_one/22638.0 )

    else:
        X = Xfunction(s)
        s2 = s*s
        one_minus_s2 = (1.0 - s2)
        one_minus_s2_2 = one_minus_s2*one_minus_s2
        rho = (M / (2.0 * M_PI * Rb2 * one_minus_s2_2)) * ((2.0 + s2) * X - 3.0) 
        if s >= 12.5:
            s3 = s*s2
            A = (G * M * 8.0) / (15.0 * s * M_PI * Rb)
            B = (8.0/M_PI - 75.0*M_PI/64.0) / s
            C = (64.0/(M_PI2) - 297.0/56.0) / s2
            D = (512.0/(M_PI3) - 1199.0/(21.0*M_PI) - 75.0*M_PI/512.0) / s3
            sigma = A * (1 + B + C + D)   
        else:
            s4 = s2*s2 
            s6 = s4*s2
            A = (G * M*M ) / (12.0 * M_PI * Rb2 * Rb * rho)
            B = 1.0 /  (2.0*one_minus_s2_2*one_minus_s2)
            C = (-3.0 * s2) * X * (8.0*s6 - 28.0*s4 + 35.0*s2 - 20.0) - 24.0*s6 + 68.0*s4 - 65.0*s2 + 6.0
            D = 6.0*M_PI*s
            sigma =  A*(B*C-D)

    herquinst.rho = rho 
    herquinst.sigma = sigma

    return herquinst

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef HERQUINST herquinst_function_central_point(my_type M, my_type Rb, my_type r,my_type central_integration_radius) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Hernquist projected surface density and velocity dispersion squared.  
                                             
        Parameters:
                M(float32)      : bulge mass
                Rb(float32)     : bulge scale radius
                r(float32)      : projected radius 
        Returns:
                HERQUINST  struct(float32;float32) : Struct containing Surface density and dispersione 

    '''
    cdef HERQUINST herquinst
    herquinst.rho = 0.
    herquinst.sigma = 0.
    cdef my_type Rb2 = Rb*Rb 
    cdef my_type s = r/Rb
    cdef my_type s2 = 0.
    cdef my_type s3 = 0.
    cdef my_type s4 = 0.
    cdef my_type s6 = 0.
    cdef my_type one_minus_s2 = 0.
    cdef my_type one_minus_s2_2 = 0.
    cdef my_type A = 0.
    cdef my_type B= 0.
    cdef my_type C = 0.
    cdef my_type D = 0.
    cdef my_type sigma = 0.
    cdef my_type X = 0.0
    cdef my_type rho = 0.
    cdef my_type M_PI2 = M_PI*M_PI
    cdef my_type M_PI3 = M_PI2*M_PI
    cdef my_type s_minus_one = 0.
    cdef my_type square_s_minus_one = 0.
    cdef my_type cube_s_minus_one = 0.
    cdef my_type fourth_s_minus_one = 0.
    cdef my_type fifth_s_minus_one = 0.

    cdef my_type G = 4.299e-6
    if (M == 0.) or (Rb == 0.): 
        return herquinst 
    
    s = 1.4

    if s >= 0.7 and s <=1.3:
        s_minus_one = s-1.0
        square_s_minus_one = s_minus_one*s_minus_one
        cube_s_minus_one = square_s_minus_one*s_minus_one
        fourth_s_minus_one = square_s_minus_one*square_s_minus_one
        fifth_s_minus_one = cube_s_minus_one*square_s_minus_one
        rho = M / (2.0*M_PI*Rb2) *( 4.0/15.0 - 16.0*s_minus_one/35.0 + (8.0* square_s_minus_one)/15.0 - (368.0* cube_s_minus_one)/693.0 + (1468.0* fourth_s_minus_one)/3003.0 - (928.0* fifth_s_minus_one)/2145.0)
        sigma = (G*M/Rb) * ( (332.0-105.0*M_PI)/28.0 + (18784.0-5985.0*M_PI)*s_minus_one/588.0 + (707800.0-225225.0*M_PI)*square_s_minus_one/22638.0 )

    else:
        X = Xfunction(s)
        s2 = s*s
        one_minus_s2 = (1.0 - s2)
        one_minus_s2_2 = one_minus_s2*one_minus_s2
        rho = (M / (2.0 * M_PI * Rb2 * one_minus_s2_2)) * ((2.0 + s2) * X - 3.0) 
        if s >= 12.5:
            s3 = s*s2
            A = (G * M * 8.0) / (15.0 * s * M_PI * Rb)
            B = (8.0/M_PI - 75.0*M_PI/64.0) / s
            C = (64.0/(M_PI2) - 297.0/56.0) / s2
            D = (512.0/(M_PI3) - 1199.0/(21.0*M_PI) - 75.0*M_PI/512.0) / s3
            sigma = A * (1 + B + C + D)   
        else:
            s4 = s2*s2 
            s6 = s4*s2
            A = (G * M*M ) / (12.0 * M_PI * Rb2 * Rb * rho)
            B = 1.0 /  (2.0*one_minus_s2_2*one_minus_s2)
            C = (-3.0 * s2) * X * (8.0*s6 - 28.0*s4 + 35.0*s2 - 20.0) - 24.0*s6 + 68.0*s4 - 65.0*s2 + 6.0
            D = 6.0*M_PI*s
            sigma =  A*(B*C-D)

    s = central_integration_radius/Rb 
    rho = M / (M_PI*Rb2) /(1.0-s*s) * (Xfunction(s)-1.0)
    
    herquinst.rho = rho 
    herquinst.sigma = sigma

    return herquinst

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef HERQUINST herquinst_functionBH(my_type Mbh, my_type Mb, my_type Rb, my_type r)nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Hernquist+BH projected surface density and velocity dispersion squared.  
                                             
        Parameters:
                Mbh(float32)    : black hole mass
                Mb(float32)      : bulge mass
                Rb(float32)     : bulge scale radius
                r(float32)      : projected radius 
        Returns:
                HERQUINST  struct(float32;float32) : Struct containing Surface density and dispersione 

    '''
    
    cdef HERQUINST herquinst
    herquinst.rho = 0.
    herquinst.sigma = 0.
    cdef my_type Rb2 = Rb*Rb 
    cdef my_type s = r/Rb
    cdef my_type s2 = 0.
    cdef my_type s3 = 0.
    cdef my_type s4 = 0.
    cdef my_type s6 = 0.
    cdef my_type one_minus_s2 = 0.
    cdef my_type one_minus_s2_2 = 0.
    cdef my_type A = 0.
    cdef my_type B= 0.
    cdef my_type C = 0.
    cdef my_type D = 0.
    cdef my_type sigma = 0.
    cdef my_type X = 0.0
    cdef my_type rho = 0.
    cdef my_type M_PI2 = M_PI*M_PI
    cdef my_type M_PI3 = M_PI2*M_PI
    cdef my_type s_minus_one = 0.
    cdef my_type square_s_minus_one = 0.
    cdef my_type cube_s_minus_one = 0.
    cdef my_type fourth_s_minus_one = 0.
    cdef my_type fifth_s_minus_one = 0.

    cdef my_type G = 4.299e-6
    cdef my_type Rb3   = Rb*Rb*Rb 
    cdef my_type norm1 = G*Mb*Mb/Rb3
    cdef my_type norm2 = G*Mb/Rb3

    if (Mb == 0.) or (Rb == 0.): 
        return herquinst 
    
    #if s == 0:
    #    s = 1e-8

    #if s >= 0.7 and s <=1.3:
    #    s_minus_one = s-1.0
    #    square_s_minus_one = s_minus_one*s_minus_one
    #    cube_s_minus_one = square_s_minus_one*s_minus_one
    #    fourth_s_minus_one = square_s_minus_one*square_s_minus_one
    #    fifth_s_minus_one = cube_s_minus_one*square_s_minus_one
    #    rho = Mb / (2.0*M_PI*Rb2) *( 4.0/15.0 - 16.0*s_minus_one/35.0 + (8.0* square_s_minus_one)/15.0 - (368.0* cube_s_minus_one)/693.0 + (1468.0* fourth_s_minus_one)/3003.0 - (928.0* fifth_s_minus_one)/2145.0)
    #    sigma = (norm1*Y2(s)+norm2*Mbh*2*Y1(s))/rho
#
    #else:
    X = Xfunction(s)
    s2 = s*s
    one_minus_s2 = (1.0 - s2)
    one_minus_s2_2 = one_minus_s2*one_minus_s2
    rho = (Mb / (2.0 * M_PI * Rb2 * one_minus_s2_2)) * ((2.0 + s2) * X - 3.0) 
    if s >= 7.5:
        s3 = s*s2
        sigma = (-157.*G*Mb/(105.*M_PI*Rb) + 8.*G*Mb*(-2.5 + 64./M_PI2)/(15.*M_PI*Rb) - 47.*G*Mbh/(21.*M_PI*Rb) + \
                8.*G*Mbh*(-2.5 + 64./M_PI2)/(15.*M_PI*Rb))/s3 + (-5.*G*Mb/(8.*Rb) + 64.*G*Mb/(15.*M_PI2*Rb) - \
                3.*G*Mbh/(8.*Rb) + 64.*G*Mbh/(15.*M_PI2*Rb))/s2 + (8.*G*Mb/(15.*M_PI*Rb) + 8.*G*Mbh/(15.*M_PI*Rb))/s
    else:
        sigma = (norm1*Y2(s)+norm2*Mbh*2*Y1(s))/rho

    herquinst.rho = rho 
    herquinst.sigma = sigma

    return herquinst



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type v_H(my_type Mb,my_type Rb,my_type r) nogil:
    #circular v^2 for herquinst (actually bulge and halo)
    cdef my_type v_c2 = 0.
    cdef my_type G = 4.299e-6
    if Mb == 0. or Rb == 0.:
        return v_c2
    cdef my_type r_tmp = r/Rb
    cdef my_type one_plus_r = 1.0 + r_tmp
    v_c2 = G * Mb * r_tmp / (Rb*one_plus_r*one_plus_r )
    
    return v_c2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type v_BH(my_type Mbh,        # bulge mass
                  my_type r) nogil:         # true radius
    ''' 
    KERNEL
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute BH circular velocity squared.
                                             
        Parameters:
                Mbh(float32)        : BH mass
                r(float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                vBH(float32)  : BH velocities

    '''

    cdef my_type G = 4.299e-6
    cdef my_type vBH = 0.
    vBH = G*Mbh/(r)

    return vBH

#---------------------------------------------------------------------------
                    #JAFFE
#---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type jaffe_rho(my_type Mb,my_type Rb,my_type r) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Jaffe projected surface density.  
                                             
        Parameters:
                Mb(float32)          : bulge mass
                Rb(float32)          : bulge scale radius
                r(float32)           : projected radius (see r_proj in coordinate_transformation)
        Returns:
                rhoB(float32)   : Jaffe surface density 

    '''
    cdef my_type s = r/Rb
    cdef my_type rho_tmp  = 0.
    cdef my_type s2,s3,s4 = 0.
    cdef my_type rho = 0.

    if 0.99<s<1.01:
        rho_tmp = 0.25-2.0/3.0*M_PI
    else:
        if s>300.0:
            s2 = s*s
            s3 = s2*s 
            s4 = s2*s2
            rho_tmp = 1.0/(s3*8.0)-2.0/(3*M_PI*s4) 
        else:
            s2 = s*s
            X  = Xfunction(s)
            rho_tmp = 1.0/(4.0*s) + (1.0-(2.0-s2)*X)/(2.0*M_PI*(1.0-s2)) 
    
    rho = Mb/(Rb*Rb)*rho_tmp 
    return rho

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type jaffe_sigma(my_type Mb,my_type Rb,my_type r) nogil:
    ''' 
    CPU funtion
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Jaffe projected velocity dispersion.  
                                             
        Parameters:
                M(float32)          : bulge mass
                Rb(float32)         : bulge scale radius
                r(float32)   : projected radius (see r_proj in coordinate_transformation)
        Returns:
                sigmaB2(float32)   : Jaffe projected velocity dispersion squared 

    '''
    cdef my_type s = r/Rb
    cdef my_type rho_tmp  = 0.
    cdef my_type s2,s3,s4 = 0.
    cdef my_type sigma_tmp = 0.
    cdef my_type M_PI2 = M_PI*M_PI
    cdef my_type A,B,X = 0.
    cdef my_type G = 4.299e-6

    if 0.99<s<1.01:
        rho_tmp = 0.25-2.0/3.0*M_PI
        sigma_tmp = (G*Mb/(4.0*M_PI*Rb*rho_tmp))*(-0.15372602*s+0.20832732)
    else:
        if s>=4.0:
            s2 = s*s
            s3 = s2*s
            sigma_tmp = G*Mb/(8.0*Rb*M_PI)*( 64.0/(15.0*s) + (1024.0/(45.0*M_PI)-3.0*M_PI)/s2 - 128.0*(-896.0+81.0*M_PI2)/(945.0*M_PI2*s3) ) 
        else:
            s2 = s*s
            s3 = s2*s
            s4 = s2*s2
            X  = Xfunction(s)
            rho_tmp = 1.0/(4.0*s) + (1.0-(2.0-s2)*X)/(2.0*M_PI*(1.0-s2)) 
            A = (M_PI/(2*s)+11.0-6.5*M_PI*s-12.0*s2+6.0*M_PI*s3-X*(6.0-19.0*s2+12.0*s4))/(1.0-s2)
            B = G*Mb/(4*M_PI*Rb*rho_tmp)
            sigma_tmp = A*B
    return sigma_tmp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type jaffe_vc(my_type Mb,my_type Rb,my_type r) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute Jaffe(bulge) circular velocity squared.
                                             
        Parameters:
                Mb(float32)         : bulge mass
                Rb(float32)         : bulge scale radius
                r(float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                v_c2(float32)   : Jaffe(bulge) circular velocity squared. 

    '''
    cdef my_type v_c2 = 0.
    cdef my_type G = 4.299e-6

    v_c2 = G*Mb/(r+Rb)
    return v_c2
#---------------------------------------------------------------------------
                    #EXP DISC
#---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type myv_D(my_type Md,my_type Rd,my_type r) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute exp disc circular velocity squared.
                                             
        Parameters:
                Md(float32)         : disc mass
                Rd(float32)         : disc scale radius
                r(float32)   : 3D radius (see r_true in coordinate_transformation)
        Returns:
                v_c2(float32)   : exp disc circular velocity squared.

    '''
    cdef my_type v_c2 = 0.
    cdef my_type G = 4.299e-6

    if Md == 0 :
        return v_c2
    cdef my_type r_tmp = 0.
    r_tmp = r/(2.0*Rd)
    if r_tmp == 0.:
        r_tmp = 1e-8
        r = 1e-8
    if r_tmp <= 1000.:
        i0e = my_i0e(r_tmp)
        i1e = my_i1e(r_tmp)
        v_c2 =  ((G * Md * r) / (Rd*Rd)) * r_tmp * (i0e*my_k0e(r_tmp) - i1e*my_k1e(r_tmp))
    else:
        r2 = r*r 
        r3 = r2*r
        r5 = r3*r2 
        v_c2 = ((G * Md * r) / (Rd*Rd)) * r_tmp * (1.0/(4.0*r3) + 9.0/(32.0*r5))
    return v_c2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type rho_D(my_type Md,my_type Rd,my_type r,my_type c_incl) nogil:
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute exponential disc projected surface density.  
                                             
        Parameters:
                Md(float32)        : disc mass
                Rd(float32)        : disc scale radius
                incl(float32)      : inclination angle
                r(float32)  : 3D distance (see r_true in coordinate_transformation)

        Returns:
                rho(float32)   : exp disc projected density 
    '''
    cdef my_type rho = 0.
    cdef my_type s   = r/Rd
    if s > 80.:
        return 1e-10
    else:
        rho = (Md / (2.0 * M_PI * Rd*Rd) ) * exp(-r/Rd) 
    return(rho/c_incl)


#---------------------------------------------------------------------------
                    #TOTAL QUANTITIES
#---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type myv_abs2( my_type Mb,
                   my_type Rb,
                   my_type gamma,
                   my_type Md1,
                   my_type Rd1,
                   my_type Md2,
                   my_type Rd2,
                   my_type Mh,
                   my_type c,
                   my_type rho_crit,
                   my_type r,         #proj radius
                   int halo_type):# nogil:#phi computed in model function
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute line of sight velocity assuming (B+D1+D2+halo)
                                             
        Parameters:
                Mb(float32)         : bulge mass
                Rb(float32)         : bulge radius
                gamma(float32)      : bulge inner slope
                Md1(float32)        : inner disc mass
                Rd1(float32)        : inner disc scale radius
                Md2(float32)        : outer disc mass
                Rd2(float32)        : outer disc scale radius
                Mh(float32)         : halo mass
                c(float32)          : halo scale radius or concentration
                halo_type(int16)    : if 1 halo is NFW if 0 halo is herquinst 
        Returns:
                v_tot(float32) : sum of v_circ^2
    
    Note: for speed up 
    '''
    cdef my_type r_tmp = 0.
    cdef my_type v_tot = 0.
    if (Md1==0) or (Rd1==0):
        return v_tot  
    if halo_type == 1:
        # NFW
        v_tot = myv_D(Md1,Rd1,r) + myv_D(Md2,Rd2,r) + v_NFW(Mh,c,rho_crit,r) + v_dehnen(Mb,Rb,gamma,r) 
    elif halo_type == 0:
        #herquinst
        v_tot = myv_D(Md1,Rd1,r) + myv_D(Md2,Rd2,r) + v_H(Mh,c,r) + v_dehnen(Mb,Rb,gamma,r) 
    return v_tot 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type myv_abs2_BD( my_type Mb,
                   my_type Rb,
                   my_type gamma,
                   my_type Md1,
                   my_type Rd1,
                   my_type Mh,
                   my_type c,
                   my_type rho_crit,
                   my_type r,         #proj radius
                   int halo_type):# nogil:#phi computed in model function
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute line of sight velocity assuming (B+D1+D2+halo)
                                             
        Parameters:
                Mb(float32)         : bulge mass
                Rb(float32)         : bulge radius
                gamma(float32)      : bulge inner slope
                Md1(float32)        : inner disc mass
                Rd1(float32)        : inner disc scale radius
                Md2(float32)        : outer disc mass
                Rd2(float32)        : outer disc scale radius
                Mh(float32)         : halo mass
                c(float32)          : halo scale radius or concentration
                halo_type(int16)    : if 1 halo is NFW if 0 halo is herquinst 
        Returns:
                v_tot(float32) : sum of v_circ^2
    
    Note: for speed up 
    '''
    cdef my_type r_tmp = 0.
    cdef my_type v_tot = 0.
    if (Md1==0) or (Rd1==0):
        return v_tot  
    if halo_type == 1:
        # NFW
        v_tot = myv_D(Md1,Rd1,r) + v_NFW(Mh,c,rho_crit,r) + v_dehnen(Mb,Rb,gamma,r) 
    elif halo_type == 0:
        #herquinst
        v_tot = myv_D(Md1,Rd1,r) + v_H(Mh,c,r) + v_dehnen(Mb,Rb,gamma,r) 
    return v_tot 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type myv_abs2_DD(my_type Md1,
                   my_type Rd1,
                   my_type Md2,
                   my_type Rd2,
                   my_type Mh,
                   my_type c,
                   my_type rho_crit,
                   my_type r,         #proj radius
                   int halo_type):# nogil:#phi computed in model function
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute line of sight velocity assuming (B+D1+D2+halo)
                                             
        Parameters:
                Mb(float32)         : bulge mass
                Rb(float32)         : bulge radius
                gamma(float32)      : bulge inner slope
                Md1(float32)        : inner disc mass
                Rd1(float32)        : inner disc scale radius
                Md2(float32)        : outer disc mass
                Rd2(float32)        : outer disc scale radius
                Mh(float32)         : halo mass
                c(float32)          : halo scale radius or concentration
                halo_type(int16)    : if 1 halo is NFW if 0 halo is herquinst 
        Returns:
                v_tot(float32) : sum of v_circ^2
    
    Note: for speed up 
    '''
    cdef my_type r_tmp = 0.
    cdef my_type v_tot = 0.
    if (Md1==0) or (Rd1==0):
        return v_tot  
    if halo_type == 1:
        # NFW
        v_tot = myv_D(Md1,Rd1,r) + myv_D(Md2,Rd2,r) + v_NFW(Mh,c,rho_crit,r)  
    elif halo_type == 0:
        #herquinst
        v_tot = myv_D(Md1,Rd1,r) + myv_D(Md2,Rd2,r) + v_H(Mh,c,r)  
    return v_tot 




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type myv_tot_BH( my_type Mbh,
                   my_type Mb,
                   my_type Rb,
                   my_type Md1,
                   my_type Rd1,
                   my_type Md2,
                   my_type Rd2,
                   my_type Mh,
                   my_type Rh,
                   my_type s_incl,    #sin(incl)
                   my_type r,         #proj radius
                   my_type phi) nogil:#phi computed in model function
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute line of sight velocity assuming (BH+B[herquinst]+D1+D2+H).
                                             
        Parameters:
                v_exp1(float32)    : circ vel squared disc1 (see v_exp in v_D kernel)
                v_exp2(float32)    : circ vel squared disc2 (see v_exp in v_D kernel)
                v_bulge(float32)   : circ vel squared bulge (see v_H/v_B in v_H/v_Jaffe_H kernel)
                v_halo(float32)    : circ vel squared halo (see v_H/v_H in v_H/v_Jaffe_H kernel)
                incl(float32)      : inclination angle
                phi(float32)       : azimuthal angle in the disc plane (see phi in coordinate_transformation)
        Returns:
                v_tot(float32)       : LOS velocity
    '''                                         
    cdef my_type r_tmp = 0.
    cdef my_type v_tot = 0.
    if (Md1==0) or (Rd1==0):
        return v_tot       
    v_tot = s_incl * cos(phi) * sqrt(myv_D(Md1,Rd1,r) + myv_D(Md2,Rd2,r) + v_H(Mh,Rh,r) + v_H(Mb,Rb,r) + v_BH(Mbh,r) ) 
    return v_tot 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type myv_abs2_BH( my_type Mbh,
                   my_type Mb,
                   my_type Rb,
                   my_type Md1,
                   my_type Rd1,
                   my_type Md2,
                   my_type Rd2,
                   my_type Mh,
                   my_type Rh,
                   my_type s_incl,    #sin(incl)
                   my_type r,         #proj radius
                   my_type phi) :#nogil:#phi computed in model function
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute circular velocity squared (BH+B[herquinst]+D1+D2+H).
                                             
        Parameters:
                v_exp1(float32)    : circ vel squared disc1 (see v_exp in v_D kernel)
                v_exp2(float32)    : circ vel squared disc2 (see v_exp in v_D kernel)
                v_bulge(float32)   : circ vel squared bulge (see v_H/v_B in v_H/v_Jaffe_H kernel)
                v_halo(float32)    : circ vel squared halo (see v_H/v_H in v_H/v_Jaffe_H kernel)
                incl(float32)      : inclination angle
                phi(float32)       : azimuthal angle in the disc plane (see phi in coordinate_transformation)
        Returns:
                v_tot(float32)       : LOS velocity
    '''                                        
    cdef my_type r_tmp = 0.
    cdef my_type v_tot = 0.
    if (Md1==0) or (Rd1==0):
        return v_tot       
    v_tot =  v_BH(Mbh,r)+ myv_D(Md1,Rd1,r) + myv_D(Md2,Rd2,r) + v_H(Mb,Rb,r) + v_H(Mh,Rh,r) 
    return v_tot

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type myv_tot_BD_BH( my_type Mbh,
                   my_type Mb,
                   my_type Rb,
                   my_type Md1,
                   my_type Rd1,
                   my_type Mh,
                   my_type Rh,
                   my_type s_incl,    #sin(incl)
                   my_type r,         #proj radius
                   my_type phi) nogil:#phi computed in model function
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute line of sight velocity assuming (BH+B[herquinst]+D1+D2+H).
                                             
        Parameters:
                v_exp1(float32)    : circ vel squared disc1 (see v_exp in v_D kernel)
                v_exp2(float32)    : circ vel squared disc2 (see v_exp in v_D kernel)
                v_bulge(float32)   : circ vel squared bulge (see v_H/v_B in v_H/v_Jaffe_H kernel)
                v_halo(float32)    : circ vel squared halo (see v_H/v_H in v_H/v_Jaffe_H kernel)
                incl(float32)      : inclination angle
                phi(float32)       : azimuthal angle in the disc plane (see phi in coordinate_transformation)
        Returns:
                v_tot(float32)       : LOS velocity
    '''                                         
    cdef my_type r_tmp = 0.
    cdef my_type v_tot = 0.
    if (Md1==0) or (Rd1==0):
        return v_tot       
    v_tot = s_incl * cos(phi) * sqrt(myv_D(Md1,Rd1,r) + v_H(Mh,Rh,r) + v_H(Mb,Rb,r) + v_BH(Mbh,r) ) 
    return v_tot 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef my_type myv_abs2_BD_BH( my_type Mbh,
                   my_type Mb,
                   my_type Rb,
                   my_type Md1,
                   my_type Rd1,
                   my_type Mh,
                   my_type Rh,
                   my_type s_incl,    #sin(incl)
                   my_type r,         #proj radius
                   my_type phi) :#nogil:#phi computed in model function
    ''' 
    CPU function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute circular velocity squared (BH+B[herquinst]+D1+D2+H).
                                             
        Parameters:
                v_exp1(float32)    : circ vel squared disc1 (see v_exp in v_D kernel)
                v_bulge(float32)   : circ vel squared bulge (see v_H/v_B in v_H/v_Jaffe_H kernel)
                v_halo(float32)    : circ vel squared halo (see v_H/v_H in v_H/v_Jaffe_H kernel)
                incl(float32)      : inclination angle
                phi(float32)       : azimuthal angle in the disc plane (see phi in coordinate_transformation)
        Returns:
                v_tot(float32)       : LOS velocity
    '''                                        
    cdef my_type r_tmp = 0.
    cdef my_type v_tot = 0.
    if (Md1==0) or (Rd1==0):
        return v_tot       
    v_tot =  v_BH(Mbh,r)+ myv_D(Md1,Rd1,r) + v_H(Mb,Rb,r) + v_H(Mh,Rh,r) 
    return v_tot


##############################################################################################################################
##############################################################################################################################
#   MODEL AND LIKELIHOOD
##############################################################################################################################
##############################################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[my_type,ndim=2,mode='c'] model_BDD (my_type[::1] x,          
                                                           my_type[::1] y,
                                                           int N,
                                                           int J,
                                                           int K,          
                                                           my_type Mb,                  
                                                           my_type Rb,                  
                                                           my_type gamma,                  
                                                           my_type Md1,                  
                                                           my_type Rd1,                  
                                                           my_type Md2,                  
                                                           my_type Rd2,                  
                                                           my_type f,                 
                                                           my_type c,   # this is not in log10                  
                                                           my_type rho_crit,   
                                                           my_type xcm,                 
                                                           my_type ycm,                 
                                                           my_type theta,              
                                                           my_type incl,                
                                                           my_type L_to_M_b,        
                                                           my_type L_to_M_d1,       
                                                           my_type L_to_M_d2,        
                                                           my_type k_D1,       
                                                           my_type k_D2,
                                                           my_type[::1] s_grid,
                                                           my_type[::1] gamma_grid,
                                                           my_type[::1] rho_grid,
                                                           my_type[::1] sigma_grid,
                                                           my_type central_integration_radius,
                                                           my_type[::1] central_rho,
                                                           my_type[::1] central_sigma,
                                                           my_type[::1]psfkin_weight,
                                                           my_type[::1]psfpho_weight,
                                                           int [::1] indexing, 
                                                           my_type [::1] goodpix,
                                                           int central_index,
                                                           int halo_type):      # 1 is NFW 0 is hernquist  
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total) 
            - mean line of sight velocity (total)
            - mean line of sight velocity dispersion (total)
            - mean L/M ratio (total)

        Parameters:
                x(1D arr float32)                   : x positions
                y(1D arr float32)                   : y positions
                N(int16)                            : # pixels in x direction
                J(int16)                            : # pixels in y direction
                K(int16)                            : # pixels for convolution
                Mb(float32)                         : log10 bulge mass
                Rb(float32)                         : log10 bulge scale radius
                gamma(float32)                      : bulge inner slope
                Md1(float32)                        : log10 inner disc mass
                Rd1(float32)                        : log10 inner disc scale radius
                Md2(float32)                        : log10 outer disc mass
                Rd2(float32)                        : log10 outer disc scale radius
                f(float32)                          : log10 halo fraction
                c(float32)                          : concentration
                rho_crit(float32)                   : cosmological critical density
                xcm(float32)                        : x position of the center
                ycm(float32)                        : y position of the center
                theta(float32)                      : P.A.
                incl(float32)                       : inclination
                L_to_M_b(float32)                   : L/M bulge
                L_to_M_d1(float32)                  : L/M inner disc
                L_to_M_d2(float32)                  : L/M outer disc
                k_D1(float32)                       : kin parameter inner disc
                k_D2(float32)                       : kin parameter outer disc
                s_grid(1D arr float32)              : s grid for Dehnen interpolation
                gamma_grid(1D arr float32)          : gamma grid for Dehene interpolation
                rho_grid(1D arr float32)            : rho grid for Dehene interepolation
                sigma_grid(1D arr float32)          : sigma grid for deHene interpolation
                central_integration_radius(float32) : radius for the avg on the central pixel
                psfkin_weight(1D arr float32)       : kinematic weights of the psf 
                psfpho_weight(1D arr float32)       : photometric weights of the psf
                indexing (1D arr int16)             : index mapping for convolution 
                goodpix (1D arr int16)              : if goodpix[i] = 1 then i is used in the analysis
                central_index(int16)                : index referring to the central pixel,
                halo_type(int16)                    : if 0 halo is Herquinst, if 1 halo is NFW                
        Returns:
                tot(2D arr float32) : 
                                     - tot[:,0] is the total brightness
                                     - tot[:,1] is the avg velocity
                                     - tot[:,2] is the avg dispersion
                                     - tot[:,3] is the avg L/M ratio

    '''                                                                              
    cdef my_type x0 = 0.               #x-xcm  traslation
    cdef my_type y0 = 0.               #y-ycm  traslation
    cdef my_type xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef my_type yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef my_type yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef my_type r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef my_type r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef my_type c_theta = cos(theta)  #cos(theta) calculated once
    cdef my_type s_theta = sin(theta)  #sin(theta) calculated once
    cdef my_type c_incl = cos(incl)    #cos(incl) calculated once
    cdef my_type s_incl = sin(incl)    #cos(incl) calculated once
    cdef my_type phi = 0.
    cdef Py_ssize_t i,j,k,l

    
    #define HERQUINST STRUC 
    cdef HERQUINST dehnen  

    #   RHO DEFINITIONS
    cdef my_type rhoB_tmp = 0.
    cdef my_type rhoD1_tmp = 0.
    cdef my_type rhoD2_tmp = 0.
    cdef my_type B_B_tmp = 0.
    cdef my_type B_D1_tmp = 0.
    cdef my_type B_D2_tmp = 0.
    cdef my_type Btot_sum = 0.
    cdef my_type Btot_sum_kin = 0.
    cdef my_type rhotot_sum = 0.
    cdef np.ndarray[my_type,ndim=1,mode='c'] rhoB = np.zeros((N*J))
    cdef my_type[::1] rhoB_view = rhoB
    
    #V DEFINITIONS
    cdef my_type v_sum =0.
    cdef my_type v_2_sum =0.
    cdef my_type v_mean_tmp = 0.

    

    #PSF definitions
    cdef my_type kinpsf_tmp=0.
    cdef my_type phopsf_tmp=0.
    cdef my_type sum_psfpho = 0.
    cdef int ind = 0


    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((N*J,4))
    cdef my_type[:,::1] tot_view = tot

    Mb  = 10**Mb 
    Md1 = 10**Md1
    Md2 = 10**Md2
    Mh  = 10**f*(Mb+Md1+Md2)
    Rb  = 10**Rb 
    Rd1 = 10**Rd1
    Rd2 = 10**Rd2

    
    for i in range(N*J):
        if goodpix[i] == 1.:
            sum_psfpho = 0.
            Btot_sum = 0.
            Btot_sum_kin = 0.
            rhotot_sum = 0.
            v_sum = 0.

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                phopsf_tmp = psfpho_weight[l]

                ind = indexing[k]
                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)

                if ind != central_index:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,0,\
                                             central_rho,central_sigma)
                else:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,1,\
                                             central_rho,central_sigma)

                rhoD1_tmp = rho_D(Md1,Rd1,r_true,c_incl) 
                rhoD2_tmp = rho_D(Md2,Rd2,r_true,c_incl) 
                rhoB_tmp  = dehnen.rho
                B_D1_tmp = L_to_M_d1*rhoD1_tmp 
                B_D2_tmp = L_to_M_d2*rhoD2_tmp 
                B_B_tmp = L_to_M_b*rhoB_tmp
                #rhoB_view[i] = rhoB_tmp

                sum_psfpho += phopsf_tmp
                
                Btot_sum += phopsf_tmp*(B_B_tmp+B_D1_tmp+B_D2_tmp)
                Btot_sum_kin +=  kinpsf_tmp*(B_B_tmp+B_D1_tmp+B_D2_tmp)
                rhotot_sum += phopsf_tmp*(rhoD1_tmp+rhoD2_tmp+rhoB_tmp)


                # include which halo in total velocity
                v_tmp        = s_incl * cos(phi) * sqrt(myv_abs2(Mb,Rb,gamma,Md1,Rd1,Md2,Rd2,Mh,c,rho_crit,r_true,halo_type))
                
                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1+B_D2_tmp*v_tmp*k_D2)
                
                tot_view[i,2]   += kinpsf_tmp*B_B_tmp*dehnen.sigma
                
                
            tot_view[i,0] = Btot_sum/sum_psfpho
            tot_view[i,1] = v_sum/Btot_sum_kin
            tot_view[i,3] = Btot_sum/rhotot_sum

    for i in range(N*J):
        if goodpix[i] == 1.:
            Btot_sum_kin = 0.
            v_2_sum = 0.
            v_mean_tmp = tot[i,1]

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                ind = indexing[k]

                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)
                
                if ind != central_index:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,0,\
                                             central_rho,central_sigma)
                else:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,1,\
                                             central_rho,central_sigma)

                B_D1_tmp = L_to_M_d1*rho_D(Md1,Rd1,r_true,c_incl)  
                B_D2_tmp = L_to_M_d2*rho_D(Md2,Rd2,r_true,c_incl)  
                B_B_tmp = L_to_M_b* dehnen.rho

                Btot_sum_kin +=  kinpsf_tmp*(B_B_tmp+B_D1_tmp+B_D2_tmp)

                # include which halo in total velocity
                v_tmp2       = myv_abs2(Mb,Rb,gamma,Md1,Rd1,Md2,Rd2,Mh,c,rho_crit,r_true,halo_type)
                v_tmp        = s_incl * cos(phi) * sqrt(v_tmp2)
                
                tot_view[i,2]   += kinpsf_tmp*(B_D1_tmp*v_tmp2*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v_tmp2*(1.0-k_D2*k_D2)/3.0 +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp) + \
                                         B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp) )

            tot_view[i,2] = sqrt(tot_view[i,2]/Btot_sum_kin)

        else:
            pass

    return tot


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef my_type likelihood_BDD (my_type[::1] x,          #x position refined grid è 99x99x100
                              my_type[::1] y,
                              int N,
                              int J,
                              int K,          
                              my_type Mb,                  
                              my_type Rb,                  
                              my_type gamma,                  
                              my_type Md1,                  
                              my_type Rd1,                  
                              my_type Md2,                  
                              my_type Rd2,                  
                              my_type f,                 
                              my_type c,   # this is not in log10   
                              my_type rho_crit,               
                              my_type xcm,                 
                              my_type ycm,                 
                              my_type theta,              
                              my_type incl,                
                              my_type L_to_M_b,        
                              my_type L_to_M_d1,       
                              my_type L_to_M_d2,        
                              my_type k_D1,       
                              my_type k_D2,
                              my_type sys_rho,
                              my_type[::1] s_grid,
                              my_type[::1] gamma_grid,
                              my_type[::1] rho_grid,
                              my_type[::1] sigma_grid,
                              my_type central_integration_radius,
                              my_type[::1] central_rho,
                              my_type[::1] central_sigma,
                              my_type[::1]psfkin_weight,
                              my_type[::1]psfpho_weight,
                              int [::1] indexing, 
                              my_type [::1] goodpix,
                              int central_index,
                              int halo_type,
                              my_type[::1]rho_data,
                              my_type[::1]v_data,
                              my_type[::1]sigma_data,
                              my_type[::1]ML_data,
                              my_type[::1]rho_error,
                              my_type[::1]v_error,
                              my_type[::1]sigma_error,
                              my_type[::1]ML_error):       # 1 is NFW 0 is hernquist  
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total) 
            - mean line of sight velocity (total)
            - mean line of sight velocity dispersion (total)
            - mean L/M ratio (total)

        Parameters:
                x(1D arr float32)                   : x positions
                y(1D arr float32)                   : y positions
                N(int16)                            : # pixels in x direction
                J(int16)                            : # pixels in y direction
                K(int16)                            : # pixels for convolution
                Mb(float32)                         : log10 bulge mass
                Rb(float32)                         : log10 bulge scale radius
                gamma(float32)                      : bulge inner slope
                Md1(float32)                        : log10 inner disc mass
                Rd1(float32)                        : log10 inner disc scale radius
                Md2(float32)                        : log10 outer disc mass
                Rd2(float32)                        : log10 outer disc scale radius
                f(float32)                          : log10 halo fraction
                c(float32)                          : concentration
                rho_crit(float32)                   : cosmological critical density
                xcm(float32)                        : x position of the center
                ycm(float32)                        : y position of the center
                theta(float32)                      : P.A.
                incl(float32)                       : inclination
                L_to_M_b(float32)                   : L/M bulge
                L_to_M_d1(float32)                  : L/M inner disc
                L_to_M_d2(float32)                  : L/M outer disc
                k_D1(float32)                       : kin parameter inner disc
                k_D2(float32)                       : kin parameter outer disc
                s_grid(1D arr float32)              : s grid for Dehnen interpolation
                gamma_grid(1D arr float32)          : gamma grid for Dehene interpolation
                rho_grid(1D arr float32)            : rho grid for Dehene interepolation
                sigma_grid(1D arr float32)          : sigma grid for deHene interpolation
                central_integration_radius(float32) : radius for the avg on the central pixel
                psfkin_weight(1D arr float32)       : kinematic weights of the psf 
                psfpho_weight(1D arr float32)       : photometric weights of the psf
                indexing (1D arr int16)             : index mapping for convolution 
                goodpix (1D arr int16)              : if goodpix[i] = 1 then i is used in the analysis
                central_index(int16)                : index referring to the central pixel,
                halo_type(int16)                    : if 0 halo is Herquinst, if 1 halo is NFW                
                rho_data(1D arr float32)            : log10 brightness data
                v_data(1D arr float32)              : v_los data
                sigma_data(1D arr float32)          : velocity dispersion data
                ML_data(1D arr float32)             : mass to light ratio data
                rho_error(1D arr float32)           : log10 brightness data error
                v_error(1D arr float32)             : v_los data error
                sigma_error(1D arr float32)         : sigma_los data error
                ML_error(1D arr float32)            : mass to light ratio data errorr
        Returns:
                lk(float32) : log likelihood 
                                     
    ''' 
    
    cdef Py_ssize_t i,j,k,l

    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((N*J,4))
    cdef my_type[:,::1] tot_view = tot

    cdef my_type rho_lk   = 0.
    cdef my_type v_lk     = 0.
    cdef my_type sigma_lk = 0.
    cdef my_type ML_lk    = 0.
    cdef my_type rho_err_tmp   = 0.
    cdef my_type v_err_tmp     = 0.
    cdef my_type sigma_err_tmp = 0.
    cdef my_type ML_err_tmp    = 0.
    cdef my_type delta_rho  = 0.
    cdef my_type delta_v    = 0.
    cdef my_type delta_sigma= 0.
    cdef my_type delta_ML   = 0.
    
    cdef my_type lk   = 0.

    tot =  model_BDD (x,         
                      y,
                      N,
                      J,
                      K,          
                      Mb,                  
                      Rb,                  
                      gamma,                  
                      Md1,                  
                      Rd1,                  
                      Md2,                  
                      Rd2,                  
                      f,                 
                      c,  
                      rho_crit,
                      xcm,                 
                      ycm,                 
                      theta,              
                      incl,                
                      L_to_M_b,        
                      L_to_M_d1,       
                      L_to_M_d2,        
                      k_D1,       
                      k_D2,
                      s_grid,
                      gamma_grid,
                      rho_grid,
                      sigma_grid,
                      central_integration_radius,
                      central_rho,
                      central_sigma,
                      psfkin_weight,
                      psfpho_weight,
                      indexing, 
                      goodpix,
                      central_index,
                      halo_type)

    
    for i in range(N*J):
        if isnan(rho_data[i]) == True:
            rho_lk = 0.
        else:
            rho_err_tmp = rho_error[i]
            rho_err_tmp = rho_err_tmp*rho_err_tmp + (sys_rho/10**rho_data[i]/log(10))*(sys_rho/10**rho_data[i]/log(10))
            delta_rho   = rho_data[i]-log10(tot[i,0])
            rho_lk      = (delta_rho*delta_rho)/rho_err_tmp+log(2.0*M_PI*rho_err_tmp)
            #print(delta_rho,rho_lk)
        if isnan(v_data[i]) == True:
            v_lk  = 0.
        else:
            v_err_tmp = v_error[i]
            v_err_tmp = v_err_tmp*v_err_tmp
            delta_v   = v_data[i]-tot[i,1]
            v_lk      = delta_v*delta_v/v_err_tmp+log(2.0*M_PI*v_err_tmp)
            #print(delta_v,v_lk)
        if isnan(sigma_data[i]) == True:
            sigma_lk = 0.
        else:
            sigma_err_tmp = sigma_error[i]
            sigma_err_tmp = sigma_err_tmp*sigma_err_tmp
            delta_sigma   = sigma_data[i]-tot[i,2]
            sigma_lk = delta_sigma*delta_sigma/sigma_err_tmp+log(2.0*M_PI*sigma_err_tmp)
            #print(delta_sigma,sigma_lk)
        if isnan(ML_data[i]) == True:
            ML_lk = 0.
        else:
            ML_err_tmp = ML_error[i]
            ML_err_tmp = ML_err_tmp*ML_err_tmp
            delta_ML   = ML_data[i]-(1./tot[i,3])
            ML_lk = delta_ML*delta_ML/ML_err_tmp+log(2*M_PI*ML_err_tmp)
            #print(ML_data[i],tot[i,3],ML_lk)

        lk += -0.5*(rho_lk+v_lk+sigma_lk+ML_lk)
    
    return lk

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[my_type,ndim=2,mode='c'] model_BDD_all_outputs (my_type[::1] x,          
                                                           my_type[::1] y,
                                                           int N,
                                                           int J,
                                                           int K,          
                                                           my_type Mb,                  
                                                           my_type Rb,                  
                                                           my_type gamma,                  
                                                           my_type Md1,                  
                                                           my_type Rd1,                  
                                                           my_type Md2,                  
                                                           my_type Rd2,                  
                                                           my_type f,                 
                                                           my_type c,   # this is not in log10                  
                                                           my_type rho_crit,   
                                                           my_type xcm,                 
                                                           my_type ycm,                 
                                                           my_type theta,              
                                                           my_type incl,                
                                                           my_type L_to_M_b,        
                                                           my_type L_to_M_d1,       
                                                           my_type L_to_M_d2,        
                                                           my_type k_D1,       
                                                           my_type k_D2,
                                                           my_type[::1] s_grid,
                                                           my_type[::1] gamma_grid,
                                                           my_type[::1] rho_grid,
                                                           my_type[::1] sigma_grid,
                                                           my_type central_integration_radius,
                                                           my_type[::1] central_rho,
                                                           my_type[::1] central_sigma,
                                                           my_type[::1]psfkin_weight,
                                                           my_type[::1]psfpho_weight,
                                                           int [::1] indexing, 
                                                           my_type [::1] goodpix,
                                                           int central_index,
                                                           int halo_type):      # 1 is NFW 0 is hernquist  
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total,B,D1,D2) 
            - mean line of sight velocity (total,B,D1,D2)
            - mean line of sight velocity dispersion (total,B,D1,D2)
            - mean L/M ratio (total,B,D1,D2)

        Parameters:
                x(1D arr float32)                   : x positions
                y(1D arr float32)                   : y positions
                N(int16)                            : # pixels in x direction
                J(int16)                            : # pixels in y direction
                K(int16)                            : # pixels for convolution
                Mb(float32)                         : log10 bulge mass
                Rb(float32)                         : log10 bulge scale radius
                gamma(float32)                      : bulge inner slope
                Md1(float32)                        : log10 inner disc mass
                Rd1(float32)                        : log10 inner disc scale radius
                Md2(float32)                        : log10 outer disc mass
                Rd2(float32)                        : log10 outer disc scale radius
                f(float32)                          : log10 halo fraction
                c(float32)                          : concentration
                rho_crit(float32)                   : cosmological critical density
                xcm(float32)                        : x position of the center
                ycm(float32)                        : y position of the center
                theta(float32)                      : P.A.
                incl(float32)                       : inclination
                L_to_M_b(float32)                   : L/M bulge
                L_to_M_d1(float32)                  : L/M inner disc
                L_to_M_d2(float32)                  : L/M outer disc
                k_D1(float32)                       : kin parameter inner disc
                k_D2(float32)                       : kin parameter outer disc
                s_grid(1D arr float32)              : s grid for Dehnen interpolation
                gamma_grid(1D arr float32)          : gamma grid for Dehene interpolation
                rho_grid(1D arr float32)            : rho grid for Dehene interepolation
                sigma_grid(1D arr float32)          : sigma grid for deHene interpolation
                central_integration_radius(float32) : radius for the avg on the central pixel
                psfkin_weight(1D arr float32)       : kinematic weights of the psf 
                psfpho_weight(1D arr float32)       : photometric weights of the psf
                indexing (1D arr int16)             : index mapping for convolution 
                goodpix (1D arr int16)              : if goodpix[i] = 1 then i is used in the analysis
                central_index(int16)                : index referring to the central pixel,
                halo_type(int16)                    : if 0 halo is Herquinst, if 1 halo is NFW                
        Returns:
                tot(2D arr float32) : 
                                     - tot[:,0] is the total brightness
                                     - tot[:,1] is the bulge surface brightness
                                     - tot[:,2] is the disc1 surface brightness
                                     - tot[:,3] is disc2 surface brightness
                                     - tot[:,4] is the avg velocity
                                     - tot[:,5] is the bulge line of sight velocity
                                     - tot[:,6] is the disc1 line of sight velocity
                                     - tot[:,7] is the disc2 line of sight velocity
                                     - tot[:,8] is the avg dispersion
                                     - tot[:,9] is the bulge line of sight velocity dispersion
                                     - tot[:,10] is the disc1 line of sight velocity dispersion
                                     - tot[:,11] is the disc2 line of sight velocity dispersion
                                     - tot[:,12] is the avg L/M ratio
                                     - tot[:,13] is the bulge L/M ratio
                                     - tot[:,14] is the disc1 L/M ratio
                                     - tot[:,15] is the disc2 L/M ratio

    '''                                                                              
    cdef my_type x0 = 0.               #x-xcm  traslation
    cdef my_type y0 = 0.               #y-ycm  traslation
    cdef my_type xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef my_type yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef my_type yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef my_type r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef my_type r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef my_type c_theta = cos(theta)  #cos(theta) calculated once
    cdef my_type s_theta = sin(theta)  #sin(theta) calculated once
    cdef my_type c_incl = cos(incl)    #cos(incl) calculated once
    cdef my_type s_incl = sin(incl)    #cos(incl) calculated once
    cdef my_type phi = 0.
    cdef Py_ssize_t i,j,k,l

    
    #define HERQUINST STRUC 
    cdef HERQUINST dehnen  

    #   RHO DEFINITIONS
    cdef my_type rhoB_tmp = 0.
    cdef my_type rhoD1_tmp = 0.
    cdef my_type rhoD2_tmp = 0.
    cdef my_type B_B_tmp = 0.
    cdef my_type B_D1_tmp = 0.
    cdef my_type B_D2_tmp = 0.
    cdef my_type Btot_sum = 0.
    cdef my_type B_B_sum = 0.
    cdef my_type B_D1_sum = 0.
    cdef my_type B_D2_sum = 0.
    cdef my_type Btot_sum_kin = 0.
    cdef my_type rhotot_sum = 0.
    cdef np.ndarray[my_type,ndim=1,mode='c'] rhoB = np.zeros((N*J))
    cdef my_type[::1] rhoB_view = rhoB
    
    #V DEFINITIONS
    cdef my_type v_sum =0.
    cdef my_type v_2_sum =0.
    cdef my_type v_mean_tmp = 0.
    cdef my_type v_sum_D1 = 0.
    cdef my_type v_sum_D2 = 0.
    cdef my_type v_2_sum_B   = 0.
    cdef my_type v_2_sum_D1   = 0.
    cdef my_type v_2_sum_D2   = 0.

    #PSF definitions
    cdef my_type kinpsf_tmp=0.
    cdef my_type phopsf_tmp=0.
    cdef my_type sum_psfpho = 0.
    cdef int ind = 0


    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((N*J,16))
    cdef my_type[:,::1] tot_view = tot

    Mb  = 10**Mb 
    Md1 = 10**Md1
    Md2 = 10**Md2
    Mh  = 10**f*(Mb+Md1+Md2)
    Rb  = 10**Rb 
    Rd1 = 10**Rd1
    Rd2 = 10**Rd2

    for i in range(N*J):
        if goodpix[i] == 1.:
            sum_psfpho = 0.
            
            Btot_sum = 0.
            B_B_sum = 0.
            B_D1_sum = 0.
            B_D2_sum = 0.

            Btot_sum_kin = 0.
            rhotot_sum = 0.
            
            v_sum = 0.
            v_sum_D1 = 0.
            v_sum_D2 = 0.

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                phopsf_tmp = psfpho_weight[l]

                ind = indexing[k]
                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)

                if ind != central_index:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,0,\
                                             central_rho,central_sigma)
                else:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,1,\
                                             central_rho,central_sigma)

                rhoD1_tmp = rho_D(Md1,Rd1,r_true,c_incl) 
                rhoD2_tmp = rho_D(Md2,Rd2,r_true,c_incl) 
                rhoB_tmp  = dehnen.rho
                B_D1_tmp = L_to_M_d1*rhoD1_tmp 
                B_D2_tmp = L_to_M_d2*rhoD2_tmp 
                B_B_tmp = L_to_M_b*rhoB_tmp
                #rhoB_view[i] = rhoB_tmp

                sum_psfpho += phopsf_tmp
                
                Btot_sum += phopsf_tmp*(B_B_tmp+B_D1_tmp+B_D2_tmp)
                B_B_sum += phopsf_tmp*(B_B_tmp)
                B_D1_sum += phopsf_tmp*(B_D1_tmp)
                B_D2_sum += phopsf_tmp*(B_D2_tmp)

                Btot_sum_kin +=  kinpsf_tmp*(B_B_tmp+B_D1_tmp+B_D2_tmp)
                rhotot_sum += phopsf_tmp*(rhoD1_tmp+rhoD2_tmp+rhoB_tmp)


                # include which halo in total velocity
                v_tmp        = s_incl * cos(phi) * sqrt(myv_abs2(Mb,Rb,gamma,Md1,Rd1,Md2,Rd2,Mh,c,rho_crit,r_true,halo_type))

                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1+B_D2_tmp*v_tmp*k_D2)
                v_sum_D1  += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1)
                v_sum_D2  += kinpsf_tmp*(B_D2_tmp*v_tmp*k_D2)
                
                tot_view[i,8]   += kinpsf_tmp*B_B_tmp*dehnen.sigma
                tot_view[i,9]   += kinpsf_tmp*B_B_tmp*dehnen.sigma
                
                
            tot_view[i,0] = Btot_sum/sum_psfpho
            tot_view[i,1] = B_B_sum/sum_psfpho
            tot_view[i,2] = B_D1_sum/sum_psfpho
            tot_view[i,3] = B_D2_sum/sum_psfpho

            tot_view[i,4] = v_sum/Btot_sum_kin
            tot_view[i,5] = 0.
            tot_view[i,6] = v_sum_D1/Btot_sum_kin
            tot_view[i,7] = v_sum_D2/Btot_sum_kin
            
            
            tot_view[i,12] = Btot_sum/rhotot_sum
            tot_view[i,13] = Btot_sum/(B_B_sum/L_to_M_b)
            tot_view[i,14] = Btot_sum/(B_D1_sum/L_to_M_d1)
            tot_view[i,15] = Btot_sum/(B_D2_sum/L_to_M_d2)

    for i in range(N*J):
        if goodpix[i] == 1.:
            Btot_sum_kin = 0.
            v_2_sum = 0.
            v_2_sum_B   = 0.
            v_2_sum_D1   = 0.
            v_2_sum_D2   = 0.

            v_mean_tmp = tot[i,4]

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                ind = indexing[k]

                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)

                if ind != central_index:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,0,\
                                             central_rho,central_sigma)
                else:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,1,\
                                             central_rho,central_sigma)

                B_D1_tmp = L_to_M_d1*rho_D(Md1,Rd1,r_true,c_incl)  
                B_D2_tmp = L_to_M_d2*rho_D(Md2,Rd2,r_true,c_incl)  
                B_B_tmp = L_to_M_b* dehnen.rho

                Btot_sum_kin +=  kinpsf_tmp*(B_B_tmp+B_D1_tmp+B_D2_tmp)

                # include which halo in total velocity
                v_tmp2       = myv_abs2(Mb,Rb,gamma,Md1,Rd1,Md2,Rd2,Mh,c,rho_crit,r_true,halo_type)
                v_tmp        = s_incl * cos(phi) * sqrt(v_tmp2)
                
                tot_view[i,8]   += kinpsf_tmp*(B_D1_tmp*v_tmp2*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v_tmp2*(1.0-k_D2*k_D2)/3.0 +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp) + \
                                         B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp) )
                v_2_sum_D1   += kinpsf_tmp*(B_D1_tmp*v_tmp2*(1.0-k_D1*k_D1)/3.0  +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp))
                v_2_sum_D2   += kinpsf_tmp*(B_D2_tmp*v_tmp2*(1.0-k_D2*k_D2)/3.0 +\
                                         B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp) )

            tot_view[i,8] = sqrt(tot_view[i,8]/Btot_sum_kin)
            tot_view[i,9] = sqrt(tot_view[i,9]/Btot_sum_kin)
            tot_view[i,10] = sqrt(v_2_sum_D1/Btot_sum_kin)
            tot_view[i,11] = sqrt(v_2_sum_D2/Btot_sum_kin)

        else:
            pass

    return tot

##############################################################################################################################
##############################################################################################################################
#   MODEL AND LIKELIHOOD bulge+disc
##############################################################################################################################
##############################################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[my_type,ndim=2,mode='c'] model_BD (my_type[::1] x,          
                                                           my_type[::1] y,
                                                           int N,
                                                           int J,
                                                           int K,          
                                                           my_type Mb,                  
                                                           my_type Rb,                  
                                                           my_type gamma,                  
                                                           my_type Md1,                  
                                                           my_type Rd1,                  
                                                           my_type f,                 
                                                           my_type c,   # this is not in log10                  
                                                           my_type rho_crit,   
                                                           my_type xcm,                 
                                                           my_type ycm,                 
                                                           my_type theta,              
                                                           my_type incl,                
                                                           my_type L_to_M_b,        
                                                           my_type L_to_M_d1,       
                                                           my_type k_D1,       
                                                           my_type[::1] s_grid,
                                                           my_type[::1] gamma_grid,
                                                           my_type[::1] rho_grid,
                                                           my_type[::1] sigma_grid,
                                                           my_type central_integration_radius,
                                                           my_type[::1] central_rho,
                                                           my_type[::1] central_sigma,
                                                           my_type[::1]psfkin_weight,
                                                           my_type[::1]psfpho_weight,
                                                           int [::1] indexing, 
                                                           my_type [::1] goodpix,
                                                           int central_index,
                                                           int halo_type):      # 1 is NFW 0 is hernquist  
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total) 
            - mean line of sight velocity (total)
            - mean line of sight velocity dispersion (total)
            - mean L/M ratio (total)

        Parameters:
                x(1D arr float32)                   : x positions
                y(1D arr float32)                   : y positions
                N(int16)                            : # pixels in x direction
                J(int16)                            : # pixels in y direction
                K(int16)                            : # pixels for convolution
                Mb(float32)                         : log10 bulge mass
                Rb(float32)                         : log10 bulge scale radius
                gamma(float32)                      : bulge inner slope
                Md1(float32)                        : log10 inner disc mass
                Rd1(float32)                        : log10 inner disc scale radius
                f(float32)                          : log10 halo fraction
                c(float32)                          : concentration
                rho_crit(float32)                   : cosmological critical density
                xcm(float32)                        : x position of the center
                ycm(float32)                        : y position of the center
                theta(float32)                      : P.A.
                incl(float32)                       : inclination
                L_to_M_b(float32)                   : L/M bulge
                L_to_M_d1(float32)                  : L/M inner disc
                k_D1(float32)                       : kin parameter inner disc
                k_D2(float32)                       : kin parameter outer disc
                s_grid(1D arr float32)              : s grid for Dehnen interpolation
                gamma_grid(1D arr float32)          : gamma grid for Dehene interpolation
                rho_grid(1D arr float32)            : rho grid for Dehene interepolation
                sigma_grid(1D arr float32)          : sigma grid for deHene interpolation
                central_integration_radius(float32) : radius for the avg on the central pixel
                psfkin_weight(1D arr float32)       : kinematic weights of the psf 
                psfpho_weight(1D arr float32)       : photometric weights of the psf
                indexing (1D arr int16)             : index mapping for convolution 
                goodpix (1D arr int16)              : if goodpix[i] = 1 then i is used in the analysis
                central_index(int16)                : index referring to the central pixel,
                halo_type(int16)                    : if 0 halo is Herquinst, if 1 halo is NFW                
        Returns:
                tot(2D arr float32) : 
                                     - tot[:,0] is the total brightness
                                     - tot[:,1] is the avg velocity
                                     - tot[:,2] is the avg dispersion
                                     - tot[:,3] is the avg L/M ratio

    '''                                                                              
    cdef my_type x0 = 0.               #x-xcm  traslation
    cdef my_type y0 = 0.               #y-ycm  traslation
    cdef my_type xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef my_type yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef my_type yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef my_type r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef my_type r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef my_type c_theta = cos(theta)  #cos(theta) calculated once
    cdef my_type s_theta = sin(theta)  #sin(theta) calculated once
    cdef my_type c_incl = cos(incl)    #cos(incl) calculated once
    cdef my_type s_incl = sin(incl)    #cos(incl) calculated once
    cdef my_type phi = 0.
    cdef Py_ssize_t i,j,k,l

    
    #define HERQUINST STRUC 
    cdef HERQUINST dehnen  

    #   RHO DEFINITIONS
    cdef my_type rhoB_tmp = 0.
    cdef my_type rhoD1_tmp = 0.
    cdef my_type B_B_tmp = 0.
    cdef my_type B_D1_tmp = 0.
    cdef my_type Btot_sum = 0.
    cdef my_type Btot_sum_kin = 0.
    cdef my_type rhotot_sum = 0.
    cdef np.ndarray[my_type,ndim=1,mode='c'] rhoB = np.zeros((N*J))
    cdef my_type[::1] rhoB_view = rhoB
    
    #V DEFINITIONS
    cdef my_type v_sum =0.
    cdef my_type v_2_sum =0.
    cdef my_type v_mean_tmp = 0.

    

    #PSF definitions
    cdef my_type kinpsf_tmp=0.
    cdef my_type phopsf_tmp=0.
    cdef my_type sum_psfpho = 0.
    cdef int ind = 0


    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((N*J,4))
    cdef my_type[:,::1] tot_view = tot

    Mb  = 10**Mb 
    Md1 = 10**Md1
    Mh  = 10**f*(Mb+Md1)
    Rb  = 10**Rb 
    Rd1 = 10**Rd1

    
    for i in range(N*J):
        if goodpix[i] == 1.:
            sum_psfpho = 0.
            Btot_sum = 0.
            Btot_sum_kin = 0.
            rhotot_sum = 0.
            v_sum = 0.

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                phopsf_tmp = psfpho_weight[l]

                ind = indexing[k]
                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)

                if ind != central_index:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,0,\
                                             central_rho,central_sigma)
                else:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,1,\
                                             central_rho,central_sigma)

                rhoD1_tmp = rho_D(Md1,Rd1,r_true,c_incl) 
                rhoB_tmp  = dehnen.rho
                B_D1_tmp = L_to_M_d1*rhoD1_tmp 
                B_B_tmp = L_to_M_b*rhoB_tmp
                #rhoB_view[i] = rhoB_tmp

                sum_psfpho += phopsf_tmp
                
                Btot_sum += phopsf_tmp*(B_B_tmp+B_D1_tmp)
                Btot_sum_kin +=  kinpsf_tmp*(B_B_tmp+B_D1_tmp)
                rhotot_sum += phopsf_tmp*(rhoD1_tmp+rhoB_tmp)


                # include which halo in total velocity
                v_tmp        = s_incl * cos(phi) * sqrt(myv_abs2_BD(Mb,Rb,gamma,Md1,Rd1,Mh,c,rho_crit,r_true,halo_type))
                
                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1)
                
                tot_view[i,2]   += kinpsf_tmp*B_B_tmp*dehnen.sigma
                
                
            tot_view[i,0] = Btot_sum/sum_psfpho
            tot_view[i,1] = v_sum/Btot_sum_kin
            tot_view[i,3] = Btot_sum/rhotot_sum

    for i in range(N*J):
        if goodpix[i] == 1.:
            Btot_sum_kin = 0.
            v_2_sum = 0.
            v_mean_tmp = tot[i,1]

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                ind = indexing[k]

                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)
                if ind != central_index:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,0,\
                                             central_rho,central_sigma)
                else:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,1,\
                                             central_rho,central_sigma)

                B_D1_tmp = L_to_M_d1*rho_D(Md1,Rd1,r_true,c_incl)  
                B_B_tmp = L_to_M_b*dehnen.rho

                Btot_sum_kin +=  kinpsf_tmp*(B_B_tmp+B_D1_tmp)

                # include which halo in total velocity
                v_tmp2       = myv_abs2_BD(Mb,Rb,gamma,Md1,Rd1,Mh,c,rho_crit,r_true,halo_type)
                v_tmp        = s_incl * cos(phi) * sqrt(v_tmp2)
                
                tot_view[i,2]   += kinpsf_tmp*(B_D1_tmp*v_tmp2*(1.0-k_D1*k_D1)/3.0  +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp)) 

            tot_view[i,2] = sqrt(tot_view[i,2]/Btot_sum_kin)

        else:
            pass

    return tot


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef my_type likelihood_BD (my_type[::1] x,          #x position refined grid è 99x99x100
                              my_type[::1] y,
                              int N,
                              int J,
                              int K,          
                              my_type Mb,                  
                              my_type Rb,                  
                              my_type gamma,                  
                              my_type Md1,                  
                              my_type Rd1,                  
                              my_type f,                 
                              my_type c,   # this is not in log10   
                              my_type rho_crit,               
                              my_type xcm,                 
                              my_type ycm,                 
                              my_type theta,              
                              my_type incl,                
                              my_type L_to_M_b,        
                              my_type L_to_M_d1,       
                              my_type k_D1,       
                              my_type sys_rho,
                              my_type[::1] s_grid,
                              my_type[::1] gamma_grid,
                              my_type[::1] rho_grid,
                              my_type[::1] sigma_grid,
                              my_type central_integration_radius,
                              my_type[::1] central_rho,
                              my_type[::1] central_sigma,
                              my_type[::1]psfkin_weight,
                              my_type[::1]psfpho_weight,
                              int [::1] indexing, 
                              my_type [::1] goodpix,
                              int central_index,
                              int halo_type,
                              my_type[::1]rho_data,
                              my_type[::1]v_data,
                              my_type[::1]sigma_data,
                              my_type[::1]ML_data,
                              my_type[::1]rho_error,
                              my_type[::1]v_error,
                              my_type[::1]sigma_error,
                              my_type[::1]ML_error):       # 1 is NFW 0 is hernquist  
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total) 
            - mean line of sight velocity (total)
            - mean line of sight velocity dispersion (total)
            - mean L/M ratio (total)

        Parameters:
                x(1D arr float32)                   : x positions
                y(1D arr float32)                   : y positions
                N(int16)                            : # pixels in x direction
                J(int16)                            : # pixels in y direction
                K(int16)                            : # pixels for convolution
                Mb(float32)                         : log10 bulge mass
                Rb(float32)                         : log10 bulge scale radius
                gamma(float32)                      : bulge inner slope
                Md1(float32)                        : log10 inner disc mass
                Rd1(float32)                        : log10 inner disc scale radius
                f(float32)                          : log10 halo fraction
                c(float32)                          : concentration
                rho_crit(float32)                   : cosmological critical density
                xcm(float32)                        : x position of the center
                ycm(float32)                        : y position of the center
                theta(float32)                      : P.A.
                incl(float32)                       : inclination
                L_to_M_b(float32)                   : L/M bulge
                L_to_M_d1(float32)                  : L/M inner disc
                k_D1(float32)                       : kin parameter inner disc
                s_grid(1D arr float32)              : s grid for Dehnen interpolation
                gamma_grid(1D arr float32)          : gamma grid for Dehene interpolation
                rho_grid(1D arr float32)            : rho grid for Dehene interepolation
                sigma_grid(1D arr float32)          : sigma grid for deHene interpolation
                central_integration_radius(float32) : radius for the avg on the central pixel
                psfkin_weight(1D arr float32)       : kinematic weights of the psf 
                psfpho_weight(1D arr float32)       : photometric weights of the psf
                indexing (1D arr int16)             : index mapping for convolution 
                goodpix (1D arr int16)              : if goodpix[i] = 1 then i is used in the analysis
                central_index(int16)                : index referring to the central pixel,
                halo_type(int16)                    : if 0 halo is Herquinst, if 1 halo is NFW                
                rho_data(1D arr float32)            : log10 brightness data
                v_data(1D arr float32)              : v_los data
                sigma_data(1D arr float32)          : velocity dispersion data
                ML_data(1D arr float32)             : mass to light ratio data
                rho_error(1D arr float32)           : log10 brightness data error
                v_error(1D arr float32)             : v_los data error
                sigma_error(1D arr float32)         : sigma_los data error
                ML_error(1D arr float32)            : mass to light ratio data errorr
        Returns:
                lk(float32) : log likelihood 
                                     
    ''' 
    
    cdef Py_ssize_t i,j,k,l

    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((N*J,4))
    cdef my_type[:,::1] tot_view = tot

    cdef my_type rho_lk   = 0.
    cdef my_type v_lk     = 0.
    cdef my_type sigma_lk = 0.
    cdef my_type ML_lk    = 0.
    cdef my_type rho_err_tmp   = 0.
    cdef my_type v_err_tmp     = 0.
    cdef my_type sigma_err_tmp = 0.
    cdef my_type ML_err_tmp    = 0.
    cdef my_type delta_rho  = 0.
    cdef my_type delta_v    = 0.
    cdef my_type delta_sigma= 0.
    cdef my_type delta_ML   = 0.
    
    cdef my_type lk   = 0.

    tot =  model_BD (x,         
                      y,
                      N,
                      J,
                      K,          
                      Mb,                  
                      Rb,                  
                      gamma,                  
                      Md1,                  
                      Rd1,                  
                      f,                 
                      c,  
                      rho_crit,
                      xcm,                 
                      ycm,                 
                      theta,              
                      incl,                
                      L_to_M_b,        
                      L_to_M_d1,       
                      k_D1,       
                      s_grid,
                      gamma_grid,
                      rho_grid,
                      sigma_grid,
                      central_integration_radius,
                      central_rho,
                      central_sigma,
                      psfkin_weight,
                      psfpho_weight,
                      indexing, 
                      goodpix,
                      central_index,
                      halo_type)

    
    for i in range(N*J):
        if isnan(rho_data[i]) == True:
            rho_lk = 0.
        else:
            rho_err_tmp = rho_error[i]
            rho_err_tmp = rho_err_tmp*rho_err_tmp + (sys_rho/10**rho_data[i]/log(10))*(sys_rho/10**rho_data[i]/log(10))
            delta_rho   = rho_data[i]-log10(tot[i,0])
            rho_lk      = (delta_rho*delta_rho)/rho_err_tmp+log(2.0*M_PI*rho_err_tmp)
            #print(delta_rho,rho_lk)
        if isnan(v_data[i]) == True:
            v_lk  = 0.
        else:
            v_err_tmp = v_error[i]
            v_err_tmp = v_err_tmp*v_err_tmp
            delta_v   = v_data[i]-tot[i,1]
            v_lk      = delta_v*delta_v/v_err_tmp+log(2.0*M_PI*v_err_tmp)
            #print(delta_v,v_lk)
        if isnan(sigma_data[i]) == True:
            sigma_lk = 0.
        else:
            sigma_err_tmp = sigma_error[i]
            sigma_err_tmp = sigma_err_tmp*sigma_err_tmp
            delta_sigma   = sigma_data[i]-tot[i,2]
            sigma_lk = delta_sigma*delta_sigma/sigma_err_tmp+log(2.0*M_PI*sigma_err_tmp)
            #print(delta_sigma,sigma_lk)
        if isnan(ML_data[i]) == True:
            ML_lk = 0.
        else:
            ML_err_tmp = ML_error[i]
            ML_err_tmp = ML_err_tmp*ML_err_tmp
            delta_ML   = ML_data[i]-(1./tot[i,3])
            ML_lk = delta_ML*delta_ML/ML_err_tmp+log(2*M_PI*ML_err_tmp)
            #print(ML_data[i],tot[i,3],ML_lk)

        lk += -0.5*(rho_lk+v_lk+sigma_lk+ML_lk)
    
    return lk

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[my_type,ndim=2,mode='c'] model_BD_all_outputs (my_type[::1] x,          
                                                           my_type[::1] y,
                                                           int N,
                                                           int J,
                                                           int K,          
                                                           my_type Mb,                  
                                                           my_type Rb,                  
                                                           my_type gamma,                  
                                                           my_type Md1,                  
                                                           my_type Rd1,                  
                                                           my_type f,                 
                                                           my_type c,   # this is not in log10                  
                                                           my_type rho_crit,   
                                                           my_type xcm,                 
                                                           my_type ycm,                 
                                                           my_type theta,              
                                                           my_type incl,                
                                                           my_type L_to_M_b,        
                                                           my_type L_to_M_d1,       
                                                           my_type k_D1,       
                                                           my_type[::1] s_grid,
                                                           my_type[::1] gamma_grid,
                                                           my_type[::1] rho_grid,
                                                           my_type[::1] sigma_grid,
                                                           my_type central_integration_radius,
                                                           my_type[::1] central_rho,
                                                           my_type[::1] central_sigma,
                                                           my_type[::1]psfkin_weight,
                                                           my_type[::1]psfpho_weight,
                                                           int [::1] indexing, 
                                                           my_type [::1] goodpix,
                                                           int central_index,
                                                           int halo_type):      # 1 is NFW 0 is hernquist  
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total,B,D1,D2) 
            - mean line of sight velocity (total,B,D1,D2)
            - mean line of sight velocity dispersion (total,B,D1,D2)
            - mean L/M ratio (total,B,D1,D2)

        Parameters:
                x(1D arr float32)                   : x positions
                y(1D arr float32)                   : y positions
                N(int16)                            : # pixels in x direction
                J(int16)                            : # pixels in y direction
                K(int16)                            : # pixels for convolution
                Mb(float32)                         : log10 bulge mass
                Rb(float32)                         : log10 bulge scale radius
                gamma(float32)                      : bulge inner slope
                Md1(float32)                        : log10 inner disc mass
                Rd1(float32)                        : log10 inner disc scale radius
                f(float32)                          : log10 halo fraction
                c(float32)                          : concentration
                rho_crit(float32)                   : cosmological critical density
                xcm(float32)                        : x position of the center
                ycm(float32)                        : y position of the center
                theta(float32)                      : P.A.
                incl(float32)                       : inclination
                L_to_M_b(float32)                   : L/M bulge
                L_to_M_d1(float32)                  : L/M inner disc
                k_D1(float32)                       : kin parameter inner disc
                s_grid(1D arr float32)              : s grid for Dehnen interpolation
                gamma_grid(1D arr float32)          : gamma grid for Dehene interpolation
                rho_grid(1D arr float32)            : rho grid for Dehene interepolation
                sigma_grid(1D arr float32)          : sigma grid for deHene interpolation
                central_integration_radius(float32) : radius for the avg on the central pixel
                psfkin_weight(1D arr float32)       : kinematic weights of the psf 
                psfpho_weight(1D arr float32)       : photometric weights of the psf
                indexing (1D arr int16)             : index mapping for convolution 
                goodpix (1D arr int16)              : if goodpix[i] = 1 then i is used in the analysis
                central_index(int16)                : index referring to the central pixel,
                halo_type(int16)                    : if 0 halo is Herquinst, if 1 halo is NFW                
        Returns:
                tot(2D arr float32) : 
                                     - tot[:,0] is the total brightness
                                     - tot[:,1] is the bulge surface brightness
                                     - tot[:,2] is the disc1 surface brightness
                                     - tot[:,3] is disc2 surface brightness
                                     - tot[:,4] is the avg velocity
                                     - tot[:,5] is the bulge line of sight velocity
                                     - tot[:,6] is the disc1 line of sight velocity
                                     - tot[:,7] is the disc2 line of sight velocity
                                     - tot[:,8] is the avg dispersion
                                     - tot[:,9] is the bulge line of sight velocity dispersion
                                     - tot[:,10] is the disc1 line of sight velocity dispersion
                                     - tot[:,11] is the disc2 line of sight velocity dispersion
                                     - tot[:,12] is the avg L/M ratio
                                     - tot[:,13] is the bulge L/M ratio
                                     - tot[:,14] is the disc1 L/M ratio
                                     - tot[:,15] is the disc2 L/M ratio

    '''                                                                              
    cdef my_type x0 = 0.               #x-xcm  traslation
    cdef my_type y0 = 0.               #y-ycm  traslation
    cdef my_type xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef my_type yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef my_type yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef my_type r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef my_type r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef my_type c_theta = cos(theta)  #cos(theta) calculated once
    cdef my_type s_theta = sin(theta)  #sin(theta) calculated once
    cdef my_type c_incl = cos(incl)    #cos(incl) calculated once
    cdef my_type s_incl = sin(incl)    #cos(incl) calculated once
    cdef my_type phi = 0.
    cdef Py_ssize_t i,j,k,l

    
    #define HERQUINST STRUC 
    cdef HERQUINST dehnen  

    #   RHO DEFINITIONS
    cdef my_type rhoB_tmp = 0.
    cdef my_type rhoD1_tmp = 0.
    cdef my_type B_B_tmp = 0.
    cdef my_type B_D1_tmp = 0.
    cdef my_type Btot_sum = 0.
    cdef my_type B_B_sum = 0.
    cdef my_type B_D1_sum = 0.
    cdef my_type Btot_sum_kin = 0.
    cdef my_type rhotot_sum = 0.
    cdef np.ndarray[my_type,ndim=1,mode='c'] rhoB = np.zeros((N*J))
    cdef my_type[::1] rhoB_view = rhoB
    
    #V DEFINITIONS
    cdef my_type v_sum =0.
    cdef my_type v_2_sum =0.
    cdef my_type v_mean_tmp = 0.
    cdef my_type v_sum_D1 = 0.
    cdef my_type v_2_sum_B   = 0.
    cdef my_type v_2_sum_D1   = 0.

    #PSF definitions
    cdef my_type kinpsf_tmp=0.
    cdef my_type phopsf_tmp=0.
    cdef my_type sum_psfpho = 0.
    cdef int ind = 0


    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((N*J,12))
    cdef my_type[:,::1] tot_view = tot

    Mb  = 10**Mb 
    Md1 = 10**Md1
    Mh  = 10**f*(Mb+Md1)
    Rb  = 10**Rb 
    Rd1 = 10**Rd1

    for i in range(N*J):
        if goodpix[i] == 1.:
            sum_psfpho = 0.
            
            Btot_sum = 0.
            B_B_sum = 0.
            B_D1_sum = 0.

            Btot_sum_kin = 0.
            rhotot_sum = 0.
            
            v_sum = 0.
            v_sum_D1 = 0.

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                phopsf_tmp = psfpho_weight[l]

                ind = indexing[k]
                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)

                if ind != central_index:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,0,\
                                             central_rho,central_sigma)
                else:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,1,\
                                             central_rho,central_sigma)

                rhoD1_tmp = rho_D(Md1,Rd1,r_true,c_incl) 
                rhoB_tmp  = dehnen.rho
                B_D1_tmp = L_to_M_d1*rhoD1_tmp 
                B_B_tmp = L_to_M_b*rhoB_tmp
                #rhoB_view[i] = rhoB_tmp

                sum_psfpho += phopsf_tmp
                
                Btot_sum += phopsf_tmp*(B_B_tmp+B_D1_tmp)
                B_B_sum += phopsf_tmp*(B_B_tmp)
                B_D1_sum += phopsf_tmp*(B_D1_tmp)

                Btot_sum_kin +=  kinpsf_tmp*(B_B_tmp+B_D1_tmp)
                rhotot_sum += phopsf_tmp*(rhoD1_tmp+rhoB_tmp)


                # include which halo in total velocity
                v_tmp        = s_incl * cos(phi) * sqrt(myv_abs2_BD(Mb,Rb,gamma,Md1,Rd1,Mh,c,rho_crit,r_true,halo_type))

                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1)
                v_sum_D1  += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1)
                
                tot_view[i,6]   += kinpsf_tmp*B_B_tmp*dehnen.sigma
                tot_view[i,7]   += kinpsf_tmp*B_B_tmp*dehnen.sigma
                
                
            tot_view[i,0] = Btot_sum/sum_psfpho
            tot_view[i,1] = B_B_sum/sum_psfpho
            tot_view[i,2] = B_D1_sum/sum_psfpho

            tot_view[i,3] = v_sum/Btot_sum_kin
            tot_view[i,4] = 0.
            tot_view[i,5] = v_sum_D1/Btot_sum_kin
            
            
            tot_view[i,9] = Btot_sum/rhotot_sum
            tot_view[i,10] = Btot_sum/(B_B_sum/L_to_M_b)
            tot_view[i,11] = Btot_sum/(B_D1_sum/L_to_M_d1)

    for i in range(N*J):
        if goodpix[i] == 1.:
            Btot_sum_kin = 0.
            v_2_sum = 0.
            v_2_sum_B   = 0.
            v_2_sum_D1   = 0.

            v_mean_tmp = tot[i,3]

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                ind = indexing[k]

                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)

                if ind != central_index:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,0,\
                                             central_rho,central_sigma)
                else:
                    dehnen = dehnen_function(Mb,Rb,gamma,r_proj,\
                                             s_grid,gamma_grid,rho_grid,sigma_grid,\
                                             central_integration_radius,1,\
                                             central_rho,central_sigma)

                B_D1_tmp = L_to_M_d1*rho_D(Md1,Rd1,r_true,c_incl)  
                B_B_tmp = L_to_M_b*dehnen.rho 

                Btot_sum_kin +=  kinpsf_tmp*(B_B_tmp+B_D1_tmp)

                # include which halo in total velocity
                v_tmp2       = myv_abs2_BD(Mb,Rb,gamma,Md1,Rd1,Mh,c,rho_crit,r_true,halo_type)
                v_tmp        = s_incl * cos(phi) * sqrt(v_tmp2)
                
                tot_view[i,6]   += kinpsf_tmp*(B_D1_tmp*v_tmp2*(1.0-k_D1*k_D1)/3.0 + \
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp))
                v_2_sum_D1   += kinpsf_tmp*(B_D1_tmp*v_tmp2*(1.0-k_D1*k_D1)/3.0  +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp))

            tot_view[i,6] = sqrt(tot_view[i,6]/Btot_sum_kin)
            tot_view[i,7] = sqrt(tot_view[i,7]/Btot_sum_kin)
            tot_view[i,8] = sqrt(v_2_sum_D1/Btot_sum_kin)

        else:
            pass

    return tot


##############################################################################################################################
##############################################################################################################################
#   MODEL AND LIKELIHOOD D+D
##############################################################################################################################
##############################################################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[my_type,ndim=2,mode='c'] model_DD (my_type[::1] x,          
                                                           my_type[::1] y,
                                                           int N,
                                                           int J,
                                                           int K,          
                                                           my_type Md1,                  
                                                           my_type Rd1,                  
                                                           my_type Md2,                  
                                                           my_type Rd2,                  
                                                           my_type f,                 
                                                           my_type c,   # this is not in log10                  
                                                           my_type rho_crit,   
                                                           my_type xcm,                 
                                                           my_type ycm,                 
                                                           my_type theta,              
                                                           my_type incl,                
                                                           my_type L_to_M_d1,       
                                                           my_type L_to_M_d2,        
                                                           my_type k_D1,       
                                                           my_type k_D2,
                                                           my_type[::1]psfkin_weight,
                                                           my_type[::1]psfpho_weight,
                                                           int [::1] indexing, 
                                                           my_type [::1] goodpix,
                                                           int halo_type):      # 1 is NFW 0 is hernquist  
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total) 
            - mean line of sight velocity (total)
            - mean line of sight velocity dispersion (total)
            - mean L/M ratio (total)

        Parameters:
                x(1D arr float32)                   : x positions
                y(1D arr float32)                   : y positions
                N(int16)                            : # pixels in x direction
                J(int16)                            : # pixels in y direction
                K(int16)                            : # pixels for convolution
                Md1(float32)                        : log10 inner disc mass
                Rd1(float32)                        : log10 inner disc scale radius
                Md2(float32)                        : log10 outer disc mass
                Rd2(float32)                        : log10 outer disc scale radius
                f(float32)                          : log10 halo fraction
                c(float32)                          : concentration
                rho_crit(float32)                   : cosmological critical density
                xcm(float32)                        : x position of the center
                ycm(float32)                        : y position of the center
                theta(float32)                      : P.A.
                incl(float32)                       : inclination
                L_to_M_d1(float32)                  : L/M inner disc
                L_to_M_d2(float32)                  : L/M outer disc
                k_D1(float32)                       : kin parameter inner disc
                k_D2(float32)                       : kin parameter outer disc
                psfkin_weight(1D arr float32)       : kinematic weights of the psf 
                psfpho_weight(1D arr float32)       : photometric weights of the psf
                indexing (1D arr int16)             : index mapping for convolution 
                goodpix (1D arr int16)              : if goodpix[i] = 1 then i is used in the analysis
                central_index(int16)                : index referring to the central pixel,
                halo_type(int16)                    : if 0 halo is Herquinst, if 1 halo is NFW                
        Returns:
                tot(2D arr float32) : 
                                     - tot[:,0] is the total brightness
                                     - tot[:,1] is the avg velocity
                                     - tot[:,2] is the avg dispersion
                                     - tot[:,3] is the avg L/M ratio

    '''                                                                              
    cdef my_type x0 = 0.               #x-xcm  traslation
    cdef my_type y0 = 0.               #y-ycm  traslation
    cdef my_type xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef my_type yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef my_type yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef my_type r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef my_type r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef my_type c_theta = cos(theta)  #cos(theta) calculated once
    cdef my_type s_theta = sin(theta)  #sin(theta) calculated once
    cdef my_type c_incl = cos(incl)    #cos(incl) calculated once
    cdef my_type s_incl = sin(incl)    #cos(incl) calculated once
    cdef my_type phi = 0.
    cdef Py_ssize_t i,j,k,l

    

    #   RHO DEFINITIONS
    cdef my_type rhoD1_tmp = 0.
    cdef my_type rhoD2_tmp = 0.
    cdef my_type B_D1_tmp = 0.
    cdef my_type B_D2_tmp = 0.
    cdef my_type Btot_sum = 0.
    cdef my_type Btot_sum_kin = 0.
    cdef my_type rhotot_sum = 0.
    cdef np.ndarray[my_type,ndim=1,mode='c'] rhoB = np.zeros((N*J))
    cdef my_type[::1] rhoB_view = rhoB
    
    #V DEFINITIONS
    cdef my_type v_sum =0.
    cdef my_type v_2_sum =0.
    cdef my_type v_mean_tmp = 0.

    

    #PSF definitions
    cdef my_type kinpsf_tmp=0.
    cdef my_type phopsf_tmp=0.
    cdef my_type sum_psfpho = 0.
    cdef int ind = 0


    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((N*J,4))
    cdef my_type[:,::1] tot_view = tot

    Md1 = 10**Md1
    Md2 = 10**Md2
    Mh  = 10**f*(Md1+Md2)
    Rd1 = 10**Rd1
    Rd2 = 10**Rd2

    
    for i in range(N*J):
        if goodpix[i] == 1.:
            sum_psfpho = 0.
            Btot_sum = 0.
            Btot_sum_kin = 0.
            rhotot_sum = 0.
            v_sum = 0.

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                phopsf_tmp = psfpho_weight[l]

                ind = indexing[k]
                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)


                rhoD1_tmp = rho_D(Md1,Rd1,r_true,c_incl) 
                rhoD2_tmp = rho_D(Md2,Rd2,r_true,c_incl) 
                B_D1_tmp = L_to_M_d1*rhoD1_tmp 
                B_D2_tmp = L_to_M_d2*rhoD2_tmp 

                sum_psfpho += phopsf_tmp
                
                Btot_sum += phopsf_tmp*(B_D1_tmp+B_D2_tmp)
                Btot_sum_kin +=  kinpsf_tmp*(B_D1_tmp+B_D2_tmp)
                rhotot_sum += phopsf_tmp*(rhoD1_tmp+rhoD2_tmp)


                # include which halo in total velocity
                v_tmp        = s_incl * cos(phi) * sqrt(myv_abs2_DD(Md1,Rd1,Md2,Rd2,Mh,c,rho_crit,r_true,halo_type))
                
                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1+B_D2_tmp*v_tmp*k_D2)
                
                
                
            tot_view[i,0] = Btot_sum/sum_psfpho
            tot_view[i,1] = v_sum/Btot_sum
            tot_view[i,3] = Btot_sum/rhotot_sum

    for i in range(N*J):
        if goodpix[i] == 1.:
            Btot_sum_kin = 0.
            v_2_sum = 0.
            v_mean_tmp = tot[i,1]

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                ind = indexing[k]

                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)


                B_D1_tmp = L_to_M_d1*rho_D(Md1,Rd1,r_true,c_incl)  
                B_D2_tmp = L_to_M_d2*rho_D(Md2,Rd2,r_true,c_incl)  

                Btot_sum_kin +=  kinpsf_tmp*(B_D1_tmp+B_D2_tmp)

                # include which halo in total velocity
                v_tmp2       = myv_abs2_DD(Md1,Rd1,Md2,Rd2,Mh,c,rho_crit,r_true,halo_type)
                v_tmp        = s_incl * cos(phi) * sqrt(v_tmp2)
                
                tot_view[i,2]   += kinpsf_tmp*(B_D1_tmp*v_tmp2*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v_tmp2*(1.0-k_D2*k_D2)/3.0 +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp) + \
                                         B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp) )

            tot_view[i,2] = sqrt(tot_view[i,2]/Btot_sum_kin)

        else:
            pass

    return tot


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef my_type likelihood_DD (my_type[::1] x,          #x position refined grid è 99x99x100
                              my_type[::1] y,
                              int N,
                              int J,
                              int K,          
                              my_type Md1,                  
                              my_type Rd1,                  
                              my_type Md2,                  
                              my_type Rd2,                  
                              my_type f,                 
                              my_type c,   # this is not in log10   
                              my_type rho_crit,               
                              my_type xcm,                 
                              my_type ycm,                 
                              my_type theta,              
                              my_type incl,                
                              my_type L_to_M_d1,       
                              my_type L_to_M_d2,        
                              my_type k_D1,       
                              my_type k_D2,
                              my_type sys_rho,
                              my_type[::1]psfkin_weight,
                              my_type[::1]psfpho_weight,
                              int [::1] indexing, 
                              my_type [::1] goodpix,
                              int halo_type,
                              my_type[::1]rho_data,
                              my_type[::1]v_data,
                              my_type[::1]sigma_data,
                              my_type[::1]ML_data,
                              my_type[::1]rho_error,
                              my_type[::1]v_error,
                              my_type[::1]sigma_error,
                              my_type[::1]ML_error):       # 1 is NFW 0 is hernquist  
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total) 
            - mean line of sight velocity (total)
            - mean line of sight velocity dispersion (total)
            - mean L/M ratio (total)

        Parameters:
                x(1D arr float32)                   : x positions
                y(1D arr float32)                   : y positions
                N(int16)                            : # pixels in x direction
                J(int16)                            : # pixels in y direction
                K(int16)                            : # pixels for convolution
                Md1(float32)                        : log10 inner disc mass
                Rd1(float32)                        : log10 inner disc scale radius
                Md2(float32)                        : log10 outer disc mass
                Rd2(float32)                        : log10 outer disc scale radius
                f(float32)                          : log10 halo fraction
                c(float32)                          : concentration
                rho_crit(float32)                   : cosmological critical density
                xcm(float32)                        : x position of the center
                ycm(float32)                        : y position of the center
                theta(float32)                      : P.A.
                incl(float32)                       : inclination
                L_to_M_d1(float32)                  : L/M inner disc
                L_to_M_d2(float32)                  : L/M outer disc
                k_D1(float32)                       : kin parameter inner disc
                k_D2(float32)                       : kin parameter outer disc
                psfkin_weight(1D arr float32)       : kinematic weights of the psf 
                psfpho_weight(1D arr float32)       : photometric weights of the psf
                indexing (1D arr int16)             : index mapping for convolution 
                goodpix (1D arr int16)              : if goodpix[i] = 1 then i is used in the analysis
                halo_type(int16)                    : if 0 halo is Herquinst, if 1 halo is NFW                
                rho_data(1D arr float32)            : log10 brightness data
                v_data(1D arr float32)              : v_los data
                sigma_data(1D arr float32)          : velocity dispersion data
                ML_data(1D arr float32)             : mass to light ratio data
                rho_error(1D arr float32)           : log10 brightness data error
                v_error(1D arr float32)             : v_los data error
                sigma_error(1D arr float32)         : sigma_los data error
                ML_error(1D arr float32)            : mass to light ratio data errorr
        Returns:
                lk(float32) : log likelihood 
                                     
    ''' 
    
    cdef Py_ssize_t i,j,k,l

    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((N*J,4))
    cdef my_type[:,::1] tot_view = tot

    cdef my_type rho_lk   = 0.
    cdef my_type v_lk     = 0.
    cdef my_type sigma_lk = 0.
    cdef my_type ML_lk    = 0.
    cdef my_type rho_err_tmp   = 0.
    cdef my_type v_err_tmp     = 0.
    cdef my_type sigma_err_tmp = 0.
    cdef my_type ML_err_tmp    = 0.
    cdef my_type delta_rho  = 0.
    cdef my_type delta_v    = 0.
    cdef my_type delta_sigma= 0.
    cdef my_type delta_ML   = 0.
    
    cdef my_type lk   = 0.

    tot =  model_DD (x,         
                      y,
                      N,
                      J,
                      K,          
                      Md1,                  
                      Rd1,                  
                      Md2,                  
                      Rd2,                  
                      f,                 
                      c,  
                      rho_crit,
                      xcm,                 
                      ycm,                 
                      theta,              
                      incl,                
                      L_to_M_d1,       
                      L_to_M_d2,        
                      k_D1,       
                      k_D2,
                      psfkin_weight,
                      psfpho_weight,
                      indexing, 
                      goodpix,
                      halo_type)

    
    for i in range(N*J):
        if isnan(rho_data[i]) == True:
            rho_lk = 0.
        else:
            rho_err_tmp = rho_error[i]
            rho_err_tmp = rho_err_tmp*rho_err_tmp + (sys_rho/10**rho_data[i]/log(10))*(sys_rho/10**rho_data[i]/log(10))
            delta_rho   = rho_data[i]-log10(tot[i,0])
            rho_lk      = (delta_rho*delta_rho)/rho_err_tmp+log(2.0*M_PI*rho_err_tmp)
            #print(delta_rho,rho_lk)
        if isnan(v_data[i]) == True:
            v_lk  = 0.
        else:
            v_err_tmp = v_error[i]
            v_err_tmp = v_err_tmp*v_err_tmp
            delta_v   = v_data[i]-tot[i,1]
            v_lk      = delta_v*delta_v/v_err_tmp+log(2.0*M_PI*v_err_tmp)
            #print(delta_v,v_lk)
        if isnan(sigma_data[i]) == True:
            sigma_lk = 0.
        else:
            sigma_err_tmp = sigma_error[i]
            sigma_err_tmp = sigma_err_tmp*sigma_err_tmp
            delta_sigma   = sigma_data[i]-tot[i,2]
            sigma_lk = delta_sigma*delta_sigma/sigma_err_tmp+log(2.0*M_PI*sigma_err_tmp)
            #print(delta_sigma,sigma_lk)
        if isnan(ML_data[i]) == True:
            ML_lk = 0.
        else:
            ML_err_tmp = ML_error[i]
            ML_err_tmp = ML_err_tmp*ML_err_tmp
            delta_ML   = ML_data[i]-(1./tot[i,3])
            ML_lk = delta_ML*delta_ML/ML_err_tmp+log(2*M_PI*ML_err_tmp)
            #print(ML_data[i],tot[i,3],ML_lk)

        lk += -0.5*(rho_lk+v_lk+sigma_lk+ML_lk)
    
    return lk

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[my_type,ndim=2,mode='c'] model_DD_all_outputs (my_type[::1] x,          
                                                           my_type[::1] y,
                                                           int N,
                                                           int J,
                                                           int K,          
                                                           my_type Md1,                  
                                                           my_type Rd1,                  
                                                           my_type Md2,                  
                                                           my_type Rd2,                  
                                                           my_type f,                 
                                                           my_type c,   # this is not in log10                  
                                                           my_type rho_crit,   
                                                           my_type xcm,                 
                                                           my_type ycm,                 
                                                           my_type theta,              
                                                           my_type incl,                
                                                           my_type L_to_M_d1,       
                                                           my_type L_to_M_d2,        
                                                           my_type k_D1,       
                                                           my_type k_D2,
                                                           my_type[::1]psfkin_weight,
                                                           my_type[::1]psfpho_weight,
                                                           int [::1] indexing, 
                                                           my_type [::1] goodpix,
                                                           int halo_type):      # 1 is NFW 0 is hernquist  
    ''' 
    CPU FUNCTION
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Compute:
            - surface brightness (total,B,D1,D2) 
            - mean line of sight velocity (total,B,D1,D2)
            - mean line of sight velocity dispersion (total,B,D1,D2)
            - mean L/M ratio (total,B,D1,D2)

        Parameters:
                x(1D arr float32)                   : x positions
                y(1D arr float32)                   : y positions
                N(int16)                            : # pixels in x direction
                J(int16)                            : # pixels in y direction
                K(int16)                            : # pixels for convolution
                Mb(float32)                         : log10 bulge mass
                Rb(float32)                         : log10 bulge scale radius
                gamma(float32)                      : bulge inner slope
                Md1(float32)                        : log10 inner disc mass
                Rd1(float32)                        : log10 inner disc scale radius
                Md2(float32)                        : log10 outer disc mass
                Rd2(float32)                        : log10 outer disc scale radius
                f(float32)                          : log10 halo fraction
                c(float32)                          : concentration
                rho_crit(float32)                   : cosmological critical density
                xcm(float32)                        : x position of the center
                ycm(float32)                        : y position of the center
                theta(float32)                      : P.A.
                incl(float32)                       : inclination
                L_to_M_b(float32)                   : L/M bulge
                L_to_M_d1(float32)                  : L/M inner disc
                L_to_M_d2(float32)                  : L/M outer disc
                k_D1(float32)                       : kin parameter inner disc
                k_D2(float32)                       : kin parameter outer disc
                s_grid(1D arr float32)              : s grid for Dehnen interpolation
                gamma_grid(1D arr float32)          : gamma grid for Dehene interpolation
                rho_grid(1D arr float32)            : rho grid for Dehene interepolation
                sigma_grid(1D arr float32)          : sigma grid for deHene interpolation
                central_integration_radius(float32) : radius for the avg on the central pixel
                psfkin_weight(1D arr float32)       : kinematic weights of the psf 
                psfpho_weight(1D arr float32)       : photometric weights of the psf
                indexing (1D arr int16)             : index mapping for convolution 
                goodpix (1D arr int16)              : if goodpix[i] = 1 then i is used in the analysis
                central_index(int16)                : index referring to the central pixel,
                halo_type(int16)                    : if 0 halo is Herquinst, if 1 halo is NFW                
        Returns:
                tot(2D arr float32) : 
                                     - tot[:,0] is the total brightness
                                     - tot[:,1] is the bulge surface brightness
                                     - tot[:,2] is the disc1 surface brightness
                                     - tot[:,3] is disc2 surface brightness
                                     - tot[:,4] is the avg velocity
                                     - tot[:,5] is the bulge line of sight velocity
                                     - tot[:,6] is the disc1 line of sight velocity
                                     - tot[:,7] is the disc2 line of sight velocity
                                     - tot[:,8] is the avg dispersion
                                     - tot[:,9] is the bulge line of sight velocity dispersion
                                     - tot[:,10] is the disc1 line of sight velocity dispersion
                                     - tot[:,11] is the disc2 line of sight velocity dispersion
                                     - tot[:,12] is the avg L/M ratio
                                     - tot[:,13] is the bulge L/M ratio
                                     - tot[:,14] is the disc1 L/M ratio
                                     - tot[:,15] is the disc2 L/M ratio

    '''                                                                              
    cdef my_type x0 = 0.               #x-xcm  traslation
    cdef my_type y0 = 0.               #y-ycm  traslation
    cdef my_type xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef my_type yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef my_type yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef my_type r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef my_type r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef my_type c_theta = cos(theta)  #cos(theta) calculated once
    cdef my_type s_theta = sin(theta)  #sin(theta) calculated once
    cdef my_type c_incl = cos(incl)    #cos(incl) calculated once
    cdef my_type s_incl = sin(incl)    #cos(incl) calculated once
    cdef my_type phi = 0.
    cdef Py_ssize_t i,j,k,l

    

    #   RHO DEFINITIONS
    cdef my_type rhoD1_tmp = 0.
    cdef my_type rhoD2_tmp = 0.
    cdef my_type B_D1_tmp = 0.
    cdef my_type B_D2_tmp = 0.
    cdef my_type Btot_sum = 0.
    cdef my_type B_D1_sum = 0.
    cdef my_type B_D2_sum = 0.
    cdef my_type Btot_sum_kin = 0.
    cdef my_type rhotot_sum = 0.
    
    #V DEFINITIONS
    cdef my_type v_sum =0.
    cdef my_type v_2_sum =0.
    cdef my_type v_mean_tmp = 0.
    cdef my_type v_sum_D1 = 0.
    cdef my_type v_sum_D2 = 0.
    cdef my_type v_2_sum_D1   = 0.
    cdef my_type v_2_sum_D2   = 0.

    #PSF definitions
    cdef my_type kinpsf_tmp=0.
    cdef my_type phopsf_tmp=0.
    cdef my_type sum_psfpho = 0.
    cdef int ind = 0


    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((N*J,12))
    cdef my_type[:,::1] tot_view = tot

    Md1 = 10**Md1
    Md2 = 10**Md2
    Mh  = 10**f*(Md1+Md2)
    Rd1 = 10**Rd1
    Rd2 = 10**Rd2

    for i in range(N*J):
        if goodpix[i] == 1.:
            sum_psfpho = 0.
            
            Btot_sum = 0.
            B_D1_sum = 0.
            B_D2_sum = 0.

            Btot_sum_kin = 0.
            rhotot_sum = 0.
            
            v_sum = 0.
            v_sum_D1 = 0.
            v_sum_D2 = 0.

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                phopsf_tmp = psfpho_weight[l]

                ind = indexing[k]
                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)


                rhoD1_tmp = rho_D(Md1,Rd1,r_true,c_incl) 
                rhoD2_tmp = rho_D(Md2,Rd2,r_true,c_incl) 
                B_D1_tmp = L_to_M_d1*rhoD1_tmp 
                B_D2_tmp = L_to_M_d2*rhoD2_tmp 

                sum_psfpho += phopsf_tmp
                
                Btot_sum += phopsf_tmp*(B_D1_tmp+B_D2_tmp)
                B_D1_sum += phopsf_tmp*(B_D1_tmp)
                B_D2_sum += phopsf_tmp*(B_D2_tmp)

                Btot_sum_kin +=  kinpsf_tmp*(B_D1_tmp+B_D2_tmp)
                rhotot_sum += phopsf_tmp*(rhoD1_tmp+rhoD2_tmp)


                # include which halo in total velocity
                v_tmp        = s_incl * cos(phi) * sqrt(myv_abs2_DD(Md1,Rd1,Md2,Rd2,Mh,c,rho_crit,r_true,halo_type))

                v_sum     += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1+B_D2_tmp*v_tmp*k_D2)
                v_sum_D1  += kinpsf_tmp*(B_D1_tmp*v_tmp*k_D1)
                v_sum_D2  += kinpsf_tmp*(B_D2_tmp*v_tmp*k_D2)
                
                
            tot_view[i,0] = Btot_sum/sum_psfpho
            tot_view[i,1] = B_D1_sum/sum_psfpho
            tot_view[i,2] = B_D2_sum/sum_psfpho

            tot_view[i,3] = v_sum/Btot_sum_kin
            tot_view[i,4] = v_sum_D1/Btot_sum_kin
            tot_view[i,5] = v_sum_D2/Btot_sum_kin
            
            
            tot_view[i,9] = Btot_sum/rhotot_sum
            tot_view[i,10] = Btot_sum/(B_D1_sum/L_to_M_d1)
            tot_view[i,11] = Btot_sum/(B_D2_sum/L_to_M_d2)

    for i in range(N*J):
        if goodpix[i] == 1.:
            Btot_sum_kin = 0.
            v_2_sum = 0.
            v_2_sum_D1   = 0.
            v_2_sum_D2   = 0.

            v_mean_tmp = tot[i,3]

            for k in range(i*K,(i+1)*K):
                l = int(k-i*K)

                kinpsf_tmp = psfkin_weight[l]
                ind = indexing[k]

                x0 = x[ind]-xcm
                y0 = y[ind]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)


                B_D1_tmp = L_to_M_d1*rho_D(Md1,Rd1,r_true,c_incl)  
                B_D2_tmp = L_to_M_d2*rho_D(Md2,Rd2,r_true,c_incl)  

                Btot_sum_kin +=  kinpsf_tmp*(B_D1_tmp+B_D2_tmp)

                # include which halo in total velocity
                v_tmp2       = myv_abs2_DD(Md1,Rd1,Md2,Rd2,Mh,c,rho_crit,r_true,halo_type)
                v_tmp        = s_incl * cos(phi) * sqrt(v_tmp2)
                
                tot_view[i,6]   += kinpsf_tmp*(B_D1_tmp*v_tmp2*(1.0-k_D1*k_D1)/3.0 + B_D2_tmp*v_tmp2*(1.0-k_D2*k_D2)/3.0 +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp) + \
                                         B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp) )
                v_2_sum_D1   += kinpsf_tmp*(B_D1_tmp*v_tmp2*(1.0-k_D1*k_D1)/3.0  +\
                                         B_D1_tmp*(v_tmp*k_D1-v_mean_tmp)*(v_tmp*k_D1-v_mean_tmp))
                v_2_sum_D2   += kinpsf_tmp*(B_D2_tmp*v_tmp2*(1.0-k_D2*k_D2)/3.0 +\
                                         B_D2_tmp*(v_tmp*k_D2-v_mean_tmp)*(v_tmp*k_D2-v_mean_tmp) )

            tot_view[i,6] = sqrt(tot_view[i,6]/Btot_sum_kin)
            tot_view[i,7] = sqrt(v_2_sum_D1/Btot_sum_kin)
            tot_view[i,8] = sqrt(v_2_sum_D2/Btot_sum_kin)

        else:
            pass

    return tot





##############################################################################################################################
##############################################################################################################################
#   HERQUINST+BH: MODEL AND LIKELIHOOD
##############################################################################################################################
##############################################################################################################################
#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
#@cython.cdivision(True)
#cpdef np.ndarray[my_type,ndim=2,mode='c'] model_hernquist_BH (my_type[::1] x,          #x position refined grid è 99x99x100
#                                                           my_type[::1] y,          #y position refined grid
#                                                           my_type Mbh,
#                                                           my_type Mb,                  #bulge mass log
#                                                           my_type Rb,                  #bulge radius
#                                                           my_type Md1,                  #disc mass log
#                                                           my_type Rd1,                  #disc radius
#                                                           my_type Md2,                  #disc mass log
#                                                           my_type Rd2,                  #disc radius
#                                                           my_type Mh,                  #halo mass log
#                                                           my_type Rh,                  #halo radius
#                                                           my_type xcm,                 #center x position
#                                                           my_type ycm,                 #center y position
#                                                           my_type theta,               #P.A. angle (rotation in sky-plane)
#                                                           my_type incl,                #inclination angle
#                                                           my_type L_to_M_b,        #const M/L bulge
#                                                           my_type L_to_M_d1,        #const M/L bulge
#                                                           my_type L_to_M_d2,        #const M/L bulge
#                                                           my_type k_D1,        #const M/L bulge
#                                                           my_type k_D2,        #const M/L bulge
#                                                           my_type[::1]psf_weight):  #PSF_weights 
#                                                                                 # if output != 0 Jaffe
# 
#    cdef my_type sigmaS2 = 0.0    
#    cdef my_type x0 = 0.               #x-xcm  traslation
#    cdef my_type y0 = 0.               #y-ycm  traslation
#    cdef my_type xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
#    cdef my_type yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
#    cdef my_type yd = 0.               #yd = yr/np.cos(incl)                 de-projection
#    cdef my_type r_true = 0.           #sqrt(xr**2+yd**2) real radius
#    cdef my_type r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
#    cdef my_type c_theta = cos(theta)  #cos(theta) calculated once
#    cdef my_type s_theta = sin(theta)  #sin(theta) calculated once
#    cdef my_type c_incl = cos(incl)    #cos(incl) calculated once
#    cdef my_type s_incl = sin(incl)    #cos(incl) calculated once
#    cdef my_type phi = 0.
#    cdef Py_ssize_t i,j,k,l
#    cdef Py_ssize_t NJK = x.shape[0]
#    cdef Py_ssize_t K = psf_weight.shape[0]
#    cdef Py_ssize_t NJ = int(NJK/K)
#    
#    #define HERQUINST STRUC 
#    cdef HERQUINST herquinst  
#
#    #   RHO DEFINITIONS
#    cdef my_type rhoB = 0.
#    cdef my_type rhoD1 = 0.
#    cdef my_type rhoD2 = 0.
#    cdef my_type rhoB_avg = 0.
#    cdef my_type rhoD1_avg = 0.
#    cdef my_type rhoD2_avg = 0.
#    cdef my_type sum_rhoD1 = 0.         #mi serve per mediare la vel
#    cdef my_type sum_rhoD2 = 0.         #mi serve per mediare la vel
#    cdef my_type sum_rhoB = 0.         #mi serve per mediare la vel
#    cdef my_type LM_sum = 0.         #mi serve per mediare la vel
#    cdef my_type LM_den = 0.         #mi serve per mediare la vel
#    #per il momento la creo 100X100X100 perchè così sono i dati
#    #si può migliorare
#    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
#    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
#    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
#    
#    #V DEFINITIONS
#    cdef my_type v_D1 = 0.
#    cdef my_type v_D2 = 0.
#    cdef my_type v_D1_2 = 0.
#    cdef my_type v_D2_2 = 0.
#    cdef my_type vB_2 = 0.
#    cdef my_type v_tmp = 0.
#    cdef my_type v_tmp2 = 0.
#    cdef my_type sigmaD1_2 = 0.
#    cdef my_type sigmaD2_2 = 0.
#    cdef my_type sigmaB_2 = 0.
#    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
#
#    #questa è una cosa inutile quando tutto funziona va modificato
#    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((NJ,4))
#    cdef my_type[:,::1] tot_view = tot
#    
#    cdef np.ndarray[my_type,ndim=1,mode='c'] all_rhoB = np.zeros((NJK))
#    cdef my_type[::1] all_rhoB_view = all_rhoB
#
#    #PSF definitions
#    cdef my_type psf_tmp=0.
#    cdef my_type sum_psf = 0.
#
#    #masse passate in logaritmo
#    Mbh = 10**Mbh
#    Mb  = 10**Mb
#    Md1 = 10**Md1
#    Md2 = 10**Md2
#    Mh  = 10**Mh
#    Rb  = 10**Rb 
#    Rd1 = 10**Rd1
#    Rd2 = 10**Rd2
#    Rh  = 10**Rh 
#
#    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
#    
#    # if output = 0 ==> tot[:,:,0] = rho, tot[:,:,1] = v, tot[:,:,2] = sigma
#    for i in range(NJ):
#        sum_rhoD1 = 0.
#        sum_rhoD2 = 0.
#        sum_psf = 0.
#        sum_rhoB = 0.
#        sigmaH2_tmp = 0.
#        vD1 = 0.
#        vD2 = 0.
#        vD1_2 = 0.
#        vD2_2 = 0.
#        vB_2 = 0.
#        k_mean = 0.
#        LM_sum = 0.
#        LM_den = 0.
#        for k in range(i*K,(i+1)*K):
#            l = int(k-i*K)
#            psf_tmp = psf_weight[l]
#            #queste operazioni è meglio farle in una funzione?
#            #per il fatto che sarebbe cdef e nogil...
#            x0 = x[k]-xcm
#            y0 = y[k]-ycm
#            xr = x0*c_theta + y0*s_theta 
#            yr = y0*c_theta - x0*s_theta
#            yd = yr/c_incl
#            r_proj = radius(x0,y0) 
#            r_true = radius(xr,yd)
#            phi = atan2(yd,xr)
#
#            herquinst = herquinst_functionBH(Mbh,Mb,Rb,r_proj)
#            rhoD1 = L_to_M_d1*rho_D(Md1,Rd1,r_true,c_incl) 
#            rhoD2 = L_to_M_d2*rho_D(Md2,Rd2,r_true,c_incl) 
#            rhoB  = L_to_M_b*herquinst.rho
#                    
#            sum_rhoD1 = sum_rhoD1 + rhoD1*psf_tmp
#            sum_rhoD2 = sum_rhoD2 + rhoD2*psf_tmp
#            sum_rhoB  = sum_rhoB + rhoB*psf_tmp
#
#            tot_view[i,0] += psf_tmp*(rhoB+rhoD1+rhoD2)
#
#            v_tmp        = myv_tot_BH(Mbh,Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh,s_incl,r_true,phi)
#            v_tmp2       = myv_abs2_BH(Mbh,Mb,Rb,Md1,Rd1,Md2,Rd2,Mh,Rh,s_incl,r_true,phi)
#            sigma_2      = herquinst.sigma
#            vD1          += rhoD1*v_tmp*psf_tmp
#            vD2          += rhoD2*v_tmp*psf_tmp
#            vD1_2        += rhoD1*v_tmp2*psf_tmp
#            vD2_2        += rhoD2*v_tmp2*psf_tmp                                        
#            vB_2         += rhoB*sigma_2*psf_tmp
#
#            k_mean       += rhoB*psf_tmp*0.0 + rhoD1*psf_tmp*k_D1 + rhoD2*psf_tmp*k_D2
#            sum_psf      += psf_tmp
#            LM_sum       += psf_tmp*(rhoB + rhoD1 + rhoD2)
#            LM_den       += psf_tmp*(rhoB/L_to_M_b + rhoD1/L_to_M_d1 + rhoD2/L_to_M_d2)
#        
#        tot_view[i,3] = LM_sum/LM_den
#        tot_view[i,0] = tot_view[i,0]/sum_psf
#
#        rhoD1_avg = sum_rhoD1/sum_psf
#        rhoD2_avg = sum_rhoD2/sum_psf
#        rhoB_avg = sum_rhoB/sum_psf
#
#        vD1  = vD1/sum_rhoD1 * k_D1
#        vD2  = vD2/sum_rhoD2 * k_D2
#
#        sigmaD1_2 = vD1_2/sum_rhoD1 * (1.0-k_D1*k_D1)
#        sigmaD2_2 = vD2_2/sum_rhoD2 * (1.0-k_D2*k_D2)
#        sigmaB_2  = vB_2/sum_rhoB 
#        
#        k_mean    = k_mean/(sum_rhoB+sum_rhoD1+sum_rhoD2)
#
#        tot_view[i,1] = v_avg(rhoD1_avg,rhoD2_avg,rhoB_avg,vD1,vD2,0.0)
#        tot_view[i,2] = sqrt(sigma_avg(rhoD1_avg,rhoD2_avg,rhoB_avg,sigmaD1_2/3.0,sigmaD2_2/3.0,sigmaB_2))
#        
#    return tot#,all_rhoB
#
#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
#@cython.cdivision(True)
#cpdef my_type likelihood_hernquist_BH (my_type[::1] x,          #x position refined grid è 99x99x100
#                                    my_type[::1] y,          #y position refined grid
#                                    my_type Mbh,
#                                    my_type Mb,                  #bulge mass log
#                                    my_type Rb,                  #bulge radius
#                                    my_type Md1,                  #disc mass log
#                                    my_type Rd1,                  #disc radius
#                                    my_type Md2,                  #disc mass log
#                                    my_type Rd2,                  #disc radius
#                                    my_type Mh,                  #halo mass log
#                                    my_type Rh,                  #halo radius
#                                    my_type xcm,                 #center x position
#                                    my_type ycm,                 #center y position
#                                    my_type theta,               #P.A. angle (rotation in sky-plane)
#                                    my_type incl,                #inclination angle
#                                    my_type L_to_M_b,        #const M/L bulge
#                                    my_type L_to_M_d1,        #const M/L bulge
#                                    my_type L_to_M_d2,        #const M/L bulge
#                                    my_type k_D1,        #const M/L bulge
#                                    my_type k_D2,        #const M/L bulge
#                                    my_type sys_rho,
#                                    my_type[::1]psf_weight,
#                                    my_type[::1]rho_data,
#                                    my_type[::1]v_data,
#                                    my_type[::1]sigma_data,
#                                    my_type[::1]ML_data,
#                                    my_type[::1]rho_error,
#                                    my_type[::1]v_error,
#                                    my_type[::1]sigma_error,
#                                    my_type[::1]ML_error):  #PSF_weights 
#                                                                                 # if output != 0 Jaffe
#    
#    cdef Py_ssize_t i,j,k,l
#    cdef Py_ssize_t NJK = x.shape[0]
#    cdef Py_ssize_t K = psf_weight.shape[0]
#    cdef Py_ssize_t NJ = int(NJK/K)
#
#    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((NJ,4))
#    cdef my_type[:,::1] tot_view = tot
#
#    cdef my_type rho_lk   = 0.
#    cdef my_type v_lk     = 0.
#    cdef my_type sigma_lk = 0.
#    cdef my_type ML_lk    = 0.
#    cdef my_type rho_err_tmp   = 0.
#    cdef my_type v_err_tmp     = 0.
#    cdef my_type sigma_err_tmp = 0.
#    cdef my_type ML_err_tmp    = 0.
#    cdef my_type delta_rho  = 0.
#    cdef my_type delta_v    = 0.
#    cdef my_type delta_sigma= 0.
#    cdef my_type delta_ML   = 0.
#    
#    cdef my_type lk   = 0.
#
#    tot =  model_hernquist_BH(x,          #x position refined grid è 99x99x100
#                            y,          #y position refined grid
#                            Mbh,
#                            Mb,                  #bulge mass log
#                            Rb,                  #bulge radius
#                            Md1,                  #disc mass log
#                            Rd1,                  #disc radius
#                            Md2,                  #disc mass log
#                            Rd2,                  #disc radius
#                            Mh,                  #halo mass log
#                            Rh,                  #halo radius
#                            xcm,                 #center x position
#                            ycm,                 #center y position
#                            theta,               #P.A. angle (rotation in sky-plane)
#                            incl,                #inclination angle
#                            L_to_M_b,        #const M/L bulge
#                            L_to_M_d1,        #const M/L bulge
#                            L_to_M_d2,        #const M/L bulge
#                            k_D1,        #const M/L bulge
#                            k_D2,        #const M/L bulge
#                            psf_weight)
#
#    
#    for i in range(NJ):
#        if isnan(rho_data[i]) == True:
#            rho_lk = 0.
#        else:
#            rho_err_tmp = rho_error[i]
#            rho_err_tmp = rho_err_tmp*rho_err_tmp + (sys_rho/10**rho_data[i]/log(10))*(sys_rho/10**rho_data[i]/log(10))
#            delta_rho   = rho_data[i]-log10(tot[i,0])
#            rho_lk      = (delta_rho*delta_rho)/rho_err_tmp+log(2.0*M_PI*rho_err_tmp)
#            #print(delta_rho,rho_lk)
#        if isnan(v_data[i]) == True:
#            v_lk  = 0.
#        else:
#            v_err_tmp = v_error[i]
#            v_err_tmp = v_err_tmp*v_err_tmp
#            delta_v   = v_data[i]-tot[i,1]
#            v_lk      = delta_v*delta_v/v_err_tmp+log(2.0*M_PI*v_err_tmp)
#            #print(delta_v,v_lk)
#        if isnan(sigma_data[i]) == True:
#            sigma_lk = 0.
#        else:
#            sigma_err_tmp = sigma_error[i]
#            sigma_err_tmp = sigma_err_tmp*sigma_err_tmp
#            delta_sigma   = sigma_data[i]-tot[i,2]
#            sigma_lk = delta_sigma*delta_sigma/sigma_err_tmp+log(2.0*M_PI*sigma_err_tmp)
#            #print(delta_sigma,sigma_lk)
#        if isnan(ML_data[i]) == True:
#            ML_lk = 0.
#        else:
#            ML_err_tmp = ML_error[i]
#            ML_err_tmp = ML_err_tmp*ML_err_tmp
#            delta_ML   = ML_data[i]-(1./tot[i,3])
#            ML_lk = delta_ML*delta_ML/ML_err_tmp+log(2*M_PI*ML_err_tmp)
#            #print(ML_data[i],tot[i,3],ML_lk)
#
#        lk += -0.5*(rho_lk+v_lk+sigma_lk+ML_lk)
#    
#    return lk

##############################################################################################################################
##############################################################################################################################
#   HERQUINST+BH (2 visible components) MODEL AND LIKELIHOOD
##############################################################################################################################
##############################################################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[my_type,ndim=2,mode='c'] model_hernquist_BD_BH (my_type[::1] x,          #x position refined grid è 99x99x100
                                                                 my_type[::1] y,          #y position refined grid
                                                                 my_type Mbh,
                                                                 my_type Mb,                  #bulge mass log
                                                                 my_type Rb,                  #bulge radius
                                                                 my_type Md1,                  #disc mass log
                                                                 my_type Rd1,                  #disc radius
                                                                 my_type Mh,                  #halo mass log
                                                                 my_type Rh,                  #halo radius
                                                                 my_type xcm,                 #center x position
                                                                 my_type ycm,                 #center y position
                                                                 my_type theta,               #P.A. angle (rotation in sky-plane)
                                                                 my_type incl,                #inclination angle
                                                                 my_type L_to_M_b,        #const M/L bulge
                                                                 my_type L_to_M_d1,        #const M/L bulge
                                                                 my_type k_D1,        #const M/L bulge
                                                                 my_type[::1]psf_weight):  #PSF_weights 
                                                                                 # if output != 0 Jaffe
 
    cdef my_type sigmaS2 = 0.0    
    cdef my_type x0 = 0.               #x-xcm  traslation
    cdef my_type y0 = 0.               #y-ycm  traslation
    cdef my_type xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef my_type yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef my_type yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef my_type r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef my_type r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef my_type c_theta = cos(theta)  #cos(theta) calculated once
    cdef my_type s_theta = sin(theta)  #sin(theta) calculated once
    cdef my_type c_incl = cos(incl)    #cos(incl) calculated once
    cdef my_type s_incl = sin(incl)    #cos(incl) calculated once
    cdef my_type phi = 0.
    cdef Py_ssize_t i,j,k,l
    cdef Py_ssize_t NJK = x.shape[0]
    cdef Py_ssize_t K = psf_weight.shape[0]
    cdef Py_ssize_t NJ = int(NJK/K)
    
    #define HERQUINST STRUC 
    cdef HERQUINST herquinst  

    #   RHO DEFINITIONS
    cdef my_type rhoB = 0.
    cdef my_type rhoD1 = 0.
    cdef my_type rhoB_avg = 0.
    cdef my_type rhoD1_avg = 0.
    cdef my_type sum_rhoD1 = 0.         #mi serve per mediare la vel
    cdef my_type sum_rhoB = 0.         #mi serve per mediare la vel
    cdef my_type LM_sum = 0.         #mi serve per mediare la vel
    cdef my_type LM_den = 0.         #mi serve per mediare la vel
    #per il momento la creo 100X100X100 perchè così sono i dati
    #si può migliorare
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    
    #V DEFINITIONS
    cdef my_type v_D1 = 0.
    cdef my_type v_D2 = 0.
    cdef my_type v_D1_2 = 0.
    cdef my_type vB_2 = 0.
    cdef my_type v_tmp = 0.
    cdef my_type v_tmp2 = 0.
    cdef my_type sigmaD1_2 = 0.
    cdef my_type sigmaB_2 = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma

    #questa è una cosa inutile quando tutto funziona va modificato
    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((NJ,4))
    cdef my_type[:,::1] tot_view = tot
    
    #PSF definitions
    cdef my_type psf_tmp=0.
    cdef my_type sum_psf = 0.

    #masse passate in logaritmo
    Mbh = 10**Mbh
    Mb  = 10**Mb
    Md1 = 10**Md1
    Mh  = 10**Mh
    Rb  = 10**Rb 
    Rd1 = 10**Rd1
    Rh  = 10**Rh 

    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    
    # if output = 0 ==> tot[:,:,0] = rho, tot[:,:,1] = v, tot[:,:,2] = sigma
    for i in range(NJ):
        sum_rhoD1 = 0.
        sum_psf = 0.
        sum_rhoB = 0.
        sigmaH2_tmp = 0.
        vD1 = 0.
        vD1_2 = 0.
        vB_2 = 0.
        k_mean = 0.
        LM_sum = 0.
        LM_den = 0.
        for k in range(i*K,(i+1)*K):
            l = int(k-i*K)
            psf_tmp = psf_weight[l]
            #queste operazioni è meglio farle in una funzione?
            #per il fatto che sarebbe cdef e nogil...
            x0 = x[k]-xcm
            y0 = y[k]-ycm
            xr = x0*c_theta + y0*s_theta 
            yr = y0*c_theta - x0*s_theta
            yd = yr/c_incl
            r_proj = radius(x0,y0) 
            r_true = radius(xr,yd)
            phi = atan2(yd,xr)

            herquinst = herquinst_functionBH(Mbh,Mb,Rb,r_proj)
            rhoD1 = L_to_M_d1*rho_D(Md1,Rd1,r_true,c_incl) 
            rhoB  = L_to_M_b*herquinst.rho
                    
            sum_rhoD1 = sum_rhoD1 + rhoD1*psf_tmp
            sum_rhoB  = sum_rhoB + rhoB*psf_tmp

            tot_view[i,0] += psf_tmp*(rhoB+rhoD1)

            v_tmp        = myv_tot_BD_BH(Mbh,Mb,Rb,Md1,Rd1,Mh,Rh,s_incl,r_true,phi)
            v_tmp2       = myv_abs2_BD_BH(Mbh,Mb,Rb,Md1,Rd1,Mh,Rh,s_incl,r_true,phi)
            sigma_2      = herquinst.sigma
            vD1          += rhoD1*v_tmp*psf_tmp
            vD1_2        += rhoD1*v_tmp2*psf_tmp
            vB_2         += rhoB*sigma_2*psf_tmp

            k_mean       += rhoB*psf_tmp*0.0 + rhoD1*psf_tmp*k_D1 
            sum_psf      += psf_tmp
            LM_sum       += psf_tmp*(rhoB + rhoD1)
            LM_den       += psf_tmp*(rhoB/L_to_M_b + rhoD1/L_to_M_d1)
        
        tot_view[i,3] = LM_sum/LM_den
        tot_view[i,0] = tot_view[i,0]/sum_psf

        rhoD1_avg = sum_rhoD1/sum_psf
        rhoB_avg = sum_rhoB/sum_psf

        vD1  = vD1/sum_rhoD1 * k_D1

        sigmaD1_2 = vD1_2/sum_rhoD1 * (1.0-k_D1*k_D1)
        sigmaB_2  = vB_2/sum_rhoB 
        
        k_mean    = k_mean/(sum_rhoB+sum_rhoD1)

        tot_view[i,1] = rhoD1_avg*vD1/(rhoB_avg+rhoD1_avg)
        tot_view[i,2] = sqrt((rhoD1_avg*sigmaD1_2/3.0+rhoB_avg*sigmaB_2)/(rhoB_avg+rhoD1_avg))
        
    return tot#,all_rhoB

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef my_type likelihood_hernquist_BD_BH (my_type[::1] x,          #x position refined grid è 99x99x100
                                          my_type[::1] y,          #y position refined grid
                                          my_type Mbh,
                                          my_type Mb,                  #bulge mass log
                                          my_type Rb,                  #bulge radius
                                          my_type Md1,                  #disc mass log
                                          my_type Rd1,                  #disc radius
                                          my_type Mh,                  #halo mass log
                                          my_type Rh,                  #halo radius
                                          my_type xcm,                 #center x position
                                          my_type ycm,                 #center y position
                                          my_type theta,               #P.A. angle (rotation in sky-plane)
                                          my_type incl,                #inclination angle
                                          my_type L_to_M_b,        #const M/L bulge
                                          my_type L_to_M_d1,        #const M/L bulge
                                          my_type k_D1,        #const M/L bulge
                                          my_type sys_rho,
                                          my_type[::1]psf_weight,
                                          my_type[::1]rho_data,
                                          my_type[::1]v_data,
                                          my_type[::1]sigma_data,
                                          my_type[::1]ML_data,
                                          my_type[::1]rho_error,
                                          my_type[::1]v_error,
                                          my_type[::1]sigma_error,
                                          my_type[::1]ML_error):  #PSF_weights 
                                                                                       # if output != 0 Jaffe
    
    cdef Py_ssize_t i,j,k,l
    cdef Py_ssize_t NJK = x.shape[0]
    cdef Py_ssize_t K = psf_weight.shape[0]
    cdef Py_ssize_t NJ = int(NJK/K)

    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((NJ,4))
    cdef my_type[:,::1] tot_view = tot

    cdef my_type rho_lk   = 0.
    cdef my_type v_lk     = 0.
    cdef my_type sigma_lk = 0.
    cdef my_type ML_lk    = 0.
    cdef my_type rho_err_tmp   = 0.
    cdef my_type v_err_tmp     = 0.
    cdef my_type sigma_err_tmp = 0.
    cdef my_type ML_err_tmp    = 0.
    cdef my_type delta_rho  = 0.
    cdef my_type delta_v    = 0.
    cdef my_type delta_sigma= 0.
    cdef my_type delta_ML   = 0.
    
    cdef my_type lk   = 0.

    tot =  model_hernquist_BD_BH(x,          #x position refined grid è 99x99x100
                            y,          #y position refined grid
                            Mbh,
                            Mb,                  #bulge mass log
                            Rb,                  #bulge radius
                            Md1,                  #disc mass log
                            Rd1,                  #disc radius
                            Mh,                  #halo mass log
                            Rh,                  #halo radius
                            xcm,                 #center x position
                            ycm,                 #center y position
                            theta,               #P.A. angle (rotation in sky-plane)
                            incl,                #inclination angle
                            L_to_M_b,        #const M/L bulge
                            L_to_M_d1,        #const M/L bulge
                            k_D1,        #const M/L bulge
                            psf_weight)

    
    for i in range(NJ):
        if isnan(rho_data[i]) == True:
            rho_lk = 0.
        else:
            rho_err_tmp = rho_error[i]
            rho_err_tmp = rho_err_tmp*rho_err_tmp + (sys_rho/10**rho_data[i]/log(10))*(sys_rho/10**rho_data[i]/log(10))
            delta_rho   = rho_data[i]-log10(tot[i,0])
            rho_lk      = (delta_rho*delta_rho)/rho_err_tmp+log(2.0*M_PI*rho_err_tmp)
            #print(delta_rho,rho_lk)
        if isnan(v_data[i]) == True:
            v_lk  = 0.
        else:
            v_err_tmp = v_error[i]
            v_err_tmp = v_err_tmp*v_err_tmp
            delta_v   = v_data[i]-tot[i,1]
            v_lk      = delta_v*delta_v/v_err_tmp+log(2.0*M_PI*v_err_tmp)
            #print(delta_v,v_lk)
        if isnan(sigma_data[i]) == True:
            sigma_lk = 0.
        else:
            sigma_err_tmp = sigma_error[i]
            sigma_err_tmp = sigma_err_tmp*sigma_err_tmp
            delta_sigma   = sigma_data[i]-tot[i,2]
            sigma_lk = delta_sigma*delta_sigma/sigma_err_tmp+log(2.0*M_PI*sigma_err_tmp)
            #print(delta_sigma,sigma_lk)
        if isnan(ML_data[i]) == True:
            ML_lk = 0.
        else:
            ML_err_tmp = ML_error[i]
            ML_err_tmp = ML_err_tmp*ML_err_tmp
            delta_ML   = ML_data[i]-(1./tot[i,3])
            ML_lk = delta_ML*delta_ML/ML_err_tmp+log(2*M_PI*ML_err_tmp)
            #print(ML_data[i],tot[i,3],ML_lk)

        lk += -0.5*(rho_lk+v_lk+sigma_lk+ML_lk)
    
    return lk


##############################################################################################################################
##############################################################################################################################
#   HERQUINST+BH (only bulge) MODEL AND LIKELIHOOD
##############################################################################################################################
##############################################################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[my_type,ndim=2,mode='c'] model_hernquist_B_BH (my_type[::1] x,          #x position refined grid è 99x99x100
                                                                 my_type[::1] y,          #y position refined grid
                                                                 my_type Mbh,
                                                                 my_type Mb,                  #bulge mass log
                                                                 my_type Rb,                  #bulge radius
                                                                 my_type xcm,                 #center x position
                                                                 my_type ycm,                 #center y position
                                                                 my_type L_to_M_b,        #const M/L bulge
                                                                 my_type[::1]psf_weight):  #PSF_weights 
                                                                                 # if output != 0 Jaffe
 
    cdef my_type sigmaS2 = 0.0    
    cdef my_type x0 = 0.               #x-xcm  traslation
    cdef my_type y0 = 0.               #y-ycm  traslation
    cdef my_type xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef my_type yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef my_type yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef my_type r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef my_type r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef my_type phi = 0.
    cdef Py_ssize_t i,j,k,l
    cdef Py_ssize_t NJK = x.shape[0]
    cdef Py_ssize_t K = psf_weight.shape[0]
    cdef Py_ssize_t NJ = int(NJK/K)
    
    #define HERQUINST STRUC 
    cdef HERQUINST herquinst  

    #   RHO DEFINITIONS
    cdef my_type rhoB = 0.
    cdef my_type rhoD1 = 0.
    cdef my_type rhoB_avg = 0.
    cdef my_type rhoD1_avg = 0.
    cdef my_type sum_rhoD1 = 0.         #mi serve per mediare la vel
    cdef my_type sum_rhoB = 0.         #mi serve per mediare la vel
    cdef my_type LM_sum = 0.         #mi serve per mediare la vel
    cdef my_type LM_den = 0.         #mi serve per mediare la vel
    #per il momento la creo 100X100X100 perchè così sono i dati
    #si può migliorare
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    
    #V DEFINITIONS
    cdef my_type v_D1 = 0.
    cdef my_type v_D2 = 0.
    cdef my_type v_D1_2 = 0.
    cdef my_type vB_2 = 0.
    cdef my_type v_tmp = 0.
    cdef my_type v_tmp2 = 0.
    cdef my_type sigmaD1_2 = 0.
    cdef my_type sigmaB_2 = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma

    #questa è una cosa inutile quando tutto funziona va modificato
    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((NJ,4))
    cdef my_type[:,::1] tot_view = tot
    
    #PSF definitions
    cdef my_type psf_tmp=0.
    cdef my_type sum_psf = 0.

    #masse passate in logaritmo
    Mbh = 10**Mbh
    Mb  = 10**Mb
    Rb  = 10**Rb 

    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    
    # if output = 0 ==> tot[:,:,0] = rho, tot[:,:,1] = v, tot[:,:,2] = sigma
    for i in range(NJ):
        sum_psf = 0.
        sum_rhoB = 0.
        sigmaH2_tmp = 0.
        vB_2 = 0.
        for k in range(i*K,(i+1)*K):
            l = int(k-i*K)
            psf_tmp = psf_weight[l]
            #queste operazioni è meglio farle in una funzione?
            #per il fatto che sarebbe cdef e nogil...
            x0 = x[k]-xcm
            y0 = y[k]-ycm
            r_proj = radius(x0,y0) 

            herquinst = herquinst_functionBH(Mbh,Mb,Rb,r_proj)
            rhoB  = L_to_M_b*herquinst.rho
                    
            sum_rhoB  = sum_rhoB + rhoB*psf_tmp

            tot_view[i,0] += psf_tmp*(rhoB)

            sigma_2      = herquinst.sigma
            vB_2         += rhoB*sigma_2*psf_tmp

            sum_psf      += psf_tmp
        
        tot_view[i,3] = L_to_M_b
        tot_view[i,0] = tot_view[i,0]/sum_psf

        rhoB_avg = sum_rhoB/sum_psf


        sigmaB_2  = vB_2/sum_rhoB 
        

        tot_view[i,1] = 0.
        tot_view[i,2] = sqrt(sigmaB_2)
        
    return tot#,all_rhoB

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef my_type likelihood_hernquist_B_BH (my_type[::1] x,          #x position refined grid è 99x99x100
                                          my_type[::1] y,          #y position refined grid
                                          my_type Mbh,
                                          my_type Mb,                  #bulge mass log
                                          my_type Rb,                  #bulge radius
                                          my_type xcm,                 #center x position
                                          my_type ycm,                 #center y position
                                          my_type L_to_M_b,        #const M/L bulge
                                          my_type sys_rho,
                                          my_type[::1]psf_weight,
                                          my_type[::1]rho_data,
                                          my_type[::1]v_data,
                                          my_type[::1]sigma_data,
                                          my_type[::1]ML_data,
                                          my_type[::1]rho_error,
                                          my_type[::1]v_error,
                                          my_type[::1]sigma_error,
                                          my_type[::1]ML_error):  #PSF_weights 
                                                                                       # if output != 0 Jaffe
    
    cdef Py_ssize_t i,j,k,l
    cdef Py_ssize_t NJK = x.shape[0]
    cdef Py_ssize_t K = psf_weight.shape[0]
    cdef Py_ssize_t NJ = int(NJK/K)

    cdef np.ndarray[my_type,ndim=2,mode='c'] tot = np.zeros((NJ,4))
    cdef my_type[:,::1] tot_view = tot

    cdef my_type rho_lk   = 0.
    cdef my_type v_lk     = 0.
    cdef my_type sigma_lk = 0.
    cdef my_type ML_lk    = 0.
    cdef my_type rho_err_tmp   = 0.
    cdef my_type v_err_tmp     = 0.
    cdef my_type sigma_err_tmp = 0.
    cdef my_type ML_err_tmp    = 0.
    cdef my_type delta_rho  = 0.
    cdef my_type delta_v    = 0.
    cdef my_type delta_sigma= 0.
    cdef my_type delta_ML   = 0.
    
    cdef my_type lk   = 0.

    tot =  model_hernquist_B_BH(x,          #x position refined grid è 99x99x100
                            y,          #y position refined grid
                            Mbh,
                            Mb,                  #bulge mass log
                            Rb,                  #bulge radius
                            xcm,                 #center x position
                            ycm,                 #center y position
                            L_to_M_b,        #const M/L bulge
                            psf_weight)

    
    for i in range(NJ):
        if isnan(rho_data[i]) == True:
            rho_lk = 0.
        else:
            rho_err_tmp = rho_error[i]
            rho_err_tmp = rho_err_tmp*rho_err_tmp + (sys_rho/10**rho_data[i]/log(10))*(sys_rho/10**rho_data[i]/log(10))
            delta_rho   = rho_data[i]-log10(tot[i,0])
            rho_lk      = (delta_rho*delta_rho)/rho_err_tmp+log(2.0*M_PI*rho_err_tmp)
            #print(delta_rho,rho_lk)
        if isnan(v_data[i]) == True:
            v_lk  = 0.
        else:
            v_err_tmp = v_error[i]
            v_err_tmp = v_err_tmp*v_err_tmp
            delta_v   = v_data[i]-tot[i,1]
            v_lk      = delta_v*delta_v/v_err_tmp+log(2.0*M_PI*v_err_tmp)
            #print(delta_v,v_lk)
        if isnan(sigma_data[i]) == True:
            sigma_lk = 0.
        else:
            sigma_err_tmp = sigma_error[i]
            sigma_err_tmp = sigma_err_tmp*sigma_err_tmp
            delta_sigma   = sigma_data[i]-tot[i,2]
            sigma_lk = delta_sigma*delta_sigma/sigma_err_tmp+log(2.0*M_PI*sigma_err_tmp)
            #print(delta_sigma,sigma_lk)
        if isnan(ML_data[i]) == True:
            ML_lk = 0.
        else:
            ML_err_tmp = ML_error[i]
            ML_err_tmp = ML_err_tmp*ML_err_tmp
            delta_ML   = ML_data[i]-(1./tot[i,3])
            ML_lk = delta_ML*delta_ML/ML_err_tmp+log(2*M_PI*ML_err_tmp)
            #print(ML_data[i],tot[i,3],ML_lk)

        lk += -0.5*(rho_lk+v_lk+sigma_lk+ML_lk)
    
    return lk


##############################################################################################################################
##############################################################################################################################
#   SUM LIKELIHOOD FOR SUPER RESOLUTION
##############################################################################################################################
##############################################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef my_type likelihood_superresolution (my_type sys_rho,
                                          my_type[:,::1] tot,          #x position refined grid è 99x99x100
                                          my_type[::1]rho_data,
                                          my_type[::1]v_data,
                                          my_type[::1]sigma_data,
                                          my_type[::1]ML_data,
                                          my_type[::1]rho_error,
                                          my_type[::1]v_error,
                                          my_type[::1]sigma_error,
                                          my_type[::1]ML_error):  #PSF_weights 
                                                                                       # if output != 0 Jaffe
    
    cdef Py_ssize_t i,j,k,l
    cdef Py_ssize_t NJ = rho_data.shape[0]


    cdef my_type rho_lk   = 0.
    cdef my_type v_lk     = 0.
    cdef my_type sigma_lk = 0.
    cdef my_type ML_lk    = 0.
    cdef my_type rho_err_tmp   = 0.
    cdef my_type v_err_tmp     = 0.
    cdef my_type sigma_err_tmp = 0.
    cdef my_type ML_err_tmp    = 0.
    cdef my_type delta_rho  = 0.
    cdef my_type delta_v    = 0.
    cdef my_type delta_sigma= 0.
    cdef my_type delta_ML   = 0.
    
    cdef my_type lk   = 0.

    
    for i in range(NJ):
        if isnan(rho_data[i]) == True:
            rho_lk = 0.
        else:
            rho_err_tmp = rho_error[i]
            rho_err_tmp = rho_err_tmp*rho_err_tmp + (sys_rho/10**rho_data[i]/log(10))*(sys_rho/10**rho_data[i]/log(10))
            delta_rho   = rho_data[i]-log10(tot[i,0])
            rho_lk      = (delta_rho*delta_rho)/rho_err_tmp+log(2.0*M_PI*rho_err_tmp)
            #print(delta_rho,rho_lk)
        if isnan(v_data[i]) == True:
            v_lk  = 0.
        else:
            v_err_tmp = v_error[i]
            v_err_tmp = v_err_tmp*v_err_tmp
            delta_v   = v_data[i]-tot[i,1]
            v_lk      = delta_v*delta_v/v_err_tmp+log(2.0*M_PI*v_err_tmp)
            #print(delta_v,v_lk)
        if isnan(sigma_data[i]) == True:
            sigma_lk = 0.
        else:
            sigma_err_tmp = sigma_error[i]
            sigma_err_tmp = sigma_err_tmp*sigma_err_tmp
            delta_sigma   = sigma_data[i]-tot[i,2]
            sigma_lk = delta_sigma*delta_sigma/sigma_err_tmp+log(2.0*M_PI*sigma_err_tmp)
            #print(delta_sigma,sigma_lk)
        if isnan(ML_data[i]) == True:
            ML_lk = 0.
        else:
            ML_err_tmp = ML_error[i]
            ML_err_tmp = ML_err_tmp*ML_err_tmp
            delta_ML   = ML_data[i]-(1./tot[i,3])
            ML_lk = delta_ML*delta_ML/ML_err_tmp+log(2*M_PI*ML_err_tmp)
            #print(ML_data[i],tot[i,3],ML_lk)

        lk += -0.5*(rho_lk+v_lk+sigma_lk+ML_lk)
    
    return lk

       