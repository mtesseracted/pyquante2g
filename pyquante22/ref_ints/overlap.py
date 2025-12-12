import numpy as np
from pyquante22.ref_ints.utils import GTO

##############################################################################
# Minhhuy Hô and Julio-Manuel Hernández-Pérez, 2012 (hh12)
# https://doi.org/doi:10.3888/tmj.14-3
##############################################################################


def overlap1d_hh12(amom, bmom, acen, bcen, alp, bet) -> float:
    r""" Compute overlap of 1D Gaussians with angular momentum indices, hh12
    ----------
    amom, bmom : angular momentum index
    acen, bcen : center A and B
    alp, bet : gaussian exponent
    -------
    return: float, scalar overlap
    """

    # Base case: both indices are zero
    if amom == 0 and bmom == 0:
        return 1.0

    # Base case: only A has angular momentum
    elif amom == 1 and bmom == 0:
        return -(acen - ((alp * acen + bet * bcen) / (alp + bet)))

    # Recursive case: decrement amom
    elif bmom == 0:
        term1 = -(acen - (alp * acen + bet * bcen) / (alp + bet)) * \
            overlap1d_hh12(amom - 1, 0, acen, bcen, alp, bet)
        term2 = ((amom - 1) / (2.0 * (alp + bet))) * \
            overlap1d_hh12(amom - 2, 0, acen, bcen, alp, bet)
        return term1 + term2

    # Transfer angular momentum from B to A
    else:
        term1 = overlap1d_hh12(amom + 1, bmom - 1, acen, bcen, alp, bet)
        term2 = (acen - bcen) * overlap1d_hh12(amom, bmom - 1, acen, bcen, alp, bet)
        return term1 + term2
#end def overlap1d_hh12(amom, bmom, acen, bcen, alp, bet):


# def overlap_hh12(alpha, beta, acen, bcen, amom, bmom):
def overlap_hh12(ga:GTO, gb:GTO):
    r""" Compute overlap of 3D Gaussians with angular momentum indices.
    ----------
    amom, bmom : angular momentum indices 
    acen, bai : center A and B
    alpha, beta : gaussian exponent
    -------
    return : float, scalar overlap
    """
    alpha, acen, amom = ga.arl
    beta, bcen, bmom = gb.arl

    # Pre-factor terms
    r1 = (alpha * beta) / (alpha + beta)
    r12m = np.sum((acen - bcen) ** 2)
    eab = np.exp(-r1 * r12m)
    norm_prefactor = (np.pi / (alpha + beta)) ** 1.5

    # 1D overlap integrals along each axis
    sx = overlap1d_hh12(amom[0], bmom[0], acen[0], bcen[0], alpha, beta)
    sy = overlap1d_hh12(amom[1], bmom[1], acen[1], bcen[1], alpha, beta)
    sz = overlap1d_hh12(amom[2], bmom[2], acen[2], bcen[2], alpha, beta)

    return sx * sy * sz * eab * norm_prefactor
#end def overlap_hh12(alpha, beta, acen, bcen, amom, bmom):


#def kinetic_hh12(alpha, beta, acen, bcen, amom, bmom) -> float:
def kinetic_hh12(ga:GTO, gb:GTO):
    r""" Compute kinetic expections of 3D Cartesian Gaussians
    ----------
    amom, bmom : angular momentum indices 
    acen, bai : center A and B
    alpha, beta : gaussian exponent
    -------
    return : float, scalar expectation
    """
    # ! (\partial_x)^2 ( (x-x0)^n exp(-a(x-x0)^2) ) = (\partial)^2 g(a,x0,n)
    # ! = n(n-1)*g(a,x0,n-2) - 2a(2n+1)*g(a,x0,n) + 4a^2*g(a,x0,n+2)
    # ! \nabla^2 ( g_x * g_y * g_z ) = g''_x * g_y * g_z 
    # !                              + g_x * g''_y * g_z
    # !                              + g_x * g_y * g''_z
    # ! This means we need to iterate 3 times
    # ! this method double calculates all the non-derivative overlaps!
    alpha, acen, amom = ga.arl
    beta, bcen, bmom = gb.arl
    gc = ga.clone(l=np.copy(ga.l))
    offs = 2 * np.eye(3, dtype=gc.l.dtype)

    term0 = term1 = term2 = 0.0
    ret = 0.0
    for ii in range(3):              #! loop over which coord is 2nd deriv

        ov_n = overlap_hh12(gc, gb) #alpha, beta, acen, bcen, a1, a2)
        gc.l += offs[ii]
        ov_np2 = overlap_hh12(gc, gb) #alpha, beta, acen, bcen, a1, a2)
        gc.l -= offs[ii] # reset

        ret = ret - alpha * float(2*(2*amom[ii]+1)) * ov_n 
        ret = ret + 4.0 * alpha * alpha * ov_np2

        if ( amom[ii] > 1 ):
            gc.l -= offs[ii]
            ov_nm2 = overlap_hh12(gc, gb) #alpha, beta, acen, bcen, a1, a2)
            gc.l += offs[ii] # reset
            ret = ret + float(amom[ii]*(amom[ii]-1)) * ov_nm2 

    ret = -0.5 * ret  #!  kinetic = -1/2 \nabla^2
    return ret
#end function gauss_kinetic


# @TODO
##############################################################################
# Hiroshi Taketa, Sigeru Huzinaga, Kiyoshi O-ohata, 1966 (THO66)
# https://doi.org/10.1143/JPSJ.21.2313
##############################################################################


overlap_funcs = [overlap_hh12,]
kinetic_funcs = [kinetic_hh12,]

if __name__ == "__main__":

    debug_var = 1