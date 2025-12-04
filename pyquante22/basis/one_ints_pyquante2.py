# one electron integrals
import numpy as np


def gto_overlap_1d(amom, bmom, acen, bcen, alp, bet) -> float:
    r""" 
    Compute overlap of 1D Gaussians with angular momentum indices.

    Parameters
    ----------
    amom, bmom : angular momentum index
    acen, bcen : center A and B
    alp, bet : gaussian exponent

    Returns
    -------
    float : scalar overlap
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
            gto_overlap_1d(amom - 1, 0, acen, bcen, alp, bet)
        term2 = ((amom - 1) / (2.0 * (alp + bet))) * \
            gto_overlap_1d(amom - 2, 0, acen, bcen, alp, bet)
        return term1 + term2

    # Transfer angular momentum from B to A
    else:
        term1 = gto_overlap_1d(amom + 1, bmom - 1, acen, bcen, alp, bet)
        term2 = (acen - bcen) * gto_overlap_1d(amom, bmom - 1, acen, bcen, alp, bet)
        return term1 + term2
#end def gto_overlap_1d(amom, bmom, acen, bcen, alp, bet):


def gto_overlap_3d(alpha, beta, acen, bcen, amom, bmom):
    r""" 
    Compute overlap of 3D Gaussians with angular momentum indices.

    Parameters
    ----------
    amom, bmom : angular momentum indices 
    acen, bai : center A and B
    alpha, beta : gaussian exponent

    Returns
    -------
    float : scalar overlap
    """

    # Pre-factor terms
    r1 = (alpha * beta) / (alpha + beta)
    r12m = np.sum((acen - bcen) ** 2)
    eab = np.exp(-r1 * r12m)
    norm_prefactor = (np.pi / (alpha + beta)) ** 1.5

    # 1D overlap integrals along each axis
    sx = gto_overlap_1d(amom[0], bmom[0], acen[0], bcen[0], alpha, beta)
    sy = gto_overlap_1d(amom[1], bmom[1], acen[1], bcen[1], alpha, beta)
    sz = gto_overlap_1d(amom[2], bmom[2], acen[2], bcen[2], alpha, beta)

    return sx * sy * sz * eab * norm_prefactor
#end def gto_overlap_3d(alpha, beta, acen, bcen, amom, bmom):


def gto_kinetic_3d(alpha, beta, acen, bcen, amom, bmom) -> float:
    # ! find the kinetic energy expectation between two gaussians

    # ! (\partial_x)^2 ( (x-x0)^n exp(-a(x-x0)^2) ) = (\partial)^2 g(a,x0,n)
    # ! = n(n-1)*g(a,x0,n-2) - 2a(2n+1)*g(a,x0,n) + 4a^2*g(a,x0,n+2)

    # ! \nabla^2 ( g_x * g_y * g_z ) = g''_x * g_y * g_z 
    # !                              + g_x * g''_y * g_z
    # !                              + g_x * g_y * g''_z
    # ! This means we need to iterate 3 times
   
    # ! this method double calculates all the non-derivative overlaps!
    # ! need to store them then iterate (have to sep overlaps into 1d)

    a1 = np.zeros_like(amom)
    a2 = bmom
    ret = 0.0
    for ii in range(3):              #! loop over which coord is 2nd deriv

        a1[:] = amom[:]
        ov_n = gto_overlap_3d(alpha, beta, acen, bcen, a1, a2)
        a1[ii] = amom[ii] + 2
        ov_np2 = gto_overlap_3d(alpha, beta, acen, bcen, a1, a2)

        ret = ret - alpha * float(2*(2*amom[ii]+1)) * ov_n 
        ret = ret + 4.0 * alpha * alpha * ov_np2

        if ( amom[ii] > 1 ):
            a1[ii] = amom[ii] - 2
            ov_nm2 = gto_overlap_3d(alpha, beta, acen, bcen, a1, a2)
            ret = ret + float(amom[ii]*(amom[ii]-1)) * ov_nm2 

    ret = -0.5 * ret  #!  kinetic = -1/2 \nabla^2
    # call stop_clock('gauss_kinetic')
    return ret
#end function gauss_kinetic



if __name__ == "__main__":
    ma, mb = 0, 0
    amom, bmom = np.array([ma,]*3), np.array([mb,]*3)
    ra, rb = 0.0, 0.0
    acen, bcen = np.array([ra,]*3), np.array([rb,]*3)
    alpha, beta = 1.0, 1.0

    tmp = gto_overlap_1d(ma, mb, ra, rb, alpha, beta)
    print(f"{tmp}")

    tmp = gto_overlap_3d(alpha, beta, acen, bcen, amom, bmom)
    print(f"{tmp}")

    tmp = gto_kinetic_3d(alpha, beta, acen, bcen, amom, bmom)
    print(f"{tmp}")



    debug_var = 1
