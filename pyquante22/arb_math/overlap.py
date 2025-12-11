# THO'66 formulation of overlap integrals
try:
    from . import core
    from . import functions as f
except ImportError:
    import core # direct run file
    import functions as f
    

##############################################################################
# Hiroshi Taketa, Sigeru Huzinaga, Kiyoshi O-ohata, 1966 (THO66)
# https://doi.org/10.1143/JPSJ.21.2313
##############################################################################

def overlap1d(l1, l2, PAx, PBx, gamma):
    r""" The one-dimensional component of the overlap integral. Taken from THO eq. 2.12
    ----------
    l1, l2 : angular momentum index
    PAx, PBx : distances from Gaussian product center to centers A and B
    gamma : sum of Gaussian exponents
    -------
    return: float, scalar overlap
    """
    total = 0.0
    for i in range(1 + int(core.floor(0.5 * (l1 + l2)))):
        total += f.binom_pre(2 * i, l1, l2, PAx, PBx) * \
            f.gamma_half_int(i) * (gamma ** -i)
    return total
    
def overlap_tho66(alpha1,lmn1,A,alpha2,lmn2,B):
    """
    Full form of the overlap integral. Taken from THO eq. 2.12
    """
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    rab2 = f.norm2(A-B)
    gamma = alpha1+alpha2
    P = f.gto_prod_cen(alpha1,A,alpha2,B)

    pre = (core.pi/gamma**core.arb(1.5))*core.exp(-alpha1*alpha2*rab2/gamma)
    wx = overlap1d(l1,l2,P[0]-A[0],P[0]-B[0],gamma)
    wy = overlap1d(m1,m2,P[1]-A[1],P[1]-B[1],gamma)
    wz = overlap1d(n1,n2,P[2]-A[2],P[2]-B[2],gamma)
    return pre*wx*wy*wz


def kinetic_tho66(alpha, beta, acen, bcen, amom, bmom):
    r""" Compute kinetic expections of 3D Cartesian Gaussians
    ----------
    amom, bmom : angular momentum indices 
    acen, bai : center A and B
    alpha, beta : gaussian exponent
    -------
    return : float, scalar expectation
    """
    l1, m1, n1 = amom
    l2, m2, n2 = bmom
    ov = overlap_tho66
    offs = [core.arr_init([0,]*3) for _ in range(3)]
    offs[0][0] = offs[1][1] == offs[2][2] == core.arb(2)

    term0  = ov(alpha, amom, acen, beta, bmom, bcen)
    term0 *= beta * (2*sum(bmom) + 3)

    term1  = ov(alpha, amom, acen, beta, bmom+offs[0], bcen)
    term1 += ov(alpha, amom, acen, beta, bmom+offs[1], bcen)
    term1 += ov(alpha, amom, acen, beta, bmom+offs[2], bcen)
    term1 *= -2 * (beta*beta)

    term2  = l2*(l2-1) * ov(alpha, amom, acen, beta, bmom-offs[0], bcen)
    term2 += m2*(m2-1) * ov(alpha, amom, acen, beta, bmom-offs[1], bcen)
    term2 += n2*(n2-1) * ov(alpha, amom, acen, beta, bmom-offs[2], bcen)
    term2 *= -core.ahalf

    return term0+term1+term2
#end def kinetic_tho66(alpha, beta, acen, bcen, amom, bmom) -> float:
