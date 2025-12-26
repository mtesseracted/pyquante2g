try:
    from . import core
    from . import functions as f
except ImportError:
    import core # direct run file
    import functions as f


def A_term(i,r,u,l1,l2,PAx,PBx,CPx,gamma):
    """ THO eq. 2.18 """
    return core.pow(-1,i) * f.binom_pre(i,l1,l2,PAx,PBx) * \
           core.pow(-1,u) * core.pow(CPx,i-2*r-2*u) * \
           core.pow(0.25/gamma,r+u) * f.binom22(i,r,u)


def A_array(l1,l2,PA,PB,CP,g):
    """ THO eq. 2.18 and 3.1 """
    Imax = l1+l2+1
    A = core.arr_init([0]*int(Imax))
    for i in core.arange(Imax):
        for r in range(int(core.floor(i/2)+1)):
            for u in range(int(core.floor((i-2*r)/2)+1)):
                I = i-2*r-u
                A[I] = A[I] + A_term(i,r,u,l1,l2,PA,PB,CP,g)
    return A


def nuclear_attraction(alpha1,lmn1,A,alpha2,lmn2,B,C):
    """ Full form of the nuclear attraction integral, THO Eq. 2.17 """
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    gamma = alpha1+alpha2

    P = f.gto_prod_cen(alpha1,A,alpha2,B)
    rab2 = f.norm2(A-B)
    rcp2 = f.norm2(C-P)

    dPA = P-A
    dPB = P-B
    dPC = P-C

    Ax = A_array(l1,l2,dPA[0],dPB[0],dPC[0],gamma)
    Ay = A_array(m1,m2,dPA[1],dPB[1],dPC[1],gamma)
    Az = A_array(n1,n2,dPA[2],dPB[2],dPC[2],gamma)

    total = 0.
    for I in core.arange(l1+l2+1):
        for J in core.arange(m1+m2+1):
            for K in core.arange(n1+n2+1):
                total += Ax[I]*Ay[J]*Az[K]*f.fboys(I+J+K,rcp2*gamma)
                
    val = -2*core.pi/gamma*core.exp(-alpha1*alpha2*rab2/gamma)*total
    return val
#end def nuclear_attraction(alpha1,lmn1,A,alpha2,lmn2,B,C):


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np 
    import scipy.special as sp

    arb = core
    alpha = arb.aone
    ra0 = arb.arr_init([0,]*3)
    am000 = arb.arr_init([0,0,0])
    am100 = arb.arr_init([1,0,0])
    am200 = arb.arr_init([2,0,0])
    am020 = arb.arr_init([0,2,0])
    am002 = arb.arr_init([0,0,2])
    g0 = [alpha, am000, ra0]
    g1 = [alpha, am100, ra0]
    g2x = [alpha, am200, ra0]
    g2y = [alpha, am020, ra0]
    g2z = [alpha, am002, ra0]
    ans0 = nuclear_attraction(*sum([g0,]*2, []), ra0)


    xs = np.linspace(0, 5, 51)
    vals = np.empty_like(xs)
    for ix, x in enumerate(xs):
        ra1 = arb.arr_init([x,0,0])
        ans = nuclear_attraction(*sum([g0,]*2, []), ra1)
        vals[ix] = -float(ans)

    # if our ansatz is c * erf(a*x)/x; @x=0: 2*c*a/sqrt(pi) = 2*pi/gamma
    # then the a/c formula is a = pi**(3/2) / (2*c)
    # if we guess c = 2 -> a = sqrt(pi)/4
    # a = 0.5 * np.sqrt(np.pi)**3/2
    gamma = float(2*g0[0])
    a = np.sqrt(gamma) # a is sqrt of s-type exponent, if each one is 1.0, the combined GTO exponent is 2
    # c = np.pi**(3/2) / (2*a)
    c = 1.0 * (np.pi/gamma) ** (3/2)
    ys = c * sp.erf(a*xs) #/ xs
    ys[1:] /= xs[1:]
    ys[0] = c * 2 * a / np.sqrt(np.pi)

    fig, ax = plt.subplots()
    ax.plot(xs, vals, label="NAI")
    amstr = ','.join(str(int(v[0])) for v in g0[1].tolist())
    ax.plot(xs, ys, label=f"g={gamma:.4f};am=({amstr})")
    ax.legend()
    plt.show()



    xs = np.linspace(0, 5, 51)
    valx = np.empty_like(xs)
    valy = np.empty_like(xs)
    valz = np.empty_like(xs)
    for ix, x in enumerate(xs):
        ra1 = arb.arr_init([x,0,0])
        ans = nuclear_attraction(*g2x, *g0, ra1)
        valx[ix] = -float(ans)
        ans = nuclear_attraction(*g2z, *g0, ra1)
        valy[ix] = -float(ans)
        ans = nuclear_attraction(*g2z, *g0, ra1)
        valz[ix] = -float(ans)
    vals = valx + valy + valz

    # for rho(r) = r^2 exp(-a*r^2); == dx2 + dy2 + dz2
    # v(r) = 3*sqrt(pi)/(8*a**(5/2)) * erf(sqrt(a)r)/r - \
    #       exp(-a*r**2)/(4*a**2) #
    # v(0) = 1/(2*a**2)
    a = float(g0[1]+g2x[1])
    ys  = 3*np.sqrt(np.pi)/(8*a**(5/2)) * sp.erf(np.sqrt(a)*xs)/xs
    ys -= np.exp(-a*xs**2)/(4*a**2)
    ys[0] = 0.5 / a**2
    ys *= 4*np.pi # spherical factor

    fig, ax = plt.subplots()
    ax.scatter(xs, vals, label=r"\sum_{{i=xyz}} NAI(d_i,s|r)")
    amstr = ','.join(str(int(v[0])) for v in g1[1].tolist())
    ax.plot(xs, ys, label=f"g={gamma:.4f};am=({amstr})")
    ax.legend()
    plt.show()







    debug_var = 1
 