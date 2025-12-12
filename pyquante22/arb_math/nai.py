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
                
    val= -2*core.pi/gamma*core.exp(-alpha1*alpha2*rab2/gamma)*total
    return val
#end def nuclear_attraction(alpha1,lmn1,A,alpha2,lmn2,B,C):


if __name__ == "__main__":
    arb = core
    ra0 = arb.arr_init([0,]*3)
    am000 = arb.arr_init([0,0,0])
    g0 = [arb.aone, am000, ra0]
    ans0 = nuclear_attraction(*sum([g0,]*2, []), ra0)
    debug_var = 1
 