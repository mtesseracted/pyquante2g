# array_ops/numpy.py

import numpy as np
from .core import ArrayBackend
from math import lgamma


def _gser(a,x):
    "Series representation of Gamma. NumRec sect 6.1."
    ITMAX=100
    EPS=3.e-7

    gln=lgamma(a)
    assert(x>=0),'x < 0 in gser'
    if x == 0 : return 0,gln

    ap = a
    delt = sum = 1./a
    for i in range(ITMAX):
        ap=ap+1.
        delt=delt*x/ap
        sum=sum+delt
        if abs(delt) < abs(sum)*EPS: break
    else:
        print('a too large, ITMAX too small in gser')
    gamser=sum*np.exp(-x+a*np.log(x)-gln)
    return gamser,gln
#end def _gser(a,x):


def _gcf(a,x):
    "Continued fraction representation of Gamma. NumRec sect 6.1"
    ITMAX=100
    EPS=3.e-7
    FPMIN=1.e-30

    gln=lgamma(a)
    b=x+1.-a
    c=1./FPMIN
    d=1./b
    h=d
    for i in range(1,ITMAX+1):
        an=-i*(i-a)
        b=b+2.
        d=an*d+b
        if abs(d) < FPMIN: d=FPMIN
        c=b+an/c
        if abs(c) < FPMIN: c=FPMIN
        d=1./d
        delt=d*c
        h=h*delt
        if abs(delt-1.) < EPS: break
    else:
        print('a too large, ITMAX too small in gcf')
    gammcf=np.exp(-x+a*np.log(x)-gln)*h
    return gammcf,gln
#end def _gcf(a,x):


def _gamm_inc(a,x):
    """
    Incomple gamma function \gamma; computed from NumRec routine gammp.
    >>> np.isclose(gamm_inc(0.5,1),1.49365)
    True
    >>> np.isclose(gamm_inc(1.5,2),0.6545103)
    True
    >>> np.isclose(gamm_inc(2.5,1e-12),0)
    True
    """
    assert (x > 0 and a >= 0), "Invalid arguments in routine gamm_inc: %s,%s" % (x,a)

    if x < (a+1.0): #Use the series representation
        gam,gln = _gser(a,x)
    else: #Use continued fractions
        gamc,gln = _gcf(a,x)
        gam = 1-gamc
    return np.exp(gln)*gam
#end def _gamm_inc(a,x):


class NumpyBackend(ArrayBackend):
    name = "numpy"
    xp = np
    ArrayType = np.ndarray

    def __init__(self, device=None, 
                 default_dtype="float64", int_dtype="int64",):
        super().__init__(device=device, 
                         default_dtype=default_dtype, int_dtype=int_dtype)

    def lgamma(self, x):
        return np.vectorize(lgamma)(x)

    def gamma_inc(self, a, x):
        return np.vectorize(_gamm_inc)(a, x)

#end class NumpyBackend(ArrayBackend):


backend_cls = NumpyBackend
