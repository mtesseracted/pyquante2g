# implement helper functions and expose the basic math functions  needed
# 
import sys
import mpmath as mpm
from fractions import Fraction


mpm.mp.dps = 32 # decimal precision, rougly double float64
# mpm.mp.prec = 53 # float64 mantissa

## Helper functions
def frac_approx(n, eps=10*sys.float_info.epsilon):
    return Fraction(n).limit_denominator(int(1.0/eps))

def num2arb(n, eps=10*sys.float_info.epsilon):
    """ convert a number to arbitrary precision 
        it will convert things like 0.1, or float(1/3) to an mpfloat
        while preserving the inferred rational value. Note if the 
        number was exact up to the last bit in floating point,
        this will probably convert it to an incorrect mpfloat value.
    >>> num2arb(0.1) == mpm.mpf("0.1")
    True
    """
    if isinstance(n, (str, mpm.mpf)):
        # return mpm.mpf(n)
        return mpm.mpmathify(n) # supports complex
    if isinstance(n, (float, int)):
        n = frac_approx(n, eps=eps)
    return mpm.mpf(n.numerator) / mpm.mpf(n.denominator)

arb = lambda x: num2arb(x) # shorthand for num2arb
arr_init = lambda x: mpm.matrix([arb(xx) for xx in x])

def arange(*args):
    return range(*[int(mpm.nint(a)) for a in args])


## Math functions
nint = lambda *args, **kwargs: mpm.nint(*args, **kwargs)
sqrt = lambda *args, **kwargs: mpm.sqrt(*args, **kwargs)
exp = lambda *args, **kwargs: mpm.exp(*args, **kwargs)
log = lambda *args, **kwargs: mpm.log(*args, **kwargs)
lgamma = lambda *args, **kwargs: mpm.loggamma(*args, **kwargs)
gammainc = lambda *args, **kwargs: mpm.gammainc(*args, **kwargs)
binom = lambda *args, **kwargs: mpm.binomial(*args, **kwargs)


## Constants
def set_constants():
    global eps, azero, aone, ahalf, pi
    azero = arb(0)
    aone = arb(1)
    ahalf = arb(0.5)
    pi = 4 * mpm.atan(aone)
    eps = arb(2)**arb(-mpm.mp.prec+1)
set_constants() # define first at import

def set_dps(dps):
    mpm.mp.dps = dps
    set_constants()

def set_prec(prec):
    mpm.mp.prec = prec
    set_constants()


if __name__ == "__main__":
    # 256 digits of pi from `bc -l <<< "scale=256; 4*a(1)"`
    pi256  = "3.141592653589793238462643383279502884197169399375105820974944592307"
    pi256 += "81640628620899862803482534211706798214808651328230664709384460955058"
    pi256 += "22317253594081284811174502841027019385211055596446229489549303819644"
    pi256 += "288109756659334461284756482337867831652712019091456484"


    for dps in [10, 20, 80, 128]:
        mpm.mp.dps = dps
        # pigt = mpm.mpf(pi256)
        pigt = arb(pi256)
        pi = 4 * mpm.atan(1)
        err = pi - pigt
        print(f"{dps:4d}\t{float(err):.4e}")

    debug_var = 1
