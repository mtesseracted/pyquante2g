import pytest
import pyquante22.utils.math as math

numbers = [0, 1, 2, 3, 4]
facts = [1, 1, 2, 6, 24]

@pytest.fixture(params=list(zip(numbers, facts)), ids=lambda p: f"{p[0]}-{p[1]}")
def fact_pairs(request):
    return request.param


def test_fact_exact(fact_pairs):
    n, f = fact_pairs
    assert math.fact_exact(n) == f


# import torch as th
# def fact_exact(n:int): # use python's arbitrary int
#     if n < 2:
#         return 1
#     return n * fact_exact(n-1)

# def binom_exact(n:int, k:int): # use python's arbitrary int
#     if k > n:
#         return 0
#     return fact_exact(n) // (fact_exact(k) * fact_exact(n-k))

# def binom2_exact(n:int, k:int): # use python's arbitrary int
#     if 2*k > n:
#         return 0
#     return fact_exact(n) // (fact_exact(k) * fact_exact(n-2*k))



# def test_binom2_large():
#     test_nks = [
#         [4, 1],
#         [4, 2],
#         [4, 3],
#         [12, 4],
#         [12, 5],
#         [12, 6],
#         [12, 7],
#         # [20, 0],
#         # [20, 1],
#         # [20, 2],
#         # [20, 4],
#         # [20, 8],
#         # [20, 9],
#         # [20, 10],
#         # [20, 11],
#         # [40, 0],
#         # [40, 1],
#         # [40, 2],
#         # [40, 4],
#         # [40, 8],
#         # [40, 16],
#         # [40, 19],
#         # [40, 20],
#         # [40, 21],
#     ]
#     for n,k in test_nks:
#         gt = th.as_tensor(binom2_exact(n,k), dtype=am.dtype)
#         nt, kt = th.as_tensor(n, dtype=am.dtype), th.as_tensor(k, dtype=am.dtype)
#         assert th.equal(am.binom2_ch(nt,kt), gt), f"@({n},{k})"
#end def test_binom2_large(digits_case):

