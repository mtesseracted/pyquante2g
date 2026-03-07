###########################################################
# old debugging code
###########################################################

# code from unrolling h_array in the CartGtoPrimPair.calc_h_array()
# there are order 1e-15 differences in the output with calc_h_array_gt()
# but after unrolling the largest difference elements then calculating
# the sub-terms for each element at the place with the largest difference,
# all the sub terms agreed. I suspect the difference is thereforse
# some backend out of order execution (OOE) that happens in the vectorized
# code.
if False:
    if False:
        # @DEBUG, compare to numpy calculated
        hgt = th.as_tensor(self.calc_h_array_gt())
        hgtf = self.calc_h_array_gt(full=True)

        # get matching unrolled elements worse at ret[3,1], diff=7.1e-15
        ibatch = 3
        ii = 1
        iflats = th.where(iiflat == ii)
        these = h_vals_flat[ibatch][iflats]

        ih = th.as_tensor(hgtf[ibatch])
        iir = th.where((irng-2*rrng)[:ih.shape[0],:ih.shape[1]]==ii)
        those = ih[iir]

        irid = th.abs(these - those[:len(these)]).argmax() # not actual rr, place in iflats
        i, r = iir[0][irid], iir[1][irid]

        diff = these.sum() - those.sum()

        factgt = binom2_ch(arr_init(i), arr_init(r))
        fact = binom2_ch(iflat, rflat)[iflats[0][irid]]
        diff = fact - factgt # 0.0
        if diff != fzero:
            debug_var = 1

        pa = self.pab - self.ga.cen
        pb = self.pab - self.gb.cen
        bpgt = binomial_prefactor(i.item(), self.ga.amom[ibatch].item(), self.gb.amom[ibatch].item(),
            pa[ibatch].item(), pb[ibatch].item())
        diff = bps[ibatch,i] - arr_init(bpgt) # 0.0
        if diff != fzero:
            debug_var = 1

        gpgt = np.pow(4*self.gamma[ibatch].item(), (r-i).item())
        gp = th.pow(g4s[:,irmask], (rflat-iflat)[None,:])[ibatch,iflats[0][irid]]
        diff = gp - gpgt
        if diff != fzero:
            debug_var = 1


# debugging in CartGtoPrimPair.calc_c_array()
if False:
    if False:
        # @DEBUG
        gts = th.as_tensor(self.calc_c_array_gt(other)) # @DEBUG
        gts = F.pad(gts, (0, ret.shape[1]-gts.shape[1]))
        # h0 = th.as_tensor(self.calc_h_array_gt())
        # h1 = self.calc_h_array()
        # h0 = F.pad(h0, (0, h1.shape[1]-h0.shape[1]))
        pdiffs = th.abs(ret - gts) / th.clamp(th.abs(gts), 1.0e-14, None)
        pdab = th.abs(self.hab - hab) / th.clamp(th.abs(hab), 1.0e-14, None)
        pdcd = th.abs(other.hab - hcd) / th.clamp(th.abs(hcd), 1.0e-14, None)
        # @DEBUG


## pretty print html for CartGtoPrim
if False:
    def html_row(self,table,i=0):
        """
        Add a row in a table for an html representation of an atom
        """
        import xml.etree.ElementTree as ET
        i0, i1 = i*self.nd, (i+1)*self.nd
        ll = ao.tolist(self.amom[i0:i1])
        xx = ao.tolist(self.cen[i0:i1])
        ee = self.alpha[i0].item()
        tr = ET.SubElement(table,"tr")
        td = ET.SubElement(tr,"td")
        td.text = str(i)
        for ii in range(self.nd):
            td = ET.SubElement(tr,"td")
            td.text = f"{ll[ii]}"
        for ii in range(self.nd):
            td = ET.SubElement(tr,"td")
            td.text = f"{xx[ii]:8.2f}"
        td = ET.SubElement(tr,"td")
        td.text = f"{ee:8.2e}"
        return

    def html(self,tablehead=True):
        import xml.etree.ElementTree as ET
        top = ET.Element("p")
        h2 = ET.SubElement(top,"h2")
        h2.text = "Cartesian GTO Primitive (CartGtoPrim)"
        p = ET.SubElement(top,"p")
        p.text = f"Hash: {self.hash}"
        table = ET.SubElement(top,"table")
        if tablehead:
            tr = ET.SubElement(table,"tr")
            for item in ["#","Lx","Ly","Lz","Rx","Ry","Rz","Exp"]:
                th = ET.SubElement(tr,"th")
                th.text = item
        for i in range(len(self)):
            self.html_row(table,i)
        return top

    def _html_repr_(self,tablehead=True):
        import xml.etree.ElementTree as ET
        top = ET.Element("html")
        top.append(self.html(tablehead=tablehead))
        return ET.tostring(top)

# ...existing code...
# array_ops instance is injected via set_array_ops() to allow arbitrary backends
AO = None
ArrType = None

# module-level constants filled by set_array_ops()
one = two = zero = half = pi = arange_max = binoms = binom2s = ghalfs = None
amax = 20

def set_array_ops(ao_inst):
    """Register an array_ops instance for this module.
    Call this once (or whenever you switch backend) before using CartGtoPrim.
    """
    global AO, ArrType, one, two, zero, half, pi, arange_max, binoms, binom2s, ghalfs, amax
    AO = ao_inst
    ArrType = AO.ArrayType
    one = AO.arr(1)
    two = AO.arr(2)
    zero = AO.arr(0)
    half = AO.arr(0.5)
    pi = AO.arr(4) * AO.atan(one)
    arange_max = AO.arange(6*amax+1)
    # build precomputed combinatorics using existing `me` helpers in this module
    binoms = AO.arr([[me.binom_exact(n, k) for k in range(2*amax+1)] for n in range(2*amax+1)])
    binom2s = AO.arr([[me.binom2_exact(n, k) for k in range(1*amax+1)] for n in range(2*amax+1)])

    def dfact2m1o2(n):
        """ double factorial of 2n-1 over 2**n """
        return AO.exp(AO.lgamma(n + 0.5*one)) / AO.sqrt(pi)

    ghalfs = dfact2m1o2(arange_max[:2*amax+1])
    globals()["ghalfs"] = ghalfs
    globals()["binoms"] = binoms
    globals()["binom2s"] = binom2s

def _ensure_ao():
    if AO is None:
        raise RuntimeError("array_ops not set for cart_gto module. Call set_array_ops(my_ao) first.")

def conv_full_batch(a, b):
    _ensure_ao()
    return AO.conv_transpose1d(a[None, :, :], b[:, None, :], groups=a.shape[0]).squeeze(0)
# ...existing code...


# def calc_kinetic2(gpair, nd=3):
#     """ formulas are hard coded to nd=3 """
#     ga, gb = gpair.ga, gpair.gb
#     bsz = (len(ga)//nd, nd)
#     amb = gb.amom.view(bsz)
#     ab = gb.alpha[::nd]
#     term0 = ab * (2*ao.sum(amb, axis=1) + 3) * \
#         gpair.calc_overlap()

#     offsets = 2 * ao.eye(nd, dtype=gpair.ga.amom.dtype)
#     term1  = gpair.calc_overlap(offset=offsets[0])
#     term1 += gpair.calc_overlap(offset=offsets[1])
#     term1 += gpair.calc_overlap(offset=offsets[2])
#     term1 *= -2 * ao.square(ab)

#     l2 = gb.amom[0::nd]
#     term2  = l2 * (l2-1) * gpair.calc_overlap(offset=-offsets[0])
#     l2 = gb.amom[1::nd]
#     term2 += l2 * (l2-1) * gpair.calc_overlap(offset=-offsets[1])
#     l2 = gb.amom[2::nd]
#     term2 += l2 * (l2-1) * gpair.calc_overlap(offset=-offsets[2])
#     term2 *= -half

#     return term0 + term1 + term2


# code from testing accuracy of calc_nai()
'''
def ndindex(shape):
    import itertools
    return itertools.product(*[range(n) for n in shape]) # generator

def fboys_arb(m, x): # @DEBUG with arbitrary precision math
    import pyquante22.arb_math as arb
    full_sz = tuple(max(a,b) for a,b in zip(m.shape,x.shape))
    mb, xb = ao.broadcast_to(m, full_sz), ao.broadcast_to(x, full_sz)
    ret = ao.zeros(full_sz)
    for idx in ndindex(full_sz):
        ret[idx] = ao.arr(float(arb.fboys(mb[idx].item(), xb[idx].item())))
    return ret

def triple_sum_dot(ax, ay, az, fxyz):
    # for debugging
    totals = ao.zeros(ax.shape[0])
    for ii in range(ax.shape[1]):
        for jj in range(ay.shape[1]):
            for kk in range(az.shape[1]):
                totals += ax[:,ii] * ay[:,jj] * az[:,kk] * fxyz[:,ii+jj+kk]
    return totals
'''
if False:

    def calc_a_array_arb(self, rcen:ArrType) -> ArrType:
        import pyquante22.arb_math as arb
        dPn = self.pab - rcen
        dPa = self.pab - self.ga.cen
        dPb = self.pab - self.gb.cen
        abmom = self.ga.amom + self.gb.amom
        llmax = abmom.max().item()
        full_sz = (len(self), llmax + 1,)
        ret = ao.zeros(full_sz)
        for ib in range(full_sz[0]):
            barr = arb.A_array(
                self.ga.amom[ib].item(), self.gb.amom[ib].item(), 
                dPa[ib].item(), dPb[ib].item(), 
                dPn[ib].item(), self.gamma[ib].item()
            )
            for ii, v in enumerate(barr):
                ret[ib,ii] = float(v)

            debug_var = 1
            aargs = (
                self.ga.amom[ib].item(), self.gb.amom[ib].item(), 
                dPa[ib].item(), dPb[ib].item(), 
                dPn[ib].item(), self.gamma[ib].item()
            )
            l1,l2,PA,PB,CP,g = aargs
            Imax = l1+l2+1
            import mpmath as mpm
            A2d = mpm.zeros(Imax+1, 2*Imax+1)
            iis = [0 for _ in range(int(2*Imax+1))]
            for i in arb.arange(Imax):
                for r in range(int(arb.floor(i/2)+1)):
                    for u in range(int(arb.floor((i-2*r)/2)+1)):
                        I = i-2*r-u
                        term =  arb.A_term(i,r,u,l1,l2,PA,PB,CP,g)
                        A2d[I,iis[I]] = term
                        iis[I] += 1


        return ret

        # code from inside calc_nai()
        # @DEBUG
        import pyquante22.arb_math as arb
        a2 = self.calc_a_array_arb(rcen).reshape(bsz, nd, -1).transpose(0, 1) # @DEBUG
        boy2 = fboys_arb(arange_max[: ijkmax + 1][None, :], fbx[:, None]) # @DEBUG
        ret2 = triple_conv_dot(a2[0], a2[1], a2[2], boys)
        ret2 *= -2*pi/gab * ao.exp(-aa*ab*rab2/gab)
        ret3 = triple_sum_dot(a2[0], a2[1], a2[2], boys)
        ret3 *= -2*pi/gab * ao.exp(-aa*ab*rab2/gab)
        # ret4 = triple_sum_dot(a2[0], a2[1], a2[2], boys)
        # # check the prefactor, assumes batch size is 1:
        # agab = arb.from_float(gab.item())
        # aaa = arb.from_float(aa.item())
        # aab = arb.from_float(ab.item())
        # arab2 = arb.from_float(rab2.item())
        # afac = -2*arb.pi/agab * arb.exp(-aaa*aab*arab2/agab)
        # gt = float(afac)
        # fac = (-2*pi/gab * ao.exp(-aa*ab*rab2/gab)).item()

        aerrs = a_array - a2
        mask = ao.abs(a2) > 0
        aerrs[mask] /= a2[mask]
        aemx = ao.abs(aerrs).max()

        berrs = boys - boy2
        mask = ao.abs(boy2) > 0
        berrs[mask] /= boy2[mask]
        bemx = ao.abs(berrs).max()

        diff = ret - ret2
        perr = diff/ret2
        if ao.abs(perr).item() > 1e-14:
            debug_var = 1
        # del fbx, boys

        return ret3 # @DEBUG
        # return ret


##################################################################################################
# the main section of cart_gto.py on 02Mar26, when switching to GPU
##################################################################################################
if False:
    loops = 100
    t0 = time.time()
    for _ in range(loops):
        cas = gp12.calc_c_array(gp12)
    t1 = time.time()
    print(f"orig method took {t1-t0:.4f}s for {loops} loops")

    t0 = time.time()
    for _ in range(loops):
        cbs = gp12.calc_c_self()
    t1 = time.time()
    print(f"new method took {t1-t0:.4f}s for {loops} loops")

    for i in range(len(cas)):
        minl = cbs[i].shape[-1]
        extra = cas[i][:,:,minl:]
        if extra.nelement() > 0:
            assert aoi.abs(extra).max() == 0, f"nonzero extra in c array {i}"
        assert aoi.abs(cas[i][:,:,:minl] - cbs[i]).max() < 1e-16, "c array self mismatch"


    # gp1 = sum([gna, gs, gmc, goa]) # needs check_valid in calc_eri_self
    gp1 = sum([gnb, gsb, gmb, gob])
    gp2 = sum([gsb, gnb, gob, gmb])
    gp12 = gp1 * gp2

    loops = 100
    t0 = time.time()
    for _ in range(loops):
        eas = gp12.calc_eri(gp12)
    t1 = time.time()
    print(f"orig method took {t1-t0:.4f}s for {loops} loops")

    t0 = time.time()
    for _ in range(loops):
        ebs = gp12.calc_eri_self()
    t1 = time.time()
    print(f"new method took {t1-t0:.4f}s for {loops} loops")

    assert aoi.abs(eas - ebs).max() < 1e-16, "eri self mismatch"


    # loops = 500
    # t0 = time.time()
    # for _ in range(loops):
    #     gab = sum(gp_list) * sum(gp_list[-2::-1])
    # t1 = time.time()
    # print(f"{loops} inits took {t1-t0:.4f}s")


    # one body ops test:
    gab = sum(gp_list) * sum(gp_list[-2::-1])
    methods = ["calc_overlap1d", "calc_overlap", "calc_kinetic", "calc_h_array"]
    results = [getattr(gab, f)() for f in methods]
    results = [getattr(gab, f)() for f in methods] # call twice to make sure data isn't corrupted

    for i,j in ndindex(gab.shape):
        gabl = tolegacy(gab[i,j])
        for ii, f in enumerate(methods):
            gt = getattr(gabl, f)()
            ans = results[ii][i,j]
            gshap, ashap = gt.shape, ans.shape
            if gt.nelement() < 2:
                a, b = ans.item(), gt.item()
                if b == 0:
                    assert a == b
                else:
                    assert aoi.abs((gt-ans)/gt).item() < 1e-15
                continue
            assert len(gshap) == len(ashap), "ndim mismatch"
            min_shape = tuple(min(a, b) for a, b in zip(gt.shape, ans.shape))
            overlap_slices = tuple(slice(0, m) for m in min_shape)
            gt2 = gt[overlap_slices]
            an2 = ans[overlap_slices]
            assert aoi.sum(aoi.abs(an2-gt2)) == 0.0, f"mismatch in {f}"
            mask = aoi.ones(min_shape, dtype=bool)
            if min_shape != gt.shape: # check extras
                gt[overlap_slices] = 0
                assert aoi.abs(aoi.sum(gt)) == 0, f"gt extras not 0 {f}"
            if min_shape != ans.shape: # check extras
                ans[overlap_slices] = 0
                assert aoi.abs(aoi.sum(ans)) == 0, f"ans extras not 0 {f}"

            debug_var = 1


    # two body ops test:
    gp4 = sum(gp_list[:4])
    gp3 = sum(gp_list[4:7])
    gp2 = sum(gp_list[:-3:-1])
    gp1 = sum(gp_list[-2:-3:-1])
    gab, gcd = gp2*gp3, gp1*gp4
    rcens = gcd.pab.reshape(-1, gcd.nd)


    methods = ["calc_a_array", "calc_nai",]
    results = [getattr(gab, f)(rcens) for f in methods]
    results = [getattr(gab, f)(rcens) for f in methods] # call twice to make sure data isn't corrupted

    for i,j in ndindex(gab.shape):
        gabl = tolegacy(gab[i,j])
        for k in range(rcens.shape[0]):
            for ii, f in enumerate(methods):
                gt = getattr(gabl, f)(rcens[k])
                ans = results[ii][i,j,k]
                gshap, ashap = gt.shape, ans.shape
                if gt.nelement() < 2:
                    assert gt.item() == ans.item()
                    continue
                assert len(gshap) == len(ashap), "ndim mismatch"
                min_shape = tuple(min(a, b) for a, b in zip(gt.shape, ans.shape))
                overlap_slices = tuple(slice(0, m) for m in min_shape)
                gt2 = gt[overlap_slices]
                an2 = ans[overlap_slices]
                assert aoi.sum(aoi.abs(an2-gt2)) == 0.0, f"mismatch in {f}"
                mask = aoi.ones(min_shape, dtype=bool)
                if min_shape != gt.shape: # check extras
                    gt[overlap_slices] = 0
                    assert aoi.abs(aoi.sum(gt)) == 0, f"gt extras not 0 {f}"
                if min_shape != ans.shape: # check extras
                    ans[overlap_slices] = 0
                    assert aoi.abs(aoi.sum(ans)) == 0, f"ans extras not 0 {f}"

                debug_var = 1


    methods = ["calc_c_array", "calc_eri",]
    results = [getattr(gab, f)(gcd) for f in methods]
    results = [getattr(gab, f)(gcd) for f in methods] # call twice to make sure data isn't corrupted

    for i,j in ndindex(gab.shape):
        gabl = tolegacy(gab[i,j])
        for k, l in ndindex(gcd.shape):
            gcdl = tolegacy(gcd[k,l])

            for ii, f in enumerate(methods):
                gt = getattr(gabl, f)(gcdl)
                if f == "calc_c_array": # handle list output
                    stack = aoi.stack(results[ii], axis=0)
                    ans = stack.reshape(gab.nd, *gab.shape, *gcd.shape, -1)[:,i,j,k,l]
                else:
                    ans = results[ii][i,j,k,l]
                if gt.nelement() < 2:
                    assert gt.item() == ans.item()
                    continue

                gshap, ashap = gt.shape, ans.shape
                assert len(gshap) == len(ashap), "ndim mismatch"

                min_shape = tuple(min(a, b) for a, b in zip(gt.shape, ans.shape))
                overlap_slices = tuple(slice(0, m) for m in min_shape)
                gt2 = gt[overlap_slices]
                an2 = ans[overlap_slices]
                assert aoi.xp.allclose(an2, gt2, rtol=1e-14, atol=1e-16), f"mismatch in {f}"
                # assert aoi.sum(aoi.abs(an2-gt2)) == 0.0, f"mismatch in {f}"


                mask = aoi.ones(min_shape, dtype=bool)
                if min_shape != gt.shape: # check extras
                    gt[overlap_slices] = 0
                    assert aoi.abs(aoi.sum(gt)) == 0, f"gt extras not 0 {f}"
                if min_shape != ans.shape: # check extras
                    ans[overlap_slices] = 0
                    assert aoi.abs(aoi.sum(ans)) == 0, f"ans extras not 0 {f}"

                debug_var = 1

    debug_var = 2

    # CartGtoPrim (Lx,Ly,Lz) (Rx,Ry,Rz) Exp
    # 4  2  0        0.0000       0.5000      -0.2500    2.0000e+00, 
    # CartGtoPrim (Lx,Ly,Lz) (Rx,Ry,Rz) Exp
    # 0  0  0        0.1250       0.1250       0.1250    1.0000e+00
    ama = aoi.arr([4, 2, 0], dtype=aoi.int_dtype)
    amb = aoi.arr([0, 0, 0], dtype=aoi.int_dtype)
    ra = aoi.arr([0, 0.5, -0.25])
    rb = aoi.arr([0.125, 0.125, 0.125])
    aa = aoi.arr([2.0])
    ab = aoi.arr([1.0])
    ga = CartGtoPrim(ama, ra, aa, ao=aoi, co=coi)
    gb = CartGtoPrim(amb, rb, ab, ao=aoi, co=coi)
    gab = CartGtoPrimPair(ga, gb)
    ans = gab.calc_kinetic().item()
    gabl = tolegacy(gab)
    ref = gabl.calc_kinetic().item()

    import numpy as np
    import pyquante22.arb_math as arb
    from pyquante22.ref_ints.utils import GTO


    def to_np(a):  # to numpy
        if hasattr(a, "shape"):
            return np.asarray(a.tolist())
        return np.asarray(a.item())

    def cgp_scalar(gto: CGP):
        l = to_np(gto.amom)
        r = to_np(gto.cen)
        if hasattr(l, "ndim") and l.ndim == 2:
            l = l[0]
        if hasattr(r, "ndim") and r.ndim == 2:
            r = r[0]
        return l, r, to_np(gto.alpha[0])


    def gto2arb(gto):
        if isinstance(gto, GTO):
            l = arb.arr_init(gto.l.tolist())
            r = arb.arr_init(gto.r.tolist())
            a = arb.arb(gto.a.item())
            return [a, l, r]
        elif isinstance(gto, CGP):
            l, r, a = cgp_scalar(gto)
            return gto2arb(GTO(l=l, r=r, a=a))
        else:
            raise ValueError(f"Does not support type {type(gto)}")

    def kin_ref(*args):
        """Kinetic reference, args is one CartGtoPrimPair"""
        ab = args[0]  # one CGPP
        la, ra, aa = cgp_scalar(ab.ga)
        lb, rb, abeta = cgp_scalar(ab.gb)
        a = GTO(l=la, r=ra, a=aa)
        b = GTO(l=lb, r=rb, a=abeta)
        fargs = sum([gto2arb(g) for g in [a, b]], [])
        return arb.kinetic_tho66(*fargs)

    gt = float(kin_ref(gab))

    err1 = (ans - ref) /ref
    err2 = (ref - gt) / gt
    err3 = (ans - gt) / gt
    print(f"{coi.amax} {err1:.4e} {err2:.4e} {err3:.4e}")

    debug_var = 1

 
