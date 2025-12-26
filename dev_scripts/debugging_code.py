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
