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

