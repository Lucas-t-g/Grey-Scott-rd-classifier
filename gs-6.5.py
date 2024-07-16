from scipy import ndimage

def gs(ma, mb, mc, la, lb, lc, da, db, dc, dt, parms):
    F, K = parms
    ndimage.convolve(ma, kernel_a, output=la, mode='wrap')
    ndimage.convolve(mb, kernel_b, output=lb, mode='wrap')
    na = ma + (- ma * mb * mb + F * (1 - ma) + da * la) * dt
    nb = mb + (  ma * mb * mb - (F + K) * mb + db * lb) * dt
    ma[:] = na
    mb[:] = nb
