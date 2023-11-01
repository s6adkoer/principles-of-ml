def slope(n_l):
    inverted_s_l=[2**l for l in range(1,8)]
    matX=np.vander(np.log(inverted_s_l),2,increasing=True)
    b,D=la.lstsq(matX, np.log(n_l),rcond=None)[0]
    return b,D