Phi = np.apply_along_axis(phi,1,x.T)

w110 = la.lstsq(Phi, y110 ,rcond=None)[0]
yhat110 = x.T @ w110
