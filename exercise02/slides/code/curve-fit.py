def weibull(t, A, alpha, beta):
    ab, tb = alpha / beta, t / beta
    return A * ab * tb**(alpha - 1) * np.exp(-tb**alpha)
(A, alpha, beta), _ = curve_fit(weibull, t, h, p0=[1000, 1.0, 1.0])
