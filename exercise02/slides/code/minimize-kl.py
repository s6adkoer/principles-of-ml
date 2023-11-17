def KL(f, q): return np.sum(f * np.log(f / q))

def objective(x): 
    return KL(weibull(t, alpha=x[0], beta=x[1]), q)
result = minimize(objective, x0=[1.0, 100.], bounds=[(0, 10), (0, 500)])
