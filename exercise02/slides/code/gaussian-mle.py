µ = np.mean(X, axis=1, keepdims=True)
# µ = [[175.729], [73.865]]
S = np.cov(X - µ, ddof=1) # ddof = degree of freedom = scales with 1/(n-1)
# S == (1 / (N-1)) * (X - µ) @ (X - µ).T
# S == [[ 75.925,  64.546 ],
#       [ 64.546, 186.953]]