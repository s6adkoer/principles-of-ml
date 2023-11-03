# Solve the Least Squares Problem
w110 = la.lstsq(matXT, y110, rcond=None)

# Calculate yhat
yhat110 = matXT @ w110[0]