import numpy as np
import numpy.linalg as la
import itertools as it

np.set_printoptions(precision=3, suppress=True)

#Task 1.2.1

# Matrix X as defined in the exercise sheet
x = np.array(
    [
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1],
    ]
)

# The Rules 110 and 126 as defined in the exercise sheet
y110 = np.array([1, -1, -1, -1, 1, -1, -1, 1])
y126 = np.array([1, -1, -1, -1, -1, -1, -1, 1])

# Calculate the weights
w110 = la.lstsq(x.T, y110,rcond=None)[0]
w126 = la.lstsq(x.T, y126,rcond=None)[0]

# Calculate yhat
y110_hat = x.T @ w110
y126_hat = x.T @ w126

print("yhat110")
print(y110_hat)
print("yhat126")
print(y126_hat)

#Task 1.2.2

def phi(x):
    n = len(x)
    # Generate all possible sets of x
    sets = it.chain.from_iterable(it.combinations(x, r) for r in range(n+1))
    # Multiply each set together to a single value and return these as an array
    return np.array([np.prod(s, dtype=int) for s in sets],dtype=int)

#Task 1.2.3

# Calculate Matrix Phi
Phi = np.apply_along_axis(phi,1,x.T)
print("Phi")
print(Phi)

# Calculate the weights
w110 = la.lstsq(Phi.T, y110 ,rcond=None)[0]
w126 = la.lstsq(Phi.T, y126 ,rcond=None)[0]

# Calculate yhat
yhat110 = Phi.T @ w110
yhat126 = Phi.T @ w126

print("yhat110")
print(yhat110)
print("yhat126")
print(yhat126)
