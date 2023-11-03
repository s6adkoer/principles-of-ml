import itertools as it

def phi(x):
    n = len(x)
    # Generate all possible sets of x
    sets = it.chain.from_iterable(it.combinations(x, r) for r in range(n+1))
    # Multiply each set together to a single value and return these as an array
    return np.array([np.prod(s, dtype=int) for s in sets], dtype=int)
