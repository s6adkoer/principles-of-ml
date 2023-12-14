import random
def farthest_first_traversal(X, num_points=3):
    data = X.T
    subset = [random.choice(data)] # Select the first point randomly

    while len(subset) < num_points:
        # Calculate vectorized distance matrix between X and subset S
        diff = data[np.newaxis, ...] - np.array(subset)[:, np.newaxis, :]  # N, S, 2
        dist = np.linalg.norm(diff, axis=-1)                               # N, S
    
        min_distances = dist.min(axis=0) # distance determined via nearest subset point
        subset.append(data[np.argmax(min_distances)])  # point furthest away

    return np.array(subset).T