inliers = w > 0
X = np.stack([h[inliers], w[inliers]])
# X.shape = [2, 37] = [F, N]