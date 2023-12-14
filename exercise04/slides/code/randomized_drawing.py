def get_random_sample(X,k):
    tmp=X[:,np.random.choice(X.shape[1], k, replace=False)]
    return tmp, calc_norm(tmp)

def get_pred(X,k,runs):
    max_sample,max_norm = get_random_sample(X,k)
    for _ in range(runs-1):
        sample, sample_norm=get_random_sample(X,k)
        if sample_norm>max_norm:
            max_sample,max_norm =  sample, sample_norm
    return max_sample, max_norm