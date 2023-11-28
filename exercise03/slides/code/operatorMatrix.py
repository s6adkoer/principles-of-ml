def outer_operator(f,u,v):
    return f(np.expand_dims(u,axis=1), np.expand_dims(v,axis=0))