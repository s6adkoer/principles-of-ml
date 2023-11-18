theta_hat=least_squares(lambda theta:get_residuals(N,hs,ts,*theta),\
                        [1,100],bounds=([0,0],np.inf),ftol=1e-14).x