theta_opt=minimize(lambda theta_prime:negLikelihood(x,y_bar,theta_prime),\
                   x0=theta,bounds=[(0, None),(0, None),(0, None),(0, None)]).x
