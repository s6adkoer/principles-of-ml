def MLE_Weibull(d,h, parameter, epoch):
    
    for e in range(epoch):
        alpha=parameter[0,0]
        betha=parameter[1,0]

        delAlpha = np.sum(h*(1/alpha+np.log(t/betha)-
                            (t/betha)**alpha*np.log(t/betha)))
        delBeta = np.sum(h*((alpha/betha)*(((d/betha)**alpha)-1)))
        delAlpha2 = -np.sum(h*((1/(alpha**2))+ 
                                 (((d/betha)**alpha)*((np.log(d/betha))**2))))
        delBeta2 = np.sum(h*(alpha/(betha**2)*(1-(alpha+1)*((d/betha)**alpha))))
        delAlphaBeta = np.sum(h*(1/betha*((d/betha)**alpha)+(alpha/betha)
                                  *((d/betha)**alpha*np.log(d/betha))-(1/betha)))
        matrix_inv = inv(np.array([[delAlpha2, delAlphaBeta],
              [delAlphaBeta, delBeta2]]))
        

        parameter = parameter+ (matrix_inv @ np.array([[-delAlpha],[-delBeta]]))
    return parameter