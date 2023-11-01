def box_counting(img):
    w, h = img.shape
    n_ls = []
    # l runs from 1 to 9-2 =7
    for l in range(1,8):
        s_l = 1/2**l
        # get box sizes
        box_sizeW = s_l * w
        box_sizeH = s_l * h
        n_l=0
        #each l has 2**l boxes
        for box_w in range(0,(2**l)):
            for box_h in range(0,(2**l)):
                #check if any value in the box is equal 1.
                #If so increment n_l by one
                if (np.any(img[int(box_w * box_sizeW):int((box_w+1) * box_sizeW),\
                               int(box_h * box_sizeH): int((box_h+1) * box_sizeH)]\
                            ==1)):
                    n_l+=1
        n_ls.append(n_l)
    return n_l