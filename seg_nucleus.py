#123456789012345678901234567890123456789012345678901234567890 # 60 limit
import os
import cv2
from matplotlib import pyplot as plt
plt.ion()

data_dir = '../data-pnu/'
imgnames = os.listdir(data_dir)
imgpaths = map(lambda path:os.path.join(data_dir,path),imgnames)
for imgpath in imgpaths:
    img = cv2.imread(imgpath,0)

    # plot histogram
    #hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #plt.hist(img.ravel(),256,[0,256]); plt.show()

    threshold = 70
    _,thr1 = cv2.threshold(img,70,1,cv2.THRESH_BINARY)
    thr2 = cv2.adaptiveThreshold(
             img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
             cv2.THRESH_BINARY,91,2
           )
    holed_img = cv2.cvtColor(img * thr1, cv2.COLOR_GRAY2RGB) 

    cv2.imshow('img', img)
    cv2.imshow('thresh1', thr1*255)
    cv2.imshow('holed_img', holed_img)
    #cv2.imshow('thresh2',thr2)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #cv2.imshow('green_thr1',thr1*255);cv2.waitKey(0)
    nucleuses = cv2.bitwise_not(thr1*255)
    #cv2.imshow('green_thr1',nucleuses);cv2.waitKey(0)
    nucleuses = cv2.cvtColor(nucleuses, cv2.COLOR_GRAY2RGB) 
    #cv2.imshow('green_thr1',nucleuses);cv2.waitKey(0)
    nucleuses[:,:,0] = 0
    nucleuses[:,:,1] = 0
    #nucleuses[:,:,2] = 0
    #cv2.imshow('green_thr1',nucleuses);cv2.waitKey(0)
    #cv2.imshow('green_thr1',nucleuses)
    img4check = holed_img + nucleuses
    cv2.imshow('image for checking', img4check)
    cv2.waitKey(0)
    print('----')

    plt.close()

# use threshold adaptive to histogram!
# and then dilation?
