#123456789012345678901234567890123456789012345678901234567890 # 60 limit
import os
import cv2
from matplotlib import pyplot as plt
plt.ion()

def look_and_feel(image, window_title):
    while True:
        cv2.imshow(window_title,image)
        key = cv2.waitKey(1) & 0xFF
        if (key == ord('w') or
            key == ord('s') or  
            key == ord('a') or  
            key == ord('d') or  
            key == ord('j') or  
            key == ord('k')):
            return chr(key)

def gray2bgr(grayimg):
    return cv2.cvtColor(grayimg, cv2.COLOR_GRAY2RGB) 
data_dir = '../data-pnu/'
imgnames = os.listdir(data_dir)
imgpaths = map(lambda path:os.path.join(data_dir,path),imgnames)
imgs = map(lambda p: cv2.imread(p,0), imgpaths)
for img in imgs:
    # plot histogram
    #hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #plt.hist(img.ravel(),256,[0,256]); plt.show()

    # return binary mask
    _,not_nucleuses = cv2.threshold(img,70,1,cv2.THRESH_BINARY)
    '''
    not_nucleuses = cv2.adaptiveThreshold(
                   img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                   cv2.THRESH_BINARY,501,30
                 )
    '''
    holed_img = gray2bgr(img * not_nucleuses)

    nucleuses = gray2bgr(cv2.bitwise_not(not_nucleuses*255))
    nucleuses[:,:,0] = 0
    nucleuses[:,:,1] = 0

    img4check = holed_img + nucleuses

    cv2.imshow('img', img)
    cv2.imshow('nucleuses', cv2.bitwise_not(nucleuses))
    cv2.imshow('not nucleuses', not_nucleuses*255)
    #cv2.imshow('holed_img', holed_img)
    cv2.imshow('image for checking', img4check)
    cv2.waitKey(0)
    print('----')

    plt.close()

# use threshold adaptive to histogram!
# and then dilation?
