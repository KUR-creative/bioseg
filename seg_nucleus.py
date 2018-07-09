#123456789012345678901234567890123456789012345678901234567890 # 60 limit
import os, sys
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

def mod_add(augend,addend, mod_n):
    return ((augend + addend) + mod_n) % mod_n

def gray2bgr(grayimg):
    return cv2.cvtColor(grayimg, cv2.COLOR_GRAY2RGB) 

data_dir = '../data-pnu/'
imgnames = os.listdir(data_dir)
imgpaths = map(lambda path:os.path.join(data_dir,path),imgnames)
imgs = list(map(lambda p: cv2.imread(p,0), imgpaths))

#global_threshold = True
idx = 0
threshold = 70

global_threshold = False
block_size = 17 # odd number!
C = 6
#adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
while True:
    img = imgs[idx]
    if global_threshold:
        _,not_nucleuses = cv2.threshold(img,threshold,1,cv2.THRESH_BINARY)
        holed_img = gray2bgr(img * not_nucleuses)

        nucleuses = gray2bgr(cv2.bitwise_not(not_nucleuses*255))
        nucleuses[:,:,0] = 0
        nucleuses[:,:,1] = 0

        img4check = holed_img + nucleuses

        cv2.imshow('img', img)
        #cv2.imshow('nucleuses', cv2.bitwise_not(nucleuses))
        cv2.imshow('not nucleuses', not_nucleuses*255)
        #cv2.imshow('holed_img', holed_img)

        cmd = look_and_feel(img4check,'image for checking')
        if cmd == 'a' or cmd == 'd':
            val = -1 if cmd == 'a' else +1
            idx = mod_add(idx, val, len(imgs))
        elif cmd == 'j' or cmd == 'k':
            val = -1 if cmd == 'j' else +1
            threshold += val
        elif cmd == 'q':
            sys.exit(0)

        print('idx =',idx,'threshold =',threshold)
    else:
        not_nucleuses = cv2.adaptiveThreshold(
                          img,1,adaptive_method,
                          cv2.THRESH_BINARY,block_size,C
                        )
        holed_img = gray2bgr(img * not_nucleuses)

        nucleuses = gray2bgr(cv2.bitwise_not(not_nucleuses*255))
        nucleuses[:,:,0] = 0
        nucleuses[:,:,1] = 0

        img4check = holed_img + nucleuses

        cv2.imshow('img', img)
        #cv2.imshow('nucleuses', cv2.bitwise_not(nucleuses))
        cv2.imshow('not nucleuses', not_nucleuses*255)
        #cv2.imshow('holed_img', holed_img)

        cmd = look_and_feel(img4check,'image for checking')
        if cmd == 'a' or cmd == 'd':
            val = -1 if cmd == 'a' else +1
            idx = mod_add(idx, val, len(imgs))
        elif cmd == 'j' or cmd == 'k':
            val = -1 if cmd == 'j' else +1
            C += val
        elif cmd == 's' or cmd == 'w':
            val = -2 if cmd == 'j' else +2
            block_size += val
        elif cmd == 'q':
            sys.exit(0)

        print('idx =',idx,'block_size =',block_size,'C =',C)


    #plt.close()

# use threshold adaptive to histogram!
# and then dilation?

# plot histogram
#hist = cv2.calcHist([img],[0],None,[256],[0,256])
#plt.hist(img.ravel(),256,[0,256]); plt.show()
# return binary mask
