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
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.hist(img.ravel(),256,[0,256]); plt.show()
    cv2.imshow('img',img);
    cv2.waitKey(0)
    plt.close()
