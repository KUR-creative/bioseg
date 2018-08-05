import sys, os, cv2 
from fp import pipe, cmap, cfilter
import numpy as np
import utils
np.set_printoptions(threshold=np.nan, linewidth=np.nan)

utils.help_option('''
get_cleanmasks
  load masks from 'mask_dir', 
  and *OVERWIRTE* masks with clean masks.

  it can know dataset is gray or rgb images.
  but DO NOT MIX graysacle & rgb images!
  
[synopsys]
  python get_cleanmasks.py mask_dir
ex)
  python get_cleanmasks.py ./35crops/test/label/
''')

def binarization(img, threshold=100):
    cv2.imshow('nb',img); cv2.waitKey(0)
    #_,binarized = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    #print(img[:100,:100])
    binarized = (img >= threshold).astype(np.uint8) * 255
    #print((img >= threshold)[:100,:100])
    cv2.imshow('b',binarized); cv2.waitKey(0)
    return binarized

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
def dilation(img, kernel=kernel):
    dilated = cv2.dilate(img, kernel, iterations=1)
    cv2.imshow('dilated',dilated); cv2.waitKey(0)
    return dilated

mask_dir = sys.argv[1]
mask_paths = list(utils.file_paths(mask_dir))

check_img = cv2.imread(mask_paths[0])
if(np.array_equal(check_img[:,:,0], check_img[:,:,1]) and 
   np.array_equal(check_img[:,:,1], check_img[:,:,2])):
    img_type = cv2.IMREAD_GRAYSCALE
else:
    img_type = cv2.IMREAD_COLOR


masks \
= pipe(cmap(lambda path: cv2.imread(path, img_type)),
       cfilter(lambda img: img is not None),
       cmap(binarization),
       cmap(dilation))

for path,mask in zip(mask_paths, masks(mask_paths)):
    cv2.imwrite(path, mask)

print('Now all masks in %s are clean!' % mask_dir)
