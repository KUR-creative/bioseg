import sys, os, cv2 
from fp import pipe, cmap, cfilter
import utils

utils.help_option('''
get_cleanmasks
  load masks from 'mask_dir', 
  and *OVERWIRTE* masks with clean masks.
  
[synopsys]
  python get_cleanmasks.py mask_dir
ex)
  python img_cutter.py 256 ./masks
''')

def binarization(img, threshold=100):
    #cv2.imshow('nb',img); cv2.waitKey(0)
    _,binarized = cv2.threshold(img, threshold, 255, 
                                cv2.THRESH_BINARY)
    #cv2.imshow('b',binarized); cv2.waitKey(0)
    return binarized

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
def dilation(img, kernel=kernel):
    dilated = cv2.dilate(img, kernel, iterations=1)
    #cv2.imshow('dilated',dilated); cv2.waitKey(0)
    return dilated

mask_paths = list(utils.file_paths(sys.argv[1]))

masks \
= pipe(cmap(lambda path: cv2.imread(path,0)),
       cfilter(lambda img: img is not None),
       cmap(binarization),
       cmap(dilation))

for path,mask in zip(mask_paths, masks(mask_paths)):
    cv2.imwrite(path, mask)
