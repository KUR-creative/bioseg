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
  python get_cleanmasks.py ./35crops/test/label/
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

mask_dir = sys.argv[1]
mask_paths = list(utils.file_paths(mask_dir))

masks \
= pipe(cmap(lambda path: cv2.imread(path,0)),
       cfilter(lambda img: img is not None),
       cmap(binarization),
       cmap(dilation))

for path,mask in zip(mask_paths, masks(mask_paths)):
    cv2.imwrite(path, mask)

print('Now all masks in %s are clean!' % mask_dir)
