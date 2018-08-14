import sys, os, cv2 
from fp import pipe, cmap, cfilter
import numpy as np
import utils
np.set_printoptions(threshold=np.nan, linewidth=np.nan)

utils.help_option('''
clean_masks
  load masks from 'mask_dir', 
  clean masks, merge 'selected_channel' to get graysacle image,
  and then *OVERWIRTE* masks with clean masks.

  it can know dataset is gray or rgb images.
  but DO NOT MIX graysacle & rgb images!
  
[synopsys]
  python clean_masks.py mask_dir selected_channel
ex)
  python clean_masks.py ./35crops/test/label/ rg
''')

def binarization(img, threshold=100):
    #cv2.imshow('nb',img); cv2.waitKey(0)
    #_,binarized = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    #print(img[:100,:100])
    binarized = (img >= threshold).astype(np.uint8) * 255
    #print((img >= threshold)[:100,:100])
    #cv2.imshow('b',binarized); cv2.waitKey(0)
    return binarized

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
def dilation(img, kernel=kernel):
    dilated = cv2.dilate(img, kernel, iterations=1)
    #cv2.imshow('dilated',dilated); cv2.waitKey(0)
    return dilated

def grayscale(bgr_img, merging_channels='rgb'):
    ''' 
    merge 'merging_channels' to 1 channel graysacle image.
    and then return the graysacle image.
    '''
    h,w,c = bgr_img.shape
    if c == 1:
        return bgr_img
    r = g = b = 0
    if 'b' in merging_channels: b = bgr_img[:,:,0]
    if 'g' in merging_channels: g = bgr_img[:,:,1]
    if 'r' in merging_channels: r = bgr_img[:,:,2]
    merged = b + g + r
    return merged.reshape((h,w,1))

def remove_intersection(bgr_img):
    '''
    Remove overlapping parts of different labels
    leave color in priority: blue > green > red
    '''
    if len(bgr_img.shape) == 3 and bgr_img.shape[2] == 3:
        b = bgr_img[:,:,0]
        g = bgr_img[:,:,1]
        r = bgr_img[:,:,2]
        #calculate and leave difference set of lower priority color
        #(remove intersection)
        r_minus_b = (r != b).astype(np.uint8) * r
        bgr_img[:,:,2] = r_minus_b
        r_minus_g = (r != g).astype(np.uint8) * r
        bgr_img[:,:,2] = r_minus_g
        g_minus_b = (g != b).astype(np.uint8) * g
        bgr_img[:,:,1] = g_minus_b
    #cv2.imshow('intersection removed',bgr_img); cv2.waitKey(0)
    return bgr_img


if __name__ == '__main__':
    mask_dir = sys.argv[1]
    mask_paths = list(utils.file_paths(mask_dir))

    check_img = cv2.imread(mask_paths[0])
    if(np.array_equal(check_img[:,:,0], check_img[:,:,1]) and 
       np.array_equal(check_img[:,:,1], check_img[:,:,2])):
        img_type = cv2.IMREAD_GRAYSCALE
    else:
        img_type = cv2.IMREAD_COLOR

    leaving_channels = sys.argv[2] if len(sys.argv) == 3 else 'rgb'

    masks \
    = pipe(cmap(lambda path: cv2.imread(path, img_type)),
           cfilter(lambda img: img is not None),
           cmap(binarization),
           cmap(dilation),
           cmap(remove_intersection),
           cmap(lambda img: grayscale(img, leaving_channels)))

    for path,mask in zip(mask_paths, masks(mask_paths)):
        cv2.imwrite(path, mask)

    print('Now all masks in %s are clean!' % mask_dir)
