import sys, os, cv2, tqdm, pathlib
from fp import pipe, cmap, cfilter, flatten
import itertools
import utils

utils.help_option('''
img_cutter
  copy_tree structure of 'src_path' to 'dst_path'
  then save square crops of images in 'src_path'
  
[synopsys]
  python img_cutter.py crop_size src_path dst_path
ex)
  python img_cutter.py 256 ./src ./dst
''')


def hw2start_yxs(origin_yx, img_hw, piece_hw):
    org_y,org_x = origin_yx  
    img_h,img_w = img_hw
    piece_h,piece_w = piece_hw
    for y in range(org_y, img_h, piece_h):
        for x in range(org_x, img_w, piece_w):
            yield (y,x)

def hw2not_excess_start_yxs(origin_yx, img_hw, piece_hw):
    img_h,img_w = img_hw
    piece_h,piece_w = piece_hw
    def not_excess(yx):
        y,x = yx
        return (y + piece_h < img_h) and (x + piece_w < img_w)
    return filter(not_excess, 
                  hw2start_yxs(origin_yx, img_hw, piece_hw))

def safe_path2img(path):
    ret = cv2.imread(path)
    print(path,type(ret))
    return ret

def gen_name_piece_pairs(img, piece_size):
    img_hw = img.shape[:2]
    h = w = piece_size
    for y,x in hw2not_excess_start_yxs( (0,0), img_hw, (h,w) ):
        yield 'piece_%d_%d.png' % (y,x), img[y:y+h,x:x+w]
    for y,x in hw2not_excess_start_yxs( (h//2,w//2), img_hw, (h,w) ):
        yield 'piece_%d_%d.png' % (y,x), img[y:y+h,x:x+w]

imgno = 0
def path_img2path_pieces(path_img, piece_size, imgs_dir, pieces_dir):
    global imgno
    path, img = path_img
    name = os.path.splitext(path)[0]

    old_parent = pathlib.Path(name).parts[0]
    name = utils.make_dstpath(name, old_parent, pieces_dir)

    img_hw = img.shape[:2]
    h = w = piece_size
    imgno += 1
    for r_y,r_x in sorted(itertools.product([0, 0.5], [0, 0.5])):
        org_yx = int(h*r_y),int(w*r_x)
        for y,x in hw2not_excess_start_yxs( org_yx, img_hw, (h,w) ):
            #print(y,x)
            #cv2.imshow('img',img[y:y+h, x:x+w]); cv2.waitKey(0)
            yield '%dpiece%d_%d.png' % (imgno,y,x), img[y:y+h, x:x+w]

if __name__ == '__main__':    
    def path2path_img(path): 
        return (path, cv2.imread(path))
    crop_size = int(sys.argv[1])
    imgs_dir = sys.argv[2]
    pieces_dir = sys.argv[3]

    utils.safe_copytree(imgs_dir,pieces_dir,
                        ['*.jpg', '*.jpeg', '*.png'])  

    timer = utils.ElapsedTimer('Total Cutting')
    #-------------------------------------------------------------
    pieces \
    = pipe(utils.file_paths,
           cmap(path2path_img),
           cfilter(lambda path_img:path_img[1] is not None),
           cmap(lambda pair: path_img2path_pieces(pair,crop_size,imgs_dir,pieces_dir)),
           flatten)(imgs_dir)

    for path,img in pieces:
        #print(path)
        #cv2.imwrite(path, img)
        gray_img = img[:,:,2] # red mask!
        cv2.imwrite(os.path.join(pieces_dir,path), gray_img)
        pass
    #-------------------------------------------------------------
    timer.elapsed_time()


import unittest
class hw2start_yxsTest(unittest.TestCase):
    def test_mod0cases(self):
        self.assertEqual(list(hw2start_yxs((0,0),(100,60),(20,20))),
                         [( 0,0),( 0,20),( 0,40),
                          (20,0),(20,20),(20,40),
                          (40,0),(40,20),(40,40),
                          (60,0),(60,20),(60,40),
                          (80,0),(80,20),(80,40),])
    def test_not_mod0cases(self):
        self.assertEqual(list(hw2start_yxs((0,0),(90,49),(20,20))),
                         [( 0,0),( 0,20),( 0,40),
                          (20,0),(20,20),(20,40),
                          (40,0),(40,20),(40,40),
                          (60,0),(60,20),(60,40),
                          (80,0),(80,20),(80,40),])
    def test_not_excess(self):
        self.assertEqual(list(hw2not_excess_start_yxs(( 0, 0),
                                                      (91,49),
                                                      (20,20))),
                         [( 0,0),( 0,20),
                          (20,0),(20,20),
                          (40,0),(40,20),
                          (60,0),(60,20),])
    def test_origin_specified(self):
        self.assertEqual(list(hw2start_yxs((40,20),
                                           (90,49), 
                                           (20,20))),
                         [(40,20),(40,40),
                          (60,20),(60,40),
                          (80,20),(80,40),])

        self.assertEqual(list(hw2start_yxs((30,10),
                                           (90,49), 
                                           (20,20))),
                         [(30,10),(30,30), 
                          (50,10),(50,30), 
                          (70,10),(70,30)])
        self.assertEqual(list(hw2not_excess_start_yxs((40,20),
                                                      (91,49),
                                                      (20,20))),
                         [(40,20),(60,20)])
