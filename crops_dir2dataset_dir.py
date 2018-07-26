import os, sys, random, pathlib
from os.path import join as pjoin
import cv2
import utils

utils.help_option('''
crops_dir2dataset_dir
  Convert 'crops_dir' to dataset directory
  It make train,valid and test directory in 'crops_dir'
  make list of pair of image:mask
  shuffle the list and then move image, masks into train/valid/test
  NOTE: the names of the masks are changed to the same as the image.
  
[synopsys]
  python crops_dir2dataset_dir.py crops_dir
ex)
  python crops_dir2dataset_dir.py 35crops
''')


crops_dir = sys.argv[1]
crops_dir = pathlib.Path(crops_dir).parts[0]

all_paths = list(utils.file_paths(crops_dir))
img_paths = sorted(filter(lambda p: '_mask_' not in p, all_paths))
mask_paths = sorted(filter(lambda p: '_mask_' in p, all_paths))

img_mask_pairs = list(zip(img_paths, mask_paths))
random.shuffle(img_mask_pairs)

# Make directory structure
train_img_dir = pjoin(pjoin(crops_dir,'train'),'image')
train_label_dir = pjoin(pjoin(crops_dir,'train'),'label')

valid_img_dir = pjoin(pjoin(crops_dir,'valid'),'image')
valid_label_dir = pjoin(pjoin(crops_dir,'valid'),'label')

test_img_dir = pjoin(pjoin(crops_dir,'test'),'image')
test_label_dir = pjoin(pjoin(crops_dir,'test'),'label')

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)

os.makedirs(valid_img_dir, exist_ok=True)
os.makedirs(valid_label_dir, exist_ok=True)

os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# Get train | valid | test
dataset_size = len(img_mask_pairs)
num_valid = num_test = dataset_size // 10
num_train = dataset_size - (num_valid + num_test)
print('dataset size =', dataset_size)
print('#train =', num_train)
print('#valid =', num_valid)
print(' #test =', num_test)

train_pairs = img_mask_pairs[:num_train]
valid_pairs = img_mask_pairs[num_train:num_train+num_valid]
test_pairs = img_mask_pairs[num_train+num_valid:dataset_size]
#print(len(train_pairs),len(valid_pairs),len(test_pairs))

# Move images and masks
# *NOTE: masks are renamed with the same as the images. 
def move(src_pairs, dst_img_dir, dst_label_dir):
    for img_path, mask_path in src_pairs:
        moved_img_path = utils.make_dstpath(img_path, crops_dir, dst_img_dir)
        moved_mask_path = utils.make_dstpath(img_path, crops_dir, dst_label_dir)
        os.rename(img_path, moved_img_path)
        os.rename(mask_path, moved_mask_path)
move(train_pairs, train_img_dir, train_label_dir)
move(valid_pairs, valid_img_dir, valid_label_dir)
move(test_pairs,  test_img_dir,  test_label_dir)

print('All images & files are moved successfully!')
