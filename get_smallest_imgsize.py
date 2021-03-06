import os, sys, cv2
import utils
from fp import pipe, cmap, cfilter, flatten

img_dir = sys.argv[1]
f = \
pipe(utils.file_paths,
     cmap(lambda path: cv2.imread(path,0)),
     cfilter(lambda img: img is not None),
     cmap(lambda img: img.shape),
     flatten,
     sorted,
     list)
     
sizes = f(img_dir)
print(sizes)
print('smallest img size =',sizes[0])
