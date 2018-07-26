# make_dataset.sh crop_size src_dir dst_dir
# make_dataset.sh gray3_32masks+3 328 35crops
python img_cutter.py $1 $2  $3
python crops_dir2dataset_dir.py $3
python get_cleanmasks.py $3/train/label
python get_cleanmasks.py $3/valid/label
python get_cleanmasks.py $3/test/label
cp -r $3/test $3/output
