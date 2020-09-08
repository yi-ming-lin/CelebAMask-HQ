import os, sys
import shutil
import pandas as pd
from shutil import copyfile
from utils import make_folder
import tqdm
import cv2

assert len(sys.argv) == 2, "usage: python g_partition.py $CelebAMaskHQ_ROOT"
root = sys.argv[1]

LB_SIZE = (512, 512)

#### source data path
s_label = os.path.join(root, 'CelebAMask-HQ-mask')
s_img = os.path.join(root, 'CelebA-HQ-img')
#### destination training data path
d_train = os.path.join(root, 'train')
d_val = os.path.join(root, 'val')
d_test = os.path.join(root, 'test')

#### make folder
make_folder(d_train)
make_folder(d_val)
make_folder(d_test)

#### calculate data counts in destination folder
train_count = 0
test_count = 0
val_count = 0

mapping = os.path.join(root, 'CelebA-HQ-to-CelebA-mapping.txt')
image_list = pd.read_csv(mapping, delim_whitespace=True, header=None).iloc[1:]

def resize_and_write(idx, count):
    in_fn = os.path.join(s_img, str(idx)+'.jpg')
    this_image = cv2.imread(in_fn)
    out_fn = os.path.join(d_val, str(val_count)+'.jpg')
    this_image_resize = cv2.resize(this_image, LB_SIZE, interpolation=cv2.INTER_LINEAR)
    rt = cv2.imwrite(out_fn, this_image_resize)
    if not rt:
        raise ValueError('Insuccessful image write at: {}'.format(save_path))
    # print('Origin: {}, Saving to: {}'.format(in_fn, out_fn))


for idx, x in enumerate(tqdm.tqdm(image_list.loc[:, 1])):
    # print (idx, x)
    x = int(x)
    if x >= 162771 and x < 182638:
        copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_val, str(val_count)+'.png'))
        # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_val, str(val_count)+'.jpg'))        
        resize_and_write(idx, val_count)
        val_count += 1

    elif x >= 182638:
        copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_test, str(test_count)+'.png'))
        # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_test, str(test_count)+'.jpg'))
        resize_and_write(idx, test_count)
        test_count += 1 
    else:
        copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_train, str(train_count)+'.png'))
        # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_train, str(train_count)+'.jpg'))
        resize_and_write(idx, train_count)
        train_count += 1  

print ('train: {}, val: {}, test: {}'.format(train_count, val_count, test_count))
print (train_count + test_count + val_count)
