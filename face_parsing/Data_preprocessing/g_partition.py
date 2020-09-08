import os, sys
import shutil
import pandas as pd
from shutil import copyfile
from utils import make_folder
import tqdm
import cv2
from multiprocessing import Pool

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

mapping = os.path.join(root, 'CelebA-HQ-to-CelebA-mapping.txt')
image_list = pd.read_csv(mapping, delim_whitespace=True, header=None, dtype=int, skiprows=1, usecols=[0, 1])

def resize_and_write(idx, d, count):
    in_fn = os.path.join(s_img, str(idx)+'.jpg')
    this_image = cv2.imread(in_fn)
    out_fn = os.path.join(d, str(count)+'.jpg')
    this_image_resize = cv2.resize(this_image, LB_SIZE, interpolation=cv2.INTER_LINEAR)
    rt = cv2.imwrite(out_fn, this_image_resize)
    if not rt:
        raise ValueError('Insuccessful image write at: {}'.format(save_path))
    # print('Origin: {}, Saving to: {}'.format(in_fn, out_fn))



# val_set = [(idx, x) for (idx,x) in image_list.iterrows() if 162771<=x<182638]
val_idx = [(d_val, idx, x) for (idx,x) in zip(image_list[0], image_list[1]) if 162771<=x<182638]
test_idx = [(d_test, idx, x) for (idx,x) in zip(image_list[0], image_list[1]) if x>=182638]
train_idx = [(d_train, idx, x) for (idx,x) in zip(image_list[0], image_list[1]) if x<162771]
val_idx = [(i, d) for i, d in enumerate(val_idx)]
test_idx = [(i, d) for i, d in enumerate(test_idx)]
train_idx = [(i, d) for i, d in enumerate(train_idx)]
assert len(val_idx) + len(test_idx) + len(train_idx) == 30000
assert len(val_idx) == 2993
assert len(test_idx) == 2824
assert len(train_idx) == 24183



def process(item):
    count, (d, idx, x) = item
    copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d, str(count)+'.png'))
    resize_and_write(idx, d, count)

with Pool(processes=os.cpu_count()//2) as pool:
    for i in tqdm.tqdm(pool.imap_unordered(process, val_idx), 'val',total=len(val_idx)):
        pass
    for i in tqdm.tqdm(pool.imap_unordered(process, test_idx), 'test', total=len(test_idx)):
        pass
    for i in tqdm.tqdm(pool.imap_unordered(process, train_idx), 'train', total=len(train_idx)):
        pass
 
# def process_one_line(idx, x):
    # x = int(x)
    # ret = 0 
    # if x >= 162771 and x < 182638:
        # copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_val, str(val_count)+'.png'))
        # # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_val, str(val_count)+'.jpg'))        
        # resize_and_write(idx, d_val, val_count)
        # # val_count += 1

    # elif x >= 182638:
        # copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_test, str(test_count)+'.png'))
        # # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_test, str(test_count)+'.jpg'))
        # resize_and_write(idx, d_test, test_count)
        # # test_count += 1 
        # ret = 1
    # else:
        # copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_train, str(train_count)+'.png'))
        # # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_train, str(train_count)+'.jpg'))
        # resize_and_write(idx, d_train, train_count)
        # # train_count += 1  
        # ret = 2
    # return ret


# with Pool(processes=os.cpu_count()//4) as pool:
    # for i in tqdm.tqdm(pool.imap_unordered(process_one_line, image_list.loc[:,1]), total=len(image_list)):
        # # pass
        # if ret == 0:
            # val_count += 1

# for idx, x in enumerate(tqdm.tqdm(image_list.loc[:, 1])):
    # # print (idx, x)
    # x = int(x)
    # if x >= 162771 and x < 182638:
        # copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_val, str(val_count)+'.png'))
        # # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_val, str(val_count)+'.jpg'))        
        # resize_and_write(idx, d_val, val_count)
        # val_count += 1

    # elif x >= 182638:
        # copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_test, str(test_count)+'.png'))
        # # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_test, str(test_count)+'.jpg'))
        # resize_and_write(idx, d_test, test_count)
        # test_count += 1 
    # else:
        # copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_train, str(train_count)+'.png'))
        # # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_train, str(train_count)+'.jpg'))
        # resize_and_write(idx, d_train, train_count)
        # train_count += 1  

# print ('train: {}, val: {}, test: {}'.format(train_count, val_count, test_count))
# print (train_count + test_count + val_count)
