import os, sys
import cv2
import glob
import numpy as np
from utils import make_folder
import tqdm
from multiprocessing import Pool

#list1
#label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
#list2 
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

assert len(sys.argv) == 2, "usage: python g_mask.py $CelebAMaskHQ_ROOT"
root = sys.argv[1]

np.savetxt(os.path.join(root, 'labelmap.txt'), ['bg'] + label_list, '%s')
folder_base = os.path.join(root, 'CelebAMask-HQ-mask-anno')
folder_save = os.path.join(root, 'CelebAMask-HQ-mask')
img_num = 30000

make_folder(folder_save)


def process_one_img(k):
    folder_num = k // 2000
    im_base = np.zeros((512, 512))
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if (os.path.exists(filename)):
            # print (label, idx+1)
            im = cv2.imread(filename)
            im = im[:, :, 0]
            im_base[im != 0] = (idx + 1)

    filename_save = os.path.join(folder_save, str(k) + '.png')
    # print (filename_save)
    cv2.imwrite(filename_save, im_base)

# for k in tqdm.trange(img_num):
with Pool(processes=os.cpu_count()//2) as pool:
    for i in tqdm.tqdm(pool.imap_unordered(process_one_img, range(img_num)), total=img_num):
        pass

    # folder_num = k // 2000
    # im_base = np.zeros((512, 512))
    # for idx, label in enumerate(label_list):
        # filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        # if (os.path.exists(filename)):
            # # print (label, idx+1)
            # im = cv2.imread(filename)
            # im = im[:, :, 0]
            # im_base[im != 0] = (idx + 1)

    # filename_save = os.path.join(folder_save, str(k) + '.png')
    # # print (filename_save)
    # cv2.imwrite(filename_save, im_base)


