import lmdb
import zipfile
import os
import numpy as np
import torch
from x.common.image_feature_reader import ImageFeaturesH5Reader
feature_path = "/export/home2/NoCsBack/hci/mingxiao/data/datasets/visual_genome/vg_resnext152_faster_rcnn_genome.lmdb"

pre_fea_path = "/export/home2/NoCsBack/hci/mingxiao/feats/BU36"
#pre_fea_path_z = "/cw/liir/NoCsBack/testliir/datasets/KR_VQR/objects36.zip"
def read_from_lmdb(path):
    env = lmdb.open(path, max_readers=1, readonly=True,
                         lock=False, readahead=False, meminit=False)

    with env.begin(write=False) as txn:
        print(txn.get("keys".encode()))

if __name__ =="__main__":
    #reader = ImageFeaturesH5Reader(feature_path)
    #print(reader[1])
    #for _,_, files in os.walk(pre_fea_path):
    #    print(files)
    #feat = zipfile.ZipFile(pre_fea_path_z)
    #t = 0
    #for f in feat.namelist():
        #print(f)
    print(torch.from_numpy(np.load(pre_fea_path+"/9.jpg.npy", encoding='latin1')).shape)
        #t += 1
        #if t== 2:
            #break