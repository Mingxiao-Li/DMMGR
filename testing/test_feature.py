import numpy as np
import csv
import sys
import base64
import lmdb, pickle
from tqdm import tqdm
import detectron2

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = "/export/home2/NoCsBack/hci/mingxiao/feats/vg_object.tsv"
outfile = "/export/home2/NoCsBack/hci/mingxiao/feats/vg_object.tsv.lmdb"

def convert_tsv_to_lmdb(folder_path, out_path, max_size = 1099511627776):
    env = lmdb.open(out_path, map_size=max_size)
    txn = env.begin(write=True)
    with open(folder_path, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        img_id_list = []
        for item in tqdm(reader):
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
                # print(item)
            item["boxes"] =  np.frombuffer(base64.b64decode(item["boxes"][1:]),
                                                dtype=np.float64).reshape((item['num_boxes'], -1))

            item["features"] = np.frombuffer(base64.b64decode(item["features"][1:]),
                                          dtype=np.float32).reshape((item['num_boxes'], -1))
            in_data = item
            in_data = pickle.dumps(in_data)
            img_id_list.append(item["image_id"])
            txn.put(key=str(item["image_id"]).encode(), value=in_data)
    txn.put(key="keys".encode(),value=pickle.dumps(img_id_list))
    txn.commit()
    env.close()


if __name__ == '__main__':
    convert_tsv_to_lmdb(infile,outfile)
    # Verify we can read a tsv
     ##in_data = {}
     ##with open(infile, "r") as tsv_in_file:
     ##    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
     ##    for item in reader:
     ##        item['image_id'] = int(item['image_id'])
     ##        item['image_h'] = int(item['image_h'])
     ##        item['image_w'] = int(item['image_w'])
     ##        item['num_boxes'] = int(item['num_boxes'])
     ##        #print(item)
     ##        for field in ['boxes', 'features']:
     ##            item[field] = np.frombuffer(base64.decodebytes(bytes(item[field],"utf-8")),
     ##                  dtype=np.float32).reshape((item['num_boxes'],-1))
     ##        in_data[item['image_id']] = item
     ##        break
     ##print(in_data)