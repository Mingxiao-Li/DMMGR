import gzip
import numpy as np
import lmdb
import csv
import base64
import pickle
import sys
from tqdm import tqdm

def write_tsv_to_lmdb(source_path,output_path,field_name, max_size = 1099511627776,):
    """
    :param source_path: list of source file path
    :param output_path: output file name
    :param max_size:
    :return: None
    """
    env = lmdb.open(output_path, map_size=max_size)
    txn = env.begin(write=True)
    for file in source_path:
        with open(file, "r") as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t', fieldnames=field_name)
            data = {}
            for item in tqdm(reader):
                data['image_id'] = int(item['image_id'])
                data['image_h'] = int(item['image_h'])
                data['image_w'] = int(item['image_w'])
                data['num_boxes'] = int(item['num_boxes'])
                data["boxes"] = np.frombuffer(base64.b64decode(item["boxes"][1:]),
                                                dtype=np.float64).reshape((int(item['num_boxes']), -1))
                data["features"] = np.frombuffer(base64.b64decode(item["features"][1:]),
                                                dtype=np.float32).reshape((int(item['num_boxes']), -1))
                #for field in ['boxes', 'features']:
                #    data[field] = np.frombuffer(base64.b64decode(item[field][1:]),
                #                                dtype=np.float32).reshape((int(item['num_boxes']), -1))

                info = pickle.dumps(data)
                txn.put(key=str(item["image_id"]).encode(),value=info)
    txn.commit()
    env.close()

if __name__ == "__main__":
    csv.field_size_limit(sys.maxsize)
    feat_path = ["/export/home2/NoCsBack/hci/mingxiao/feats/vg_object.tsv"]
    write_tsv_to_lmdb(feat_path,"vg_objects.lmdb",field_name=["image_id","image_h","image_w","num_boxes","boxes","features"])

