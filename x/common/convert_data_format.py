import base64
import csv
import lmdb
import pickle
import numpy as np
from typing import List


def convert_csv_to_lmdb(
    in_path: str, out_path: str, field_name: List[str], max_size=1099511627776
) -> None:
    """
    Handle pretrained image feature (bottom-up attention)
    :param in_path: list of source file path
    :param out_path: output file path
    """
    env = lmdb.open(out_path, map_size=max_size)
    txn = env.begin(write=True)
    for file in in_path:
        with open(file, "r") as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter="\t", filed=field_name)
            data = {}
            for item in reader:
                item["image_id"] = int(item["image_id"])
                data["image_h"] = int(item["image_h"])
                data["image_w"] = int(item["image_w"])
                data["num_boxes"] = int(item["num_boxes"])
                for field in ["boxes", "features", "cls_prob", "box_obj_id"]:
                    data[field] = np.frombuffer(
                        base64.b64decode(item[field][1:]), dtype=np.float32
                    ).reshape((data["num_bpxes"], -1))
                info = pickle.dump(data)
                txn.put(key=str(item["image_id"]).encode(), value=info)
    txn.commit()
    env.close()