from typing import List
import numpy as np
import copy
import pickle
import lmdb  # install lmdb by "pip install lmdb"
import base64


class ImageFeaturesH5Reader(object):
    """
    A reader for H5 files containing pre-extracted image features. A typical
    H5 file is expected to have a column named "image_id", and another column
    named "features".

    Example of an H5 file:
    ```
    faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    Parameters
    ----------
    features_h5path : str
        Path to an H5 file containing COCO train / val image features.
    in_memory : bool
        Whether to load the whole H5 file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.
    cls_prediction : bool
        Whether or not to return prediction of each bounding box
    """

    def __init__(
        self,
        features_path: str,
        g_img: bool = False,
        in_memory: bool = False,
        cls_prediction: bool = False,
    ):
        self.features_path = features_path
        self._in_memory = in_memory
        self._cls_prediction = cls_prediction
        self.g_img = g_img

        self.env = lmdb.open(
            self.features_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self._image_ids = pickle.loads(txn.get("keys".encode()))

        self.features = [None] * len(self._image_ids)
        self.num_boxes = [None] * len(self._image_ids)
        self.boxes = [None] * len(self._image_ids)
        self.boxes_ori = [None] * len(self._image_ids)

        if cls_prediction:
            self.cls_prob = [None] * len(self._image_ids)

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id):
        # image_id = str(image_id).encode()
        index = self._image_ids.index(image_id)
        image_id = str(image_id).encode()
        # image_id = str(image_id).encode()
        cls_prob = None
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.features[index] is not None:
                features = self.features[index]
                num_boxes = self.num_boxes[index]
                image_location = self.boxes[index]
                image_location_ori = self.boxes_ori[index]
                if self._cls_prediction:
                    cls_prob = self.cls_prob[index]
            else:
                with self.env.begin(write=False) as txn:
                    item = pickle.loads(txn.get(image_id))
                    image_id = item["image_id"]
                    image_h = int(item["image_h"])
                    image_w = int(item["image_w"])
                    num_boxes = int(item["num_boxes"])
                    # features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(num_boxes,2048)
                    features = item["features"].reshape(-1, 4)
                    boxes = np.frombuffer(
                        base64.b64decode(item["boxes"]), dtype=np.float32
                    ).reshape(num_boxes, 4)

                    if self._cls_prediction:
                        cls_prob = item["cls_prob"].reshape(num_boxes, 1601)
                        # add an extra row at the top for the <IMG> tokens
                        if self.g_img:
                            g_cls_prob = np.zeros(1601, dtype=np.float32)
                            g_cls_prob[0] = 1
                            cls_prob = np.concatenate(
                                [np.expand_dims(g_cls_prob, axis=0), cls_prob], axis=0
                            )

                        self.cls_prob[index] = cls_prob

                    image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
                    image_location[:, :4] = boxes
                    image_location[:, 4] = (
                        (image_location[:, 3] - image_location[:, 1])
                        * (image_location[:, 2] - image_location[:, 0])
                        / (float(image_w) * float(image_h))
                    )

                    image_location_ori = copy.deepcopy(image_location)

                    image_location[:, 0] = image_location[:, 0] / float(image_w)
                    image_location[:, 1] = image_location[:, 1] / float(image_h)
                    image_location[:, 2] = image_location[:, 2] / float(image_w)
                    image_location[:, 3] = image_location[:, 3] / float(image_h)

                    if self.g_img:
                        g_feat = np.sum(features, axis=0) / num_boxes
                        num_boxes = num_boxes + 1

                        features = np.concatenate(
                            [np.expand_dims(g_feat, axis=0), features], axis=0
                        )

                        g_location = np.array([0, 0, 1, 1, 1])
                        image_location = np.concatenate(
                            [np.expand_dims(g_location, axis=0), image_location], axis=0
                        )
                        g_location_ori = np.array(
                            [0, 0, image_w, image_h, image_w * image_h]
                        )
                        image_location_ori = np.concatenate(
                            [
                                np.expand_dims(g_location_ori, axis=0),
                                image_location_ori,
                            ],
                            axis=0,
                        )

                    self.boxes[index] = image_location
                    self.features[index] = features
                    self.boxes_ori[index] = image_location_ori
                    self.num_boxes[index] = num_boxes
        else:
            # Read chunk from file everytime if not loaded in memory.
            with self.env.begin(write=False) as txn:
                item = pickle.loads(txn.get(image_id))
                image_id = item["image_id"]
                image_h = int(item["image_h"])
                image_w = int(item["image_w"])
                num_boxes = int(item["num_boxes"])

                if self._cls_prediction:
                    cls_prob = item["cls_prob"].reshape(num_boxes, 1601)
                    # add an extra row at the top for the <IMG> tokens
                    if self.g_img:
                        g_cls_prob = np.zeros(1601, dtype=np.float32)
                        g_cls_prob[0] = 1
                        cls_prob = np.concatenate(
                            [np.expand_dims(g_cls_prob, axis=0), cls_prob], axis=0
                        )

                # features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(num_boxes, 2048)
                features = item["features"].reshape(-1, 2048)[:36, :]
                # boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(num_boxes, 4)
                boxes = item["boxes"].reshape(-1, 4)[:36, :]

                image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
                image_location[:, :4] = boxes
                image_location[:, 4] = (
                    (image_location[:, 3] - image_location[:, 1])
                    * (image_location[:, 2] - image_location[:, 0])
                    / (float(image_w) * float(image_h))
                )
                image_location_ori = copy.deepcopy(image_location)
                image_location[:, 0] = image_location[:, 0] / float(image_w)  # x1
                image_location[:, 1] = image_location[:, 1] / float(image_h)  # y1
                image_location[:, 2] = image_location[:, 2] / float(image_w)  # x2
                image_location[:, 3] = image_location[:, 3] / float(image_h)  # y2

                if self.g_img:
                    g_feat = np.sum(features, axis=0) / num_boxes
                    num_boxes = num_boxes + 1
                    features = np.concatenate(
                        [np.expand_dims(g_feat, axis=0), features], axis=0
                    )
                    g_location = np.array([0, 0, 1, 1, 1])
                    image_location = np.concatenate(
                        [np.expand_dims(g_location, axis=0), image_location], axis=0
                    )

                    g_location_ori = np.array(
                        [0, 0, image_w, image_h, image_w * image_h]
                    )
                    image_location_ori = np.concatenate(
                        [np.expand_dims(g_location_ori, axis=0), image_location_ori],
                        axis=0,
                    )

        return features, num_boxes, image_location, image_location_ori, cls_prob

    def keys(self) -> List[int]:
        return self._image_ids


if __name__ == "__main__":
    path = "/cw/liir/NoCsBack/testliir/datasets/KR_VQR/img_feats/trainval_resnet101_faster_rcnn_genome_36.lmdb"
    img_reader = ImageFeaturesH5Reader(path)
