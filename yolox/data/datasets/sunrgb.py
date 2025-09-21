import copy
import os
import json
import cv2
import numpy as np
from .datasets_wrapper import CacheDataset, cache_read_img

class SUNRGBDataset(CacheDataset):
    def __init__(
        self,
        data_dir,
        json_file,
        name="SUNRGBD",
        img_size=(640, 640),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        assert data_dir is not None, "data_dir must be specified"
        self.data_dir = data_dir
        self.json_file = json_file
        with open(os.path.join(self.data_dir, name, self.json_file)) as f: self.sunrgb = json.load(f)
        self.ids = list(self.sunrgb.keys())
        self.num_imgs = len(self.ids)
        self.preproc = preproc
        self.name = name
        self.img_size = img_size
        self.annotations = self._load_sun_annotations()
        
        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )
        
    def __len__(self):
        return self.num_imgs

    def _load_sun_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.sunrgb[id_]
        width = im_ann["width"]
        height = im_ann["height"]
        depth = 1024
        annotations = im_ann["objects"]
        objs = []
        for obj in annotations:
            o={"clean_point":[]}
            for poly in obj["polygon"][0]:
                o["clean_point"].append([poly[0],poly[1],poly[2]])
            objs.append(o)
        #num_objs = len(objs)
        edges =np.array([[0,1],[1,2],[2,3],[3,0],
                [4,5],[5,6],[6,7],[7,4],
                [0,4],[1,5],[2,6],[3,7]])
        faces = np.array([[0,1,2,3],
                          [1,5,6,2],
                          [1,0,4,5],
                          [0,3,7,4],
                          [2,6,7,3],
                          [4,7,6,5]])

        res=[]
        for ix, obj in enumerate(objs):
            #cls = self.class_ids.index(obj["category_id"])
            res.append(np.array(obj["clean_point"])[faces]) 
            #res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        for i,_ in enumerate(res): 
            res[i] *= r
            
        res = np.array(res, dtype=object)
        img_info = (height, width)
        resized_info = (int(height * r), int(width * r),int(depth * r))
        
        file_name = im_ann["img_path"]

        return (res, img_info, resized_info, file_name)


    def load_anno(self, index):
        return self.annotations[index][0]
    
    
    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, id_

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
