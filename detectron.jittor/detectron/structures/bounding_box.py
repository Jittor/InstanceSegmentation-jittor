# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import jittor as jt 
import numpy as np

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy",to_jittor=True):
        #bbox = jt.array(bbox).float32()
        # if bbox.ndim != 2:
        #     raise ValueError(
        #         "bbox should have 2 dimensions, got {}".format(bbox.ndim)
        #     )
        # if bbox.size()>0 and bbox.shape[-1] != 4:
        #     raise ValueError(
        #         "last dimension of bbox should have a "
        #         "size of 4, got {}".format(bbox.shape[-1])
        #     )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}
        self.jittor=to_jittor
        if to_jittor:
            self.to_jittor()
    
    def to_jittor(self):
        self.bbox = jt.array(self.bbox).float32()

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        if isinstance(self.bbox,jt.Var):
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
        else:
            xmin, ymin, xmax, ymax = self._split_into_xyxy_numpy()
        if mode == "xyxy":
            if isinstance(self.bbox,jt.Var):
                bbox = jt.contrib.concat((xmin, ymin, xmax, ymax), dim=-1)
            else:
                bbox = np.concatenate((xmin, ymin, xmax, ymax), axis=-1)
            bbox = BoxList(bbox, self.size, mode=mode,to_jittor=self.jittor)
        else:
            TO_REMOVE = 1
            if isinstance(self.bbox,jt.Var):
                bbox = jt.contrib.concat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
               )
            else:
                bbox = np.concatenate(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), axis=-1
               )
            bbox = BoxList(bbox, self.size, mode=mode,to_jittor=self.jittor)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            if self.bbox.shape[0]==1:
                xmin,ymin,xmax,ymax = self.bbox[:,:1],self.bbox[:,1:2],self.bbox[:,2:3],self.bbox[:,3:]
            else:
                xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            if self.bbox.shape[0]==1:
                xmin,ymin,w,h = self.bbox[:,:1],self.bbox[:,1:2],self.bbox[:,2:3],self.bbox[:,3:]
            else:
                xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                    xmin + jt.clamp(w - TO_REMOVE,min_v=0,max_v = 9999999),
                ymin + jt.clamp(h - TO_REMOVE,min_v=0,max_v = 9999999),
            )
        else:
            raise RuntimeError("Should not be here")

    def _split_into_xyxy_numpy(self):
        if self.mode == "xyxy":
            xmin,ymin,xmax,ymax = self.bbox[:,:1],self.bbox[:,1:2],self.bbox[:,2:3],self.bbox[:,3:]
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin,ymin,w,h = self.bbox[:,:1],self.bbox[:,1:2],self.bbox[:,2:3],self.bbox[:,3:]
            return (
                xmin,
                ymin,
                xmin + np.clip(w - TO_REMOVE,a_min=0,a_max = 9999999),
                ymin + np.clip(h - TO_REMOVE,a_min=0,a_max = 9999999),
            
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode,to_jittor=self.jittor)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, jt.Var) and hasattr(v,'resize'):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        if isinstance(self.bbox,jt.Var):
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
        else:
            xmin, ymin, xmax, ymax = self._split_into_xyxy_numpy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        if isinstance(self.bbox,jt.Var):
            scaled_box = jt.contrib.concat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
            )
        else:
            scaled_box = np.concatenate(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), axis=-1
            )
        bbox = BoxList(scaled_box, size, mode="xyxy",to_jittor=self.jittor)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, jt.Var) and hasattr(v,'resize'):
                v = v.resize(size, *args, **kwargs)
            
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = jt.contrib.concat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, jt.Var):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Crops a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = jt.contrib.concat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, jt.Var):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)


    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        if not isinstance(self.bbox,jt.Var):
            self.to_jittor()
        #print(self.bbox)
        if self.bbox.numel()==0:
            return self
        TO_REMOVE = 1
        self.bbox[:, 0] = jt.clamp(self.bbox[:, 0] ,min_v=0, max_v=self.size[0] - TO_REMOVE)
        self.bbox[:, 1]= jt.clamp(self.bbox[:, 1],min_v=0, max_v=self.size[1] - TO_REMOVE)
        self.bbox[:, 2]= jt.clamp(self.bbox[:, 2],min_v=0, max_v=self.size[0] - TO_REMOVE)
        self.bbox[:, 3]= jt.clamp(self.bbox[:, 3],min_v=0, max_v=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = jt.logical_and((box[:, 3] > box[:, 1]),(box[:, 2] > box[:, 0]))
            #print(keep)
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if box.shape[0] == 0:
            return jt.zeros((0,),dtype=str(box.dtype))
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def _split_into_xywh(self):
        if self.mode == "xyxy":
            # TO_REMOVE = 1
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, (xmax - xmin), (ymax - ymin)
        elif self.mode == "xywh":
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return xmin, ymin, w, h

    def max_length(self):
        _, _, w, h = self._split_into_xywh()
        return jt.maximum(w, h).squeeze(1)

    def max_image_size(self):
        return jt.maximum(jt.array([self.size[0]]).float(), jt.array([self.size[1]]).float())

    def prepare_image_size(self):
        num_box = self.bbox.shape[0]
        max_img_size = self.max_image_size()
        tmp = jt.zeros((num_box, 1), dtype='float')
        return (max_img_size + tmp).squeeze(1)

    def image_area(self):
        num_box = self.bbox.shape[0]
        input_area = jt.array([self.size[0] * self.size[1]], dtype='float')
        tmp = jt.zeros((num_box, 1), dtype='float')
        return (input_area + tmp).squeeze(1)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
