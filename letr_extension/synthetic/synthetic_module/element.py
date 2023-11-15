import numpy as np
from PIL import Image
from abc import ABCMeta, abstractmethod



class AbstractElement:
    """Abstract class that defines the characteristics of a document's element."""

    __metaclass__ = ABCMeta

    label = NotImplemented
    color = NotImplemented
    content_width = NotImplemented
    content_height = NotImplemented
    name = NotImplemented
    pos_x = NotImplemented
    pos_y = NotImplemented

    def __init__(self, width, height, seed=None, **kwargs):
        self.width, self.height = width, height
        self.parameters = kwargs
        self.generate_content()

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def content_size(self):
        return (self.content_width, self.content_height)

    @property
    def position(self):
        return (self.pos_x, self.pos_y)

    @abstractmethod
    def generate_content(self):
        pass

    @abstractmethod
    def to_image(self):
        pass

    def to_image_as_array(self):
        return np.array(self.to_image(), dtype=np.float32) / 255

    @abstractmethod
    def to_label_as_array(self):
        pass

    def to_label_as_img(self):
        arr = self.to_label_as_array()
        res = np.zeros(arr.shape + (3,), dtype=np.uint8)
        res[arr == self.label] = self.color
        return Image.fromarray(res)
