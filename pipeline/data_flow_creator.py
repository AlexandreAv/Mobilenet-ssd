import json
from random import shuffle
from tensorflow import convert_to_tensor
from tensorflow import float32, float64, int64, int32
from itertools import islice
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from random import uniform
import pdb


class DataFlowCreator:
    def __init__(self, images_train_path="./", images_valid_path="./", batch_size=32, image_size=None,
                 randomize=False, flipping=False, rotation=False, shearing=False, zooming=False,
                 contrast=False, brightness=False, mirror=False, expand=False, gaussian_blur=False, saturation=False):
        self.images_train_path = images_train_path
        self.images_valid_path = images_valid_path
        self.train_set = []  # [id, filename, category, bbox]
        self.valid_set = []
        self.seq = None
        self.encoder = LabelBinarizer()
        self.batch_size = batch_size
        self.image_size = image_size
        self.randomize = randomize
        self.flipping = flipping
        self.rotation = rotation
        self.shearing = shearing
        self.zooming = zooming
        self.contrast = contrast
        self.brightness = brightness
        self.mirror = mirror
        self.expand = expand
        self.gaussian_blur = gaussian_blur
        self.saturation = saturation
        self.is_fed = False

    def feed_from_json_files(self, train_label_path, valid_label_path):
        try:
            with open(train_label_path) as train_data:
                self.train_set = json.loads(train_data.read())
        except IOError:
            print("Le fichier suivant n'a pas pu être ouvert, : " + train_label_path)

        try:
            with open(valid_label_path) as valid_data:
                self.valid_set = json.loads(valid_data.read())
        except IOError:
            print("Le fichier suivant n'a pas pu être ouvert, : " + valid_label_path)

        self.is_fed = True
        self.prepare_data()

    @staticmethod
    def get_label_list(label):
        data = list(zip(*label))
        li = data[2]
        label_list = []

        for category in li:
            if category not in label_list:
                label_list.append(category)

        return label_list

    def prepare_data(self):
        if self.randomize:
            self.randomize_data()

        self.init_seq_augmenters()
        self.train_set = self.dict_to_list(self.train_set)
        self.valid_set = self.dict_to_list(self.valid_set)

        self.encoder.fit(self.get_label_list(self.train_set))

        self.train_set = self.make_batch(self.train_set, self.batch_size)
        self.valid_set = self.make_batch(self.valid_set, self.batch_size)

    def randomize_data(self):
        self.train_set = shuffle(self.train_set)
        self.valid_set = shuffle(self.valid_set)

    @staticmethod
    def dict_to_list(list_of_dict):
        li = []
        for x in list_of_dict:
            li.append(list(x.values()))

        return list(li)

    def init_seq_augmenters(self):
        n, h, w = self.image_size

        rotate_range = 0
        shear_range = 0
        scale_range = 1
        flipping_range = 0
        saturation_range = 1
        brightness_range = 1
        contrast_range = 1
        gaussian_blur_range = 0

        if self.rotation:
            rotate_range = self.rotation

        if self.shearing:
            shear_range = self.shearing

        if self.zooming:
            scale_range = self.zooming

        if self.flipping:
            flipping_range = self.flipping

        if self.saturation:
            saturation_range = uniform(1, self.saturation)

        if self.brightness:
            brightness_range = uniform(1, self.brightness)

        if self.contrast:
            contrast_range = self.contrast

        if self.gaussian_blur:
            gaussian_blur_range = self.gaussian_blur

        self.seq = iaa.Sequential([iaa.Resize({"height": h, "width": w}),
                                   iaa.Affine(rotate=rotate_range, shear=shear_range, scale=scale_range),
                                   iaa.Fliplr(flipping_range),
                                   iaa.imgcorruptlike.Saturate(saturation_range),
                                   iaa.imgcorruptlike.Brightness(brightness_range),
                                   iaa.GammaContrast(contrast_range),
                                   iaa.GaussianBlur(gaussian_blur_range)])

    def pipeline_augmenters(self, batch_image):
        print(batch_image.shape)

        return self.seq(images=batch_image)

    @staticmethod
    def make_batch(input_list, batch_size):
        x = int(len(input_list) / batch_size)
        length_to_split = [batch_size for i in range(x)]

        iter_train_set = iter(input_list)
        li = [list(islice(iter_train_set, elem)) for elem in length_to_split]

        for i in range(len(li)):
            li[i] = list(zip(*li[i]))

        return li

    def get_train_set(self):
        try:
            if not self.is_fed:
                raise ValueError

        except ValueError:
            print("La classe n'a pas été rempli")

        else:
            return DataSetIterator(self.train_set, self.images_train_path, self.seq, self.encoder)

    def get_valid_set(self):
        try:
            if not self.is_fed:
                raise ValueError

        except ValueError:
            print("La classe n'a pas été rempli")

        else:
            return DataSetIterator(self.valid_set, self.images_valid_path, self.seq, self.encoder)


class DataSetIterator:
    def __init__(self, data_set, images_path, augmenters, encoder):
        self.data_set = data_set
        self.images_path = images_path
        self.augmenters = augmenters
        self.encoder = encoder

    def __iter__(self):
        for batch_data in self.data_set:
            yield self.load_image_set(batch_data, self.images_path)

    def load_image_set(self, batch_data, path):
        data = []
        list_bbox = []

        for i in range(len(batch_data[0])):  # [id, filename, category, bbox]
            filename = batch_data[1][i]
            category = batch_data[2][i]
            bbox = batch_data[3][i]
            with Image.open("{}/{}".format(path, filename)) as img:
                img = np.array(img)

                if len(img.shape) == 2:
                    dim = np.zeros((img.shape[0], img.shape[1]))
                    img = np.stack((img, dim, dim), axis=2)
                    img = img.astype(np.uint8)

                list_bbox.append(BoundingBoxesOnImage([BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])], img.shape)) # TODO amener la gestion de plusieurs boudings boxe par images

                data.append([img, category, bbox])

        data = list(zip(*data))
        pdb.set_trace()
        data[0] = convert_to_tensor((self.augmenters.augment_images(data[0])), dtype=float32)
        data[2] = convert_to_tensor(self.augmenters.augment_bounding_boxes(data[2]), dtype=int32)
        data[1] = convert_to_tensor(self.encoder.transform(np.array(data[1])), dtype=int32)


        return data

    # @staticmethod
    # def regroup_list_by_indices(list_x):
    #     x = []
    #     y = []
    #     z = []
    #
    #     for element in list_x:
    #         x.append(element[0])
    #         y.append(element[1])
    #         z.append(element[2])
    #
    #     return [x, y, z]

