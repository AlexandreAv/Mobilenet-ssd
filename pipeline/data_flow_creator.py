import json
from random import shuffle
from tensorflow import convert_to_tensor
from tensorflow import float64, int64
from itertools import islice
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pdb


class DataFlowCreator:
    def __init__(self, images_train_path="./", images_valid_path="./", batch_size=32, image_size=None, randomize=False, flipping=False, rotation=False, shearing=False, zooming=False,
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
                self.train_set = json.loads(valid_data.read())
        except IOError:
            print("Le fichier suivant n'a pas pu être ouvert, : " + valid_label_path)

        self.is_fed = True
        self.prepare_data()

    def prepare_data(self):
        if self.randomize:
            self.randomize_data()

        self.init_seq_augmenters()
        self.train_set = self.make_batch(self.train_set, self.batch_size)
        self.valid_set = self.make_batch(self.valid_set, self.batch_size)
        pdb.set_trace()

    def randomize_data(self):
        self.train_set = shuffle(self.train_set)
        self.valid_set = shuffle(self.valid_set)

    def init_seq_augmenters(self):
        n, h, w = self.image_size
        augmenters = [iaa.Resize({"height": h, "width": w})]

        if self.flipping:
            augmenters.append(iaa.Fliplr(1.0))

        if self.rotation:
            augmenters.append(iaa.Affine(rotate=(-45, 45)))

        if self.shearing:
            augmenters.append(iaa.Affine(shear=(-16, 16)))

        if self.zooming:
            augmenters.append(iaa.ScaleY((0.5, 1.5)))

        if self.saturation:
            augmenters.append(iaa.MultiplySaturation((0.5, 1.5)))

        if self.brightness:
            augmenters.append(iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)))

        if self.contrast:
            augmenters.append(iaa.GammaContrast((0.5, 2.0)))

        if self.gaussian_blur:
            augmenters.append(iaa.GaussianBlur(sigma=(0.0, 3.0)))

        self.seq = iaa.Sequential(augmenters)

    def pipeline_augmenters(self, batch_image):
        return self.seq(images=batch_image)

    @staticmethod
    def make_batch(input_list, batch_size):
        x = int(len(input_list) / batch_size)
        length_to_split = [batch_size for i in range(x)]

        iter_train_set = iter(input_list)
        return [list(islice(iter_train_set, elem)) for elem in length_to_split]

    def get_train_set(self):
        try:
            if not self.is_fed:
                raise ValueError

        except ValueError:
            print("La classe n'a pas été rempli")

        else:
            for batch_data in self.train_set:
                yield self.load_image_set(batch_data, self.images_train_path)

    def get_valid_set(self):
        try:
            if not self.is_fed:
                raise ValueError

        except ValueError:
            print("La classe n'a pas été rempli")

        else:
            for batch_data in self.valid_set:
                yield self.load_image_set(batch_data, self.images_valid_path)

    def load_image_set(self, batch_data, path):
        data = []
        for x in batch_data:  # [id, filename, category, bbox]
            id, filename, category, bbox = x.values()
            with Image.open("{}/{}".format(path, filename)) as img:
                img = np.array(img, dtype=np.float32)
                data.append([img, category, bbox])

        data = list(zip(*data))
        data[0] = convert_to_tensor(self.pipeline_augmenters(np.array(data[0])), dtype=float64)
        data[1] = convert_to_tensor(np.array(data[1]), dtype=int64)

        return data




















