from model.model import MobileNetV1
from PIL import Image
import numpy as np

model = MobileNetV1((224, 224, 3), classes=2)

with Image.open('../camera1-1.jpg') as img:
    img = img.resize((224, 224))
    img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    model.predict(img)

model.summary()