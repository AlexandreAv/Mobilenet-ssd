import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from pipeline.data_flow_creator import DataFlowCreator
from model.model import MobileNetV1
import pdb

IMAGE_DIM = (None, 224, 224)
LEARNING_RATE = 0.0001
EPOCH = 8
BATCH_SIZE = 32

model = MobileNetV1(input_shape=IMAGE_DIM)
loss = CategoricalCrossentropy()
optimizer = Adam(learning_rate=LEARNING_RATE)
data_flow = DataFlowCreator(images_train_path=r'C:\dataset\COCO dataset\train2014',
                            images_valid_path=r'C:\dataset\COCO dataset\val2014', batch_size=BATCH_SIZE, image_size=IMAGE_DIM)
data_flow.feed_from_json_files(train_label_path='labels_files/labels train.json',
                               valid_label_path='labels_files/labels_val.json')

metrics_train_loss = tf.metrics.CategoricalCrossentropy()
metrics_train_acc = tf.metrics.Accuracy()
metrics_valid_loss = tf.metrics.CategoricalCrossentropy()
metrics_valid_acc = tf.metrics.Accuracy()


@tf.function
def train(batch_image, batch_labels):
    with tf.GradientTape() as tape:

        batch_pred = model(batch_image)
        loss_value = loss(tf.squeeze(batch_pred, axis=[1, 2]), batch_labels)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    metrics_train_loss.update_state(batch_labels, batch_pred)
    metrics_train_acc.update_state(batch_labels, tf.math.argmax(tf.squeeze(batch_pred, axis=1), axis=1))


@tf.function
def test(batch_image, batch_labels):
    batch_pred = model(batch_image)

    metrics_valid_loss.update_state(batch_labels, batch_pred)
    metrics_valid_acc.update_state(batch_labels, tf.math.argmax(tf.squeeze(batch_pred, axis=1), axis=1))


for epoch in range(EPOCH):
    i = 0
    for batch_data in data_flow.get_valid_set(): # remettre train_set
        # train(batch_data[0], batch_data[1])
        print(i)
        i += 1

    # print('{}; train loss: {}, train accuracy: {}'.format(str(epoch), metrics_train_loss.result(),
    #                                                       metrics_train_acc.result()))
    #
    # for batch_data in data_flow.get_valid_set():
    #   test(batch_data[0], batch_data[1])
    #
    # print('{}; train loss: {}, train accuracy: {}'.format(str(epoch), metrics_valid_loss.result(),
    #                                                       metrics_valid_acc.result()))
