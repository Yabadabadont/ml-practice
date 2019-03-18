import tensorflow as tf
import numpy as np
import keras
from keras import backend as K

from sklearn.model_selection import train_test_split

mnist = tf.keras.datasets.cifar10

img_x, img_y = 32, 32

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

K.set_image_dim_ordering('tf')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_valid = x_valid.astype('float32')

x_train /= 255
x_test /= 255
x_valid /= 255

y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)
y_valid = keras.utils.np_utils.to_categorical(y_valid)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding='same',
        activation='relu',
        input_shape=(32,32,3)
    ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding='same',
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding='same',
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0,2),
    tf.keras.layers.Dense(
        512,
        activation='relu',
        kernel_constraint=keras.constraints.maxnorm(3)
    ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, batch_size=32)

loss, acc = model.evaluate(x_test, y_test)

model.save_weights('weights')

print(acc)
