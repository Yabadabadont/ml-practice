import tensorflow as tf
mnist = tf.keras.datasets.cifar10

img_x, img_y = 32, 32

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# 50k, 32, 32, 3(rgb)

# reshape the data into a 4d tensor, (sample_number, x image size, y mage size, n channels)
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 3)
input_shape = (img_x, img_y, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation='relu',
        input_shape=input_shape
    ),
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ),
    tf.keras.layers.Flatten(
        input_shape=input_shape
    ),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

loss, acc = model.evaluate(x_test, y_test)

print(loss, acc)
