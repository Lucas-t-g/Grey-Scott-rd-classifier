from math import log

from keras.utils import plot_model

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


def create_model(
        img_shape,
        num_classes,
        initial_filters=16,
        kernel_size=3,
        layers=1,
        optimizer="Adam",
        learning_rate=0.001,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        seed=None,
        display_model=False,
    ):
    max_layers = int(log(img_shape[0] - (kernel_size - 1), 2)) -1
    # print("max layers: ", max_layers, img_shape)
    if layers > max_layers: layers = max_layers

    model = Sequential()

    model.add(Input(shape=img_shape))
    for i in range(layers):
        # print("i: ", i)
        model.add(Conv2D(filters=initial_filters, kernel_size=(kernel_size , kernel_size ), activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        initial_filters *= 2

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation="softmax"))

    if display_model:
        # resumo legível da arquitetura deste modelo
        print(model.summary())

        # figura da arquitetura deste modelo
        # plot_model(model, 'model.png', show_shapes=True)


    optimizer = getattr(optimizers, optimizer)(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return model