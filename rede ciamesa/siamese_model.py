from math import log

from keras.utils import plot_model
from keras.utils import set_random_seed

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.optimizers import RMSprop

from utils.siamese import euclidean_distance, eucl_dist_output_shape, contrastive_loss_with_margin

# https://www.kaggle.com/code/peremartramanonellas/how-to-create-a-siamese-network-to-compare-images


def create_model(
        img_shape,
        num_classes_list,
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
    print(f"kernel size: {kernel_size}")
    max_layers = int(log(img_shape[0] - (kernel_size - 1), 2)) -1
    # print("max layers: ", max_layers, img_shape)
    if layers > max_layers: layers = max_layers

    inputs = Input(shape=img_shape)
    x = inputs

    for i in range(layers):
        x = Conv2D(filters=initial_filters, kernel_size=(kernel_size, kernel_size), activation="relu")(x)
        x = MaxPooling2D(pool_size=2)(x)
        initial_filters *= 2

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)

    model = Model(inputs=inputs, outputs=x)

    if display_model:
        # resumo legível da arquitetura deste modelo
        print(model.summary())

        # figura da arquitetura deste modelo
        plot_model(model, 'model_base.png', show_shapes=True)

    return model


def siamese(
        img_shape,
        num_classes_list,
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

    base_model = create_model(
        img_shape=img_shape,
        num_classes_list=num_classes_list,
        initial_filters=initial_filters,
        kernel_size=kernel_size,
        layers=layers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss=loss,
        metrics=metrics,
        seed=seed,
        display_model=display_model,
    )


    #Input for the left part of the pair. We are going to pass training_pairs[:,0] to his layer.
    input_l = Input(shape=img_shape, name='left_input')
    #ATENTION!!! base_model is not an function, is model and we are adding our input layer.
    vect_output_l = base_model(input_l)

    #Input layer for the right part of the siamse model. Will receive: training_pairs[:,1]
    input_r = Input(shape=img_shape, name='right_input')
    vect_output_r = base_model(input_r)

    output = Lambda(
        euclidean_distance,
        name='output_layer',
        output_shape=eucl_dist_output_shape
    )([vect_output_l, vect_output_r])

    model = Model([input_l, input_r], output)

    plot_model(model, 'model.png', show_shapes=True)

    rms = RMSprop()

    model.compile(loss=contrastive_loss_with_margin(margin=1),
              optimizer=rms)

    return model



if __name__ == "__main__":
    siamese((128, 128, 3), [7, 3], layers=5, display_model=True)

