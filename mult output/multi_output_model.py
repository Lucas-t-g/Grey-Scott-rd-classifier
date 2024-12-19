from math import log

from keras.utils import plot_model
from keras.utils import set_random_seed

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input



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

    print("num classes: ", num_classes_list)
    outputs = [Dense(num_classes, activation="softmax", name=f"output_{i}")(x) for i, num_classes in enumerate(num_classes_list)]
    output_losses = {f"output_{i}": loss for i, num_classes in enumerate(num_classes_list)}
    output_metrics = {f"output_{i}": metrics for i, num_classes in enumerate(num_classes_list)}
    print(f"outputs: {outputs}")
    for i, num_classes in enumerate(num_classes_list):
        print(f"{i} - {num_classes}")
    print(f"losses: {output_losses}")
    print(f"output_metrics: {output_metrics}")

    model = Model(inputs=inputs, outputs=outputs)

    if display_model:
        # resumo legível da arquitetura deste modelo
        print(model.summary())

        # figura da arquitetura deste modelo
        plot_model(model, 'model.png', show_shapes=True)


    optimizer = getattr(optimizers, optimizer)(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=output_losses,
        metrics=output_metrics,
    )
    return model

if __name__ == "__main__":
    create_model((128, 128, 3), [7, 3], layers=5, display_model=True)


# https://www.kaggle.com/code/peremartramanonellas/guide-multiple-outputs-with-keras-functional-api