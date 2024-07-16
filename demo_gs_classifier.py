import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from sim_folder_count import find_highest_simulation_number
from get_ratio import get_ratio


def load_data(path):
    images = []
    ratios = []
    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            ratio = get_ratio(filename)
            img_array = img_to_array(load_img(os.path.join(path, filename)))
            if ratio is not None and ratio != 18:
                ratios.append(ratio)
                images.append(img_array)

    return np.array(images), np.array(ratios)


if __name__ == "__main__":

    # Carregar os dados
    data_path = f"simulation_{find_highest_simulation_number("./")}"
    data_path = "simulation_57"
    images, ratios = load_data(data_path)
    img_shape = images[0].shape

    print("shape: ", img_shape)

    # Converter strings para inteiros
    unique_ratios = np.unique(ratios)
    num_classes = len(unique_ratios)
    ratio_to_class = {ratio: i for i, ratio in enumerate(unique_ratios)}
    classes = np.array([ratio_to_class[ratio] for ratio in ratios])

    # print("num_classes: ", num_classes)
    # print("ratio_to_class: ", ratio_to_class)
    # print("ratios: ", ratios)
    # print("classes: ", classes)

    # Dividir em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(images, classes, test_size=0.3)
    print("y_train: ", len(y_train))
    print("y_test: ", len(y_test))

    # Garantir que todos os valores em y_train e y_test estejam dentro do intervalo correto
    assert np.all(y_train < num_classes), "Existe um índice fora do intervalo em y_train"
    assert np.all(y_test < num_classes), "Existe um índice fora do intervalo em y_test"

    # Converter rótulos para formato categórico
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # print(y_train)
    # print(y_test)

    # Construir o modelo
    initial_filters = 16
    kernel_size=3
    model = Sequential([
        Input(shape=img_shape),
        Conv2D(filters=initial_filters, kernel_size=(kernel_size , kernel_size ), activation="relu"),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=initial_filters*2, kernel_size=(kernel_size , kernel_size ), activation="relu"),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=initial_filters*4, kernel_size=(kernel_size , kernel_size ), activation="relu"),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=initial_filters*8, kernel_size=(kernel_size , kernel_size ), activation='relu'),
        MaxPooling2D(pool_size=2),
        # Conv2D(filters=initial_filters*16, kernel_size=(kernel_size , kernel_size ), activation='relu'),
        # MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation="softmax")  # Uma única saída para regressão
    ])

    # resumo legível da arquitetura deste modelo
    print(model.summary())

    # figura da arquitetura deste modelo
    keras.utils.plot_model(model, 'model.png', show_shapes=True)

    # Compilar o modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss', # Monitorar a perda no conjunto de validação
        # monitor='acurracy', # Monitorar a perda no conjunto de validação
        # mode="min",
        # min_delta=0.001,
        patience=10,        # Parar se a perda no conjunto de validação não melhorar por 10 épocas consecutivas
        # verbose=1,
        restore_best_weights=True # Restaurar os pesos do modelo para os da época com a melhor perda no conjunto de validação
    )

    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )

    # Avaliar o modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # Plotar os resultados
    fig, axes = plt.subplots()
    # plt.axes.set_xlim(left=0, right=40)
    axes.set_ylim(bottom=0, top=3)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()
