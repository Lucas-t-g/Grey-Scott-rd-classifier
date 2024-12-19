import os
import numpy as np
from datetime import datetime
from itertools import product
from json import dumps, loads, dump, load
from pprint import pprint
from statistics import mean, stdev

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import backend as K

from model import create_model
from utils.sim_folder_count import find_highest_simulation_number
from get_ratio import get_ratio

N_JOBS = 0
if N_JOBS:
    tf.config.threading.set_intra_op_parallelism_threads(N_JOBS)
    tf.config.threading.set_inter_op_parallelism_threads(N_JOBS)


def load_data(data_path):
    images = []
    ratios = []
    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            ratio = get_ratio(filename)
            img_array = img_to_array(load_img(os.path.join(data_path, filename)))
            if ratio is not None:
            # and ratio != 18:
                ratios.append(ratio)
                images.append(img_array)

    return np.array(images), np.array(ratios)


# def simulation_is_done(sim_data, file_data):
#     for


if __name__ == "__main__":

    # Carregar os dados
    data_path = f"simulations/simulation_{find_highest_simulation_number("./")}"
    data_path = "simulations/simulation_57"
    images, ratios = load_data(data_path)
    file_name = f"cnn_results/test_2.json"
    img_shape = images[0].shape


    # Converter strings para inteiros
    unique_ratios = np.unique(ratios)
    num_classes = len(unique_ratios)
    ratio_to_class = {ratio: i for i, ratio in enumerate(unique_ratios)}
    classes = np.array([ratio_to_class[ratio] for ratio in ratios])

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


    data_result_keys = ["std_accuracy", "average_accuracy", "average_loss", "accuracy", "loss"]
    ingnore_params = ["seeds"]

    params = {
        "img_shape": [img_shape],
        "num_classes": [num_classes],
        "initial_filters": [2**i for i in range(2, 7)],
        "kernel_size": [3],
        "layers": [3, 4, 5],
        "optimizer": ["Adam", "SGD"],
        "learning_rate": [0.001, 0.01, 0.1],
        "loss": ["categorical_crossentropy"],
        "metrics": [["accuracy"]],
        "early_stopping__use_early_stopping": [True],
        "early_stopping__monitor": ["val_loss"],
        "early_stopping__mode": ["min"],
        "early_stopping__min_delta": [0, 0.001],
        "early_stopping__patience": [1, 5],
        "early_stopping__restore_best_weights": True,
    }
    samples_for_each_params_combination = 10

    early_stopping_prefix = "early_stopping__"

    # # Gerando todas as combinações possíveis
    keys = params.keys()
    values = params.values()
    params_combinations = [dict(zip(keys, combinacao)) for combinacao in product(*values)]
    # pprint(params_combinations)
    try:
        with open(file_name, "r") as file:
            file_data = load(file)
            params_combinations = file_data.copy()
        print("file already exists")
    except:
        for i, params_combination in enumerate(params_combinations):
            params_combination["loss"] = []
            params_combination["accuracy"] = []
            params_combination["seeds"] = []

            params_combination["average_loss"] = None
            params_combination["average_accuracy"] = None
            params_combination["std_accuracy"] = None

        with open(file_name, "x") as file:
            dump(params_combinations, file, indent=4)
        print("create new file")


    for i, params_combination in enumerate(params_combinations):
        print(f"running param combinations: {i}/{len(params_combinations)}")
        if (params_combination["average_loss"] is not None
            and params_combination["average_accuracy"] is not None
            and params_combination["std_accuracy"] is not None
        ):
            print(f"combiantion {i} already done.")
            continue

        loss_data = []
        accuracy_data = []
        seeds = []
        for j in range(samples_for_each_params_combination):
            # comando para 'zerar' a biblioteca Keras
            backend.clear_session()

            # definição de sementes aleatórias
            np.random.seed(j)
            tf.random.set_seed(j)
            seeds.append(j)
            # Selecionando apenas os campos com o prefixo especificado e removendo o prefixo
            early_stopping_params = {
                k[len(early_stopping_prefix):]: v
                for k, v in params_combination.items()
                if k.startswith(early_stopping_prefix)
            }
            model_params = {
                k: v
                for k, v in params_combination.items()
                if not k.startswith(early_stopping_prefix) and k not in data_result_keys and k not in ingnore_params
            }

            model = create_model(**model_params)

            callbacks = []
            use_early_stopping = early_stopping_params.pop("use_early_stopping")
            if use_early_stopping == True:
                callbacks.append(EarlyStopping(**early_stopping_params))

            # Treinar o modelo
            history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0,
            )

            # Avaliar o modelo
            loss, accuracy = model.evaluate(X_test, y_test)
            print(f'Loss: {loss}')
            print(f'Accuracy: {accuracy}')
            loss_data.append(loss)
            accuracy_data.append(accuracy)

        params_combination["loss"] = loss_data
        params_combination["accuracy"] = accuracy_data
        params_combination["seeds"] = seeds

        params_combination["average_loss"] = mean(loss_data)
        params_combination["average_accuracy"] = mean(accuracy_data)
        params_combination["std_accuracy"] = stdev(accuracy_data)

        # with open(f"cnn_results/{datetime.now().isoformat().replace(":", "-")}.json", "x") as file:
        with open(file_name, "w") as file:
            dump(params_combinations, file, indent=4)
