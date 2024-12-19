import os
import numpy as np
from itertools import product
from json import dumps, loads, dump, load
from pprint import pprint
from statistics import mean, stdev
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import set_random_seed

from siamese_model import siamese
from utils.sim_folder_count import find_highest_simulation_number
from utils.get_ratio import get_ratio
from utils.get_scale import get_scale
from utils.siamese import create_pairs, compute_accuracy

N_JOBS = 0
if N_JOBS:
    tf.config.threading.set_intra_op_parallelism_threads(N_JOBS)
    tf.config.threading.set_inter_op_parallelism_threads(N_JOBS)


def load_data(data_path):
    images = []
    ratios = []
    scales = []
    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            ratio = get_ratio(filename)
            scale = get_scale(filename)
            img_array = img_to_array(load_img(os.path.join(data_path, filename)))
            if ratio is not None and ratio != 18:
                ratios.append(ratio)
                scales.append(scale)
                images.append(img_array)

    return np.array(images), np.array(ratios), np.array(scales)


if __name__ == "__main__":

    # Carregar os dados
    data_path = os.path.join("..", "simulations", f"simulation_{find_highest_simulation_number('./')}")
    data_path = os.path.join("..", "simulations", "simulation_57")
    images, ratios, scales = load_data(data_path)
    file_name = os.path.join("cnn_results", "test_2.json")
    img_shape = images[0].shape

    # print(os.sep)
    # print(data_path)
    # print(file_name)

    # Converter strings para inteiros
    unique_ratios = np.unique(ratios)
    unique_scales = np.unique(scales)
    num_classes = [len(unique_ratios), len(unique_scales)]
    ratio_to_class = {ratio: i for i, ratio in enumerate(unique_ratios)}
    scale_to_class = {scale: i for i, scale in enumerate(unique_scales)}
    ratio_classes = np.array([ratio_to_class[ratio] for ratio in ratios])
    scale_calsses = np.array([scale_to_class[scale] for scale in scales])
    # print(len(ratio_classes))
    # print(len(images))

    # pairs, labels = create_pairs(X=images, y=ratio_classes)
    # X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.4, random_state=0)
    # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)


    X_train, X_test, y_train, y_test = train_test_split(images, ratio_classes, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    X_train, y_train = create_pairs(X=X_train, y=y_train)
    X_val, y_val = create_pairs(X=X_val, y=y_val)
    X_test, y_test = create_pairs(X=X_test, y=y_test)

    # Garantir que todos os valores em y_train e y_test estejam dentro do intervalo correto
    assert np.all(y_train < num_classes[0]), "Existe um índice fora do intervalo em y_train"
    assert np.all(y_test < num_classes[0]), "Existe um índice fora do intervalo em y_test"
    assert np.all(y_val < num_classes[0]), "Existe um índice fora do intervalo em y_test"

    # Converter rótulos para formato categórico
    # y_train = to_categorical(y_train, num_classes=len(unique_ratios))
    # y_test = to_categorical(y_test, num_classes=len(unique_ratios))
    # y_val = to_categorical(y_val, num_classes=len(unique_ratios))

    print("y_train: ", len(y_train))
    print("y_test: ", len(y_test))
    print("y_val: ", len(y_val))

    data_result_keys = [
        "loss",
        "output_0_accuracy",
        "accuracy_1_data",
        "average_loss",
        "average_accuracy_0",
        "std_accuracy_0",
        "average_accuracy_1",
        "std_accuracy_1",
    ]
    ingnore_params = ["seeds"]

    print("num classes: ", num_classes)
    params = {
        "img_shape": [img_shape],
        "num_classes_list": [num_classes],
        "initial_filters": [8],
        "kernel_size": [3],
        "layers": [5],
        "optimizer": ["Adam"],
        "learning_rate": [0.001],
        "loss": ["categorical_crossentropy"],
        "metrics": [["accuracy"]],
        "early_stopping__use_early_stopping": [True],
        "early_stopping__monitor": ["loss"],
        "early_stopping__mode": ["min"],
        "early_stopping__min_delta":  [0.0001, 0.0002, 0.0005],
        "early_stopping__patience": [5],
        "early_stopping__restore_best_weights": [True],
    }
    samples_for_each_params_combination = 1

    early_stopping_prefix = "early_stopping__"

    keys = params.keys()
    values = params.values()
    params_combinations = [dict(zip(keys, combinacao)) for combinacao in product(*values)]

    try:
        with open(file_name, "r") as file:
            file_data = load(file)
            params_combinations = file_data.copy()
        print("file already exists")
    except Exception as e:
        for i, params_combination in enumerate(params_combinations):
            params_combination["loss"] = []
            params_combination["output_0_accuracy"] = []
            params_combination["accuracy_1_data"] = []
            params_combination["seeds"] = []

            params_combination["average_loss"] = None
            params_combination["average_accuracy_0"] = None
            params_combination["std_accuracy_0"] = None
            params_combination["average_accuracy_1"] = None
            params_combination["std_accuracy_1"] = None

        with open(file_name, "x") as file:
            dump(params_combinations, file, indent=4)
        print("create new file")

    for i, params_combination in enumerate(params_combinations):
        print(f"running param combinations: {i}/{len(params_combination)}")
        print(f"param combinations: {params_combination}")
        if (params_combination["average_loss"] is not None
            and params_combination["average_accuracy"] is not None
            and params_combination["std_accuracy"] is not None
        ):
            print(f"combiantion {i} already done.")
            continue

        loss_data = []
        accuracy_0_data = []
        accuracy_1_data = []
        seeds = []
        for j in range(samples_for_each_params_combination):
            # comando para 'zerar' a biblioteca Keras
            # backend.clear_session()

            # definição de sementes aleatórias
            # np.random.seed(j)
            # tf.random.set_seed(j)

            # set_random_seed(j)
            # tf.random.set_seed(j)
            tf.keras.utils.set_random_seed(j)
            tf.config.experimental.enable_op_determinism()
            # enable_op_determinism()

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

            model = siamese(**model_params)

            callbacks = []
            use_early_stopping = early_stopping_params.pop("use_early_stopping")
            if use_early_stopping == True:
                callbacks.append(EarlyStopping(**early_stopping_params))

            # Treinar o modelo
            history = model.fit(
                [X_train[:,0], X_train[:,1]],
                y_train,
                epochs=100,
                validation_data=([X_val[:,0], X_val[:,1]], y_val),
                callbacks=callbacks,
                verbose=1,
            )
            print("__________________")
            print(history.history)
            print("__________________")

            loss = model.evaluate(x=[X_test[:,0], X_test[:,1]], y=y_test)

            print(f'Loss: {loss}')

            y_predict = model.predict([X_test[:,0], X_test[:,1]])
            print(f"y_predict mean: {np.mean(y_predict)}")
            # for i, j in zip(y_test, y_predict):
            #     print(i, " - ", j, " = ", i-j)

            for cut_threshold in np.arange(0, 1, 0.1):
                accuracy = compute_accuracy(y_true=y_test, y_pred=y_predict, cut_threshold=cut_threshold)
                print(f'cut_threshold: {cut_threshold} - accuracy: {accuracy}')

            fig, axes = plt.subplots()
            axes.set_ylim(bottom=0, top=3)
            for key, value in history.history.items():
                # print(key, " :", len(value))
                plt.plot(value, label=key)
            plt.xlabel('Epoch')
            plt.ylabel('Loss/Accuracy')
            plt.legend()
            plt.savefig("graph.png")
            # plt.show()
            exit()

        params_combination["loss"] = loss_data
        params_combination["output_0_accuracy"] = accuracy_0_data
        params_combination["accuracy_1_data"] = accuracy_1_data
        params_combination["seeds"] = seeds

        params_combination["average_loss"] = mean(loss_data)
        params_combination["average_accuracy_0"] = mean(accuracy_0_data)
        params_combination["std_accuracy_0"] = stdev(accuracy_0_data)
        params_combination["average_accuracy_1"] = mean(accuracy_1_data)
        params_combination["std_accuracy_1"] = stdev(accuracy_1_data)

        # print("loss: ", params_combination["loss"])
        # print("output_0_accuracy: ", params_combination["output_0_accuracy"])
        # print("accuracy_1_data: ", params_combination["accuracy_1_data"])
        # print("seeds: ", params_combination["seeds"])

        # print("average_loss: ", params_combination["average_loss"])
        # print("average_accuracy_0: ", params_combination["average_accuracy_0"])
        # print("std_accuracy_0: ", params_combination["std_accuracy_0"])
        # print("average_accuracy_1: ", params_combination["average_accuracy_1"])
        # print("std_accuracy_1: ", params_combination["std_accuracy_1"])

        with open(file_name, "w") as file:
            dump(params_combinations, file, indent=4)