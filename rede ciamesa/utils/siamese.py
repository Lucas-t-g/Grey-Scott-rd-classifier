import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import backend as K

def show_img_dataset(X, y=None, nrows = 4, ncols=4, firstimg=100, numimg=4):
    for i in range(numimg):
        sp = plt.subplot(nrows, ncols, i + 1)

        sp.axis('Off')
        plt.imshow(X[firstimg+i], cmap="Greys")
        if (y is not None):
            plt.title(y[firstimg+i])
    plt.show()


def show_pairs(X, y, image):
    sp = plt.subplot(1, 2, 1)
    plt.imshow(X[image][0])
    sp = plt.subplot(1, 2, 2)
    plt.imshow(X[image][1])
    plt.figtext(0.5, 0.01, str(y[image]), wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return (y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss


#The third parameter: min_equals. indicate how many equal pairs, as minimun, we want in the dataset.
#If we just created random pairs the number of equal pairs would be very small.
def create_pairs(X, y, min_equals = 3000):
    pairs = []
    labels = []
    equal_items = 0

    #index with all the positions containing a same value
    # Index[1] all the positions with values equals to 1
    # Index[2] all the positions with values equals to 2
    #.....
    # Index[9] all the positions with values equals to 9
    index = [np.where(y == i)[0] for i in range(10)]

    for n_item in range(len(X)):
        if equal_items < min_equals:
            #Select the number to pair from index containing equal values.
            num_rnd = np.random.randint(len(index[y[n_item]]))
            num_item_pair = index[y[n_item]][num_rnd]

            equal_items += 1
        else:
            #Select any number in the list
            num_item_pair = np.random.randint(len(X))

        #I'm not checking that numbers is different.
        #That's why I calculate the label depending if values are equal.
        labels += [int(y[n_item] == y[num_item_pair])]
        pairs += [[X[n_item], X[num_item_pair]]]

    return np.array(pairs), np.array(labels).astype('float32')


def create_pairs(X, y, min_equals=0.25):
    pairs = []
    labels = []
    equal_items = 0

    min_equals = int(len(X)*min_equals)

    index = [np.where(y == i)[0] for i in np.unique(y)]
    # print(index)

    for n_item in range(len(X)):
        # print(f"n_item: {n_item}")
        if equal_items < min_equals:
            # a = y[n_item]
            # print("a: ", a)
            # b = index[a]
            # print("b: ", b)
            # c = len(b)
            # print("c: ", c)
            # num_random = np.random.randint(c)
            # print("num_random: ", num_random)

            index_random = np.random.choice(index[y[n_item]])
            equal_items += 1
        else:
            index_random = np.random.randint(len(X))

        labels += [int(y[n_item]) == y[index_random]]
        pairs += [[X[n_item], X[index_random]]]

    return np.array(pairs), np.array(labels).astype("float32")


def compute_accuracy(y_true, y_pred, cut_threshold=0.5):
    pred = y_pred.ravel() < cut_threshold
    return np.mean(pred == y_true)


if __name__ == "__main__":
    X, y = create_pairs(X=['b', 'a', 'c', 'd', 'a'], y=[1, 0, 2, 3, 0])
    print(f"X: {X}")
    print(f"y: {y}")
