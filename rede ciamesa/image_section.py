
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.preprocessing.image import load_img, img_to_array

matplotlib.use('TkAgg')

def section_image(image, section_shape):
    sections = []
    section_height = section_shape[0]
    section_width = section_shape[1]
    for h in range(int(image.shape[0] / section_height)):
        sections_w = []
        for w in range(int(image.shape[1] / section_width)):  # width
            h_start = h*section_height
            h_end = (h+1)*section_height
            w_start = w*section_width
            w_end = (w+1)*section_width
            print(h_start, h_end, w_start, w_end)
            sections_w.append(
                image[h_start:h_end, w_start:w_end]
            )
        sections.append(sections_w)

    return np.array(sections)


if __name__ == "__main__":

    data_path = "../amostras"
    filename = "caribbean-brain-coral-diploria-sp-footage-038389970_iconl.webp"
    # filename = "cb_24M0211-85D.jpg"
    img_array = img_to_array(
        load_img(
            os.path.join(data_path, filename),
            color_mode='grayscale',
        )
    )
    # print(img_array.astype(int))
    print(img_array.shape)
    print(type(img_array))

    # plt.imshow(img_array.astype(int))
    # plt.show()

    sections = section_image(img_array, (128, 128))
    print(sections.shape)

    # plt.figure()
    # f, axarr = plt.subplots(len(sections), 1)
    fig = plt.figure(
        # figsize=sections.shape[2:4]
    )

    for i in range(sections.shape[0]):
        for j in range(sections.shape[1]):
            fig.add_subplot(sections.shape[0], sections.shape[1], i*sections.shape[1] + j + 1)
            plt.imshow(sections[i, j])
    plt.show()