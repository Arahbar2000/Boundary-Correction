""" Shows the output of a U-Net designed to classify pixels as background, interior,
 or border. Compares the interior estimates with the annotations"""

import sys
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

import shared_resources as shres
import predictors as pred


def show_all_images(resources):
    class_img = skimage.io.imread(resources.class_file)
    annot_img = skimage.io.imread(resources.annot_file)
    raw_img = skimage.io.imread(resources.image_file)

    binary_annot = np.where(annot_img > 0, 1, 0)

    pred_labels = pred.Predictor(resources).predict()

    bin_pred_labels = np.where(pred_labels > 0, 1, 0)

    error_data = binary_annot - bin_pred_labels

    error_data = np.where(error_data > 0, 1, error_data)
    error_data = np.where(error_data < 0, -1, error_data)

    if len(class_img.shape) != 3:
        print("Images representing classes must have 3 channels")
        print("The selected class image has {} channels".format(len(class_img.shape)))
        sys.exit(1)

    if len(annot_img.shape) != 2:
        print("Images representing annotations must have 1 color channel (i.e. gray-scale")
        print("The selected annotation image has {} channels".format(len(annot_img.shape)))
        sys.exit(1)

    background_img = class_img[:, :, 0]
    interior_img = class_img[:, :, 1]
    border_img = class_img[:, :, 2]

    # show_stats(background_img, "background class:")
    # show_stats(interior_img, "interior class:")
    # show_stats(border_img, "border class: ")

    fig = plt.figure(figsize=(9, 8))

    a11 = fig.add_subplot(3, 3, 1)
    plt.imshow(raw_img)
    a11.set_title("original img")
    plt.axis("off")

    a12 = fig.add_subplot(3, 3, 2)
    plt.imshow(annot_img)
    a12.set_title("annotations")
    plt.axis("off")

    a13 = fig.add_subplot(3, 3, 3)
    plt.imshow(binary_annot)
    a13.set_title("annot mask")
    plt.axis("off")

    a21 = fig.add_subplot(3, 3, 4)
    plt.imshow(background_img, cmap='bone')
    a21.set_title("background")
    plt.axis("off")

    a22 = fig.add_subplot(3, 3, 5)
    plt.imshow(interior_img, cmap='bone')
    a22.set_title("interior")
    plt.axis("off")

    a23 = fig.add_subplot(3, 3, 6)
    plt.imshow(border_img, cmap='bone')
    a23.set_title("border")
    plt.axis("off")

    a31 = fig.add_subplot(3, 3, 7)
    plt.imshow(pred_labels)
    a31.set_title("pred labels")
    plt.axis("off")

    a32 = fig.add_subplot(3, 3, 8)
    plt.imshow(bin_pred_labels)
    a32.set_title("pred mask")
    plt.axis("off")

    a33 = fig.add_subplot(3, 3, 9)
    plt.imshow(error_data, cmap='jet')
    a33.set_title("errors")
    plt.axis("off")

    plt.show()


def show_stats(img_array, name=""):
    print(name + ":")
    print("max: {}, min: {}".format(np.amax(img_array), np.amin(img_array)))


if __name__ == "__main__":
    show_all_images(shres.parse_arguments())
