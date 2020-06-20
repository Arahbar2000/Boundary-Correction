""" Given a directory of predicted classes of nuclei segmentations, this script generates output masks and 
displays the masks as overlays on top of raw images (i.e. tissue images). 
"""

import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import shared_resources as shres
import predictors as pred


def show_overlays(resources):

    # Read target images (raw image and its class image)
    raw_img = skimage.io.imread(resources.image_file)
    class_img = skimage.io.imread(resources.class_file)

    pred_labels = pred.Predictor(resources).predict()

    # Created predicted images (labels and binary mask)
    # pred_labels = shres.transform_classes_to_labels(class_img)
    bin_pred_labels = np.where(pred_labels > 0, 1, 0)

    fig = plt.figure(figsize=(12, 6))

    a11 = fig.add_subplot(1, 4, 1)
    plt.imshow(raw_img)
    a11.set_title("original img")
    plt.axis("off")

    a12 = fig.add_subplot(1, 4, 2)
    plt.imshow(class_img)
    a12.set_title("classes")
    plt.axis("off")

    a13 = fig.add_subplot(1, 4, 3)
    plt.imshow(raw_img)
    a13.set_title("overlay labels")
    plt.axis("off")
    plt.imshow(pred_labels, alpha=0.5)

    a14 = fig.add_subplot(1, 4, 4)
    plt.imshow(raw_img)
    a14.set_title("overlay masks")
    plt.axis("off")
    plt.imshow(bin_pred_labels, alpha=0.5)

    plt.show()


if __name__ == "__main__":
    show_overlays(shres.parse_arguments())
