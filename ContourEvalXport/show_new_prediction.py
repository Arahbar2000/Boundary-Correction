""" Shows the output of a U-Net designed to classify pixels as background, interior,
 or border. Compares the interior estimates with the annotations"""

import sys
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

import shared_resources as shres
import predictors as pred


def show_all_images(resources_new, out_path):
    resources_baseline = shres.Resources(root_dir=resources_new.root_dir, image_index=resources_new.index)
    class_img = skimage.io.imread(resources_new.class_file)
    annot_img = skimage.io.imread(resources_new.annot_file)
    raw_img = skimage.io.imread(resources_new.image_file)

    binary_annot = np.where(annot_img > 0, 1, 0)

    pred_labels_new = pred.Predictor(resources_new).predict()

    bin_pred_labels_new = np.where(pred_labels_new > 0, 1, 0)

    error_data_new = binary_annot - bin_pred_labels_new

    error_data_new = np.where(error_data_new > 0, 1, error_data_new)
    error_data_new = np.where(error_data_new < 0, -1, error_data_new)

    pred_labels_baseline = pred.Predictor(resources_baseline).predict()
    bin_pred_labels_baseline = np.where(pred_labels_baseline > 0, 1, 0)

    error_data_baseline = binary_annot - bin_pred_labels_baseline
    error_data_baseline = np.where(error_data_baseline > 0, 1, error_data_baseline)
    error_data_baseline = np.where(error_data_baseline < 0, -1, error_data_baseline)

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

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(12, 9)

    a11 = fig.add_subplot(gs[:3, :3])
    plt.imshow(raw_img)
    a11.set_title("original img")
    plt.axis("off")

    a12 = fig.add_subplot(gs[:3, 3:6])
    plt.imshow(annot_img)
    a12.set_title("annotations")
    plt.axis("off")

    a13 = fig.add_subplot(gs[:3, 6:9])
    plt.imshow(binary_annot)
    a13.set_title("annot mask")
    plt.axis("off")

    a21 = fig.add_subplot(gs[3:6, :3])
    plt.imshow(background_img, cmap='bone')
    a21.set_title("background")
    plt.axis("off")

    a22 = fig.add_subplot(gs[3:6, 3:6])
    plt.imshow(interior_img, cmap='bone')
    a22.set_title("interior")
    plt.axis("off")

    a23 = fig.add_subplot(gs[3:6, 6:9])
    plt.imshow(border_img, cmap='bone')
    a23.set_title("border")
    plt.axis("off")

    a31 = fig.add_subplot(gs[6:9, :3])
    plt.imshow(pred_labels_baseline)
    a31.set_title("pred baseline labels")
    plt.axis("off")

    a32 = fig.add_subplot(gs[6:9, 3:6])
    plt.imshow(bin_pred_labels_baseline)
    a32.set_title("pred baseline mask")
    plt.axis("off")

    a33 = fig.add_subplot(gs[6:9, 6:9])
    plt.imshow(error_data_baseline, cmap='jet')
    a33.set_title("baseline errors")
    plt.axis("off")

    a41 = fig.add_subplot(gs[9:12, :3])
    plt.imshow(pred_labels_new)
    a41.set_title("pred new labels")
    plt.axis("off")

    a42 = fig.add_subplot(gs[9:12, 3:6])
    plt.imshow(bin_pred_labels_new)
    a42.set_title("pred new mask")
    plt.axis("off")

    a43 = fig.add_subplot(gs[9:12, 6:9])
    plt.imshow(error_data_new, cmap='jet')
    a43.set_title("new errors")
    plt.axis("off")

    plt.savefig(out_path)


def show_stats(img_array, name=""):
    print(name + ":")
    print("max: {}, min: {}".format(np.amax(img_array), np.amin(img_array)))


if __name__ == "__main__":
    show_all_images(shres.parse_arguments())
