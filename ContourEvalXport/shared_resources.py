""" Resources shared by other scripts in this package"""

import os
import argparse
import skimage.morphology
import skimage.io
import numpy as np

CLASS_DIR_NAME = "Classes"
ANNOT_DIR_NAME = "Annotations"
IMAGES_DIR_NAME = "Images"
DEFAULT_PREDICTOR = "baseline"


class Resources:
    """ Stores directory, file locations (paths) and other resources necessary to run contour evaluations. """
    def __init__(self, root_dir, image_index=None, predictor_name='baseline', borders=0, connectivity=0, passes=None):
        # Directory locations
        self.root_dir = root_dir
        self.class_dir = os.path.join(self.root_dir, CLASS_DIR_NAME)
        self.annot_dir = os.path.join(self.root_dir, ANNOT_DIR_NAME)
        self.images_dir = os.path.join(self.root_dir, IMAGES_DIR_NAME)
        # File locations
        self.image_name = None
        self.image_file = None
        self.class_file = None
        self.annot_file = None

        # Additional info
        self.index = image_index
        self.predictor = predictor_name
        self.borders = borders
        self.connectivity = connectivity
        self.passes = passes

        assert os.path.isdir(self.root_dir), "Unable to find {}".format(self.root_dir)
        assert os.path.isdir(self.class_dir), "Unable to find {}".format(self.class_dir)
        assert os.path.isdir(self.annot_dir), "Unable to find {}".format(self.annot_dir)

        if image_index is not None:
            self.image_name = "{}_crop.png".format(image_index)

            self.image_file = os.path.join(self.images_dir, self.image_name)
            self.class_file = os.path.join(self.class_dir, self.image_name)
            self.annot_file = os.path.join(self.annot_dir, self.image_name)

            assert os.path.isfile(self.image_file), "Unable to find image at {}".format(self.image_file)
            assert os.path.isfile(self.class_file), "Unable to find class image at {}".format(self.class_file)
            assert os.path.isfile(self.annot_file), "Unable to find annotation image at {}".format(self.annot_file)





def parse_arguments():
    """ Parse command line arguments. Returns an object with resource locations and the index to the target image """
    describe = "Separate a class png into its individual class probabilities and "
    describe += "compare it with the ground truth. "
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group("reuired arguments")

    help_r = "Root directory with {} and {} subdirectories".format(CLASS_DIR_NAME, ANNOT_DIR_NAME)
    required.add_argument("-r", "--root_dir", help=help_r, type=str, required=True)

    help_i = "ID of image (the number that appears in its name)"
    required.add_argument("-i", "--image_id", help=help_i, type=int, required=True)

    help_p = "Predictor name. There is a default predictor invoked if this argument is not used"
    parser.add_argument("-p", "--predictor", help=help_p, type=str, default=None)

    help_b = "Whether to separate different borders. Type 0 to separate, and 1 to join. Default is to separate"
    parser.add_argument("-b", "--borders", help=help_b, type=int, default=0, required=False)

    help_c = "Whether to process image using pixel edges or vertices. 0 is vertices, 1 is edges." \
             "Default is to separate."
    parser.add_argument("-c", "--connectivity", help=help_c, type=int, default=0, required=False)

    help_pa = "Counts how many passes the pixel_method does."
    parser.add_argument("-pa", "--passes", help=help_pa, type=int, default=None, required=False)
    args = parser.parse_args()

    if args.predictor is None:
        resources_obj = Resources(root_dir=args.root_dir, image_index=args.image_id)
    else:
        resources_obj = Resources(root_dir=args.root_dir, image_index=args.image_id,
                                  predictor_name=args.predictor, borders=args.borders,
                                  connectivity=args.connectivity, passes=args.passes)

    return resources_obj
