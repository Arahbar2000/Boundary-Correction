import os
import argparse
import show_new_prediction as show
import shared_resources as shres

def parse_arguments():
    describe = "Finds input directory containing classes and output directory for images"
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group("required arguments")

    help_r = "input directory containing image classes, images, and annotations directories"
    required.add_argument("-r", "--root_dir", help=help_r, type=str, required=False, default="SmallEval")

    help_o = "output directory to show predictions for each image"
    required.add_argument("-o", "--output_dir", help=help_o, type=str, required=False, default="predictions")

    help_p = "new method to use"
    required.add_argument("-p", "--predictor", help=help_p, type=str, required=False, default='pixel')

    help_b = "Enter 0 to separate boundaries, 1 otherwise"
    required.add_argument("-b", "--border", help=help_b, type=int, required=False, default=0)

    help_c = "Enter 0 to use edge connectivity, 1 to use vertex connectivity"
    required.add_argument("-c", "--connectivity", help=help_c, type=int, required=False, default=0)

    args= parser.parse_args()

    return {"root_dir": args.root_dir,
            "output_dir": args.output_dir,
            "border": args.border,
            "connectivity": args.connectivity,
            "predictor": args.predictor}

def show_all_predictions(user_options):
    class_dir_path = os.path.join(user_options["root_dir"], 'Classes')
    if not os.path.isdir(user_options["output_dir"]):
        os.makedirs(user_options["output_dir"])
    for image in os.listdir(class_dir_path):
        id = image[:3]
        resources = shres.Resources(root_dir=user_options["root_dir"], image_index=id,
                                    predictor_name=user_options["predictor"],
                                    borders=user_options["border"],
                                    connectivity=user_options["connectivity"])
        output_path = os.path.join(user_options["output_dir"], image)
        show.show_all_images(resources, output_path)

if __name__ == '__main__':
    show_all_predictions(parse_arguments())

