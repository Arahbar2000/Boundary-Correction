import os
import argparse
import pandas as pd
import shared_resources as shres
import show_eval as eval


def parse_arguments():
    describe = "finds predictor method to compare to the baseline and output file for table"
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group("required arguments")

    help_r = "root directory containing images, classes, and annotations"
    required.add_argument("-r", "--root_dir", help=help_r, type=str, required=False, default="SmallEval")
    help_p = "prediction method to compare"
    required.add_argument("-p", "--predictor", help=help_p, type=str, required=False, default="pixel")

    help_o = "output file for table"
    required.add_argument("-o", "--output_file", help=help_o, type=str, required=True)
    args = parser.parse_args()
    return {"root_dir": args.root_dir,
            "predictor": args.predictor,
            "output_file": args.output_file}


def create_table(user_options):
    jaccard_info = pd.DataFrame(columns=['ID', 'baseline', 'b=0 & c=0', 'b=1 & c=0', 'b=0 & c=1', 'b=1 & c=1'],
                                copy=True)
    class_dir_path = os.path.join(user_options["root_dir"], 'Classes')
    for image in os.listdir(class_dir_path):
        image_id = image[:3]
        resources_baseline = shres.Resources('SmallEval', image_index=image_id)
        resources_b0_c0 = shres.Resources('SmallEval', image_index=image_id, predictor_name=user_options["predictor"])
        resources_b0_c1 = shres.Resources('SmallEval', image_index=image_id, predictor_name=user_options["predictor"],
                                          connectivity=1)
        resources_b1_c0 = shres.Resources('SmallEval', image_index=image_id, predictor_name=user_options["predictor"],
                                          borders=1)
        resources_b1_c1 = shres.Resources('SmallEval', image_index=image_id, predictor_name=user_options["predictor"],
                                          borders=1,
                                          connectivity=1)

        jaccard_baseline = eval.evaluate_jaccard_score(resources_baseline)
        jaccard_b0_c0 = eval.evaluate_jaccard_score(resources_b0_c0)
        jaccard_b0_c1 = eval.evaluate_jaccard_score(resources_b0_c1)
        jaccard_b1_c0 = eval.evaluate_jaccard_score(resources_b1_c0)
        jaccard_b1_c1 = eval.evaluate_jaccard_score(resources_b1_c1)

        jaccard_info = jaccard_info.append({'ID': image_id, 'baseline': jaccard_baseline,
                                            'b=0 & c=0': jaccard_b0_c0, 'b=0 & c=1': jaccard_b0_c1,
                                            'b=1 & c=0': jaccard_b1_c0, 'b=1 & c=1': jaccard_b1_c1},
                                           ignore_index=True)
    output_table(jaccard_info, user_options)


def output_table(table, user_options):
    info = 'Jaccard index table that compares the baseline method with the four types of new methods.\nb=0 stands for' \
           ' separating borders, b=1 stands for joining borders.\nc=0 stands for using pixel vertices, c=1 stands for' \
           ' using pixel edges.\n\n'
    f = open(user_options["output_file"], 'w')
    f.write(info)
    f.close()
    f = open(user_options["output_file"], 'a')
    f.write(table.to_string(index=False))
    f.close()


if __name__ == '__main__':
    create_table(parse_arguments())
