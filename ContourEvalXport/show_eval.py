""" Given a root directory containing images, annotations, and output class probabilities, this
program evaluates performance for a particular target image. """

import numpy as np
import skimage.io
import skimage.morphology
import skimage.segmentation
from prettytable import PrettyTable

import shared_resources as shres
import predictors as pred


def evaluate_jaccard_score(resources):
    ground_truth_path = resources.annot_file
    ground_truth = skimage.io.imread(ground_truth_path)

    if len(ground_truth.shape) == 3:
        ground_truth = ground_truth[:, :, 0]

    ground_truth = skimage.morphology.label(ground_truth)
    prediction = pred.Predictor(resources).predict()

    ground_truth = skimage.segmentation.relabel_sequential(ground_truth)[0]
    prediction = skimage.segmentation.relabel_sequential(prediction)[0]

    iou_array = intersection_over_union(ground_truth, prediction)

    if iou_array.shape[0] > 0:
        jaccard = np.amax(iou_array, axis=0).mean()
    else:
        jaccard = 0.0

    print("-------------------------------------------------------")
    print("Jaccard index: {}".format(round(jaccard, 4)))
    print("-------------------------------------------------------")

    return jaccard


def evaluate_object_scores(resources):
    ground_truth_path = resources.annot_file
    ground_truth = skimage.io.imread(ground_truth_path)

    if len(ground_truth.shape) == 3:
        ground_truth = ground_truth[:, :, 0]

    ground_truth = skimage.morphology.label(ground_truth)
    prediction = pred.Predictor(resources).predict()

    ground_truth = skimage.segmentation.relabel_sequential(ground_truth)[0]
    prediction = skimage.segmentation.relabel_sequential(prediction)[0]

    iou_array = intersection_over_union(ground_truth, prediction)

    # Compute scores at different thresholds
    scores = []
    for t in np.arange(0.5, 0.95, 0.05):
        f1, tp, fp, fn = measures_at(t, iou_array)
        res = {"threshold": t, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        scores.append(res)

    # Create a display table
    table = PrettyTable()
    table.field_names = ["Thres", "F1", "TP", "FP", "FN"]
    for score in scores:
        table.add_row([round(score["threshold"], 2), round(score["f1"], 4), score["tp"], score["fp"], score["fn"]])

    print(table)
    return scores


def intersection_over_union(ground_truth, prediction):
    """ Computes the IoU metric given two images; the ground truth and the predicted image.
    Returns: A numpy array of size MxN where M is the number of nuclei in the ground truth and N the number
    of nuclei in the predicted image. The values of the array represent IoU values on a per object basis.
    """

    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))

    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects, pred_objects))
    intersection = h[0]

    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]

    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]

    # Compute Intersection over Union
    union[union == 0] = 1e-9
    iou_array = intersection / union

    return iou_array


def measures_at(threshold, iou_array):
    matches = iou_array > threshold

    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects

    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))

    true_pos, false_pos, false_neg = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-9)

    return f1, true_pos, false_pos, false_neg


if __name__ == "__main__":
    RESOURCES = shres.parse_arguments()

    evaluate_jaccard_score(RESOURCES)
    evaluate_object_scores(RESOURCES)
