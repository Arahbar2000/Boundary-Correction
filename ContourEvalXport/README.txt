show_predictions.py :

generates predictions for a single image
options:
    -r : oot directory containing classes, images, and annotations
    -i : Image ID
    -p : method to compare; available methods are 'pixel'
    -b : only applies to the new method; enter 0 to separate borders and 1 to not separate
    -c : only applies to the new method; enter 0 to use vertex connectivity and 1 to use edge connectivity


evaluate_new_method.py :

generates data table comparing the jaccard indices of the new method and the baseline method
options:
    -p : method to compare; available methods are 'pixel'
    -o : output file to store table


show_all_new_predictions.py :

generates predictions for baseline and the new method and stores predictions in new specified directory
options:
    -r : root directory containing classes, images, and annotations
    -o : output directory to store predictions
    -p : method to compare; available methods are 'pixel'
    -b : only applies to the pixel method; enter 0 to separate borders and 1 to not separate
    -c : only applies to the pixel method; enter 0 to use vertex connectivity and 1 to use edge connectivity