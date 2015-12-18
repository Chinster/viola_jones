#Viola Jones
Performs Viola-Jones algorithm for facial detection.

Currently is a bit messy, in terms of code organization.

Run using:
cargo run POSITIVE_SET NEGATIVE_SET \[POSITIVE_TRAINING\] \[NEGATIVE_TRAINING\]

Datasets should be in should be in 24x24 Grayscale image format. Test sets are optional but will use half the training set if left out.
