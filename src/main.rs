extern crate image;

mod classifiers;

use classifiers::*;
use image::{DynamicImage, GrayImage, Pixel};
use std::fs::DirEntry;
use std::path::Path;

const SUBWINDOW_W: u32 = 24;
const SUBWINDOW_H: u32 = 24;
const CLASSIFIER_SIZE: usize = 50;

/// Parses program arguments and loads image data
fn arg_parse() -> (Vec<GrayImage>, Vec<GrayImage>, Option<Vec<GrayImage>>, Option<Vec<GrayImage>>) {
    let usage = "viola_jones [POSITIVE] [NEGATIVE] TEST_POS TEST_NEG";

    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() >= 2 && args.len() <= 4, usage);

    let pos_path = Path::new(args.get(0).unwrap());
    let neg_path = Path::new(args.get(1).unwrap());

    let mut pos_tests = None;
    let mut neg_tests = None;
    if args.len() > 2 {
        let pos_test_path = Path::new(args.get(2).unwrap());
        let neg_test_path = Path::new(args.get(3).unwrap());

        pos_tests = Some(grays(read_img_data(&pos_test_path)));
        neg_tests = Some(grays(read_img_data(&neg_test_path)));
    }

    let pos_images = grays(read_img_data(&pos_path));
    let neg_images = grays(read_img_data(&neg_path));

    (pos_images, neg_images, pos_tests, neg_tests)
}

/// Reads directory for image files. Ignores subdirectories
fn read_img_data(dir: &Path) -> Vec<DynamicImage> {
    let (files, dirs): (Vec<DirEntry>, Vec<DirEntry>) =
        std::fs::read_dir(dir).unwrap()
                .map(|entry| entry.unwrap())
                .partition(|entry| entry.file_type().unwrap().is_file());

    // Recurse into subdirectories.
    let subdirs: Vec<Vec<DynamicImage>> =
        dirs.iter().map(|entry| read_img_data(&entry.path())).collect();

    let mut images = files.iter()
                          .map(|entry| image::open(entry.path()).unwrap())
                          .collect::<Vec<DynamicImage>>();

    for mut subdir in subdirs {
        images.append(&mut subdir);
    }

    images
}

/// Takes a vector of images and maps them to their gray image
fn grays(images: Vec<DynamicImage>) -> Vec<GrayImage> {
    images.iter()
          .map(|image| image.to_luma())
          .collect()
}

/// Creates a vector of Images with their tag.
fn generate_test_data(pos_data: Vec<GrayImage>,
                      neg_data: Vec<GrayImage>) -> Vec<(IntegralImage, usize)>
{
    let mut test_data = Vec::new();
    for img in pos_data {
        test_data.push((IntegralImage::new(&img), 1));
    }
    for img in neg_data {
        test_data.push((IntegralImage::new(&img), 0));
    }

    test_data
}

/// Creates as many features as can fit in the windows size.
fn generate_features(width: usize, height: usize) -> Vec<HaarFeature> {
    let mut features = Vec::new();
    let mut hori_twin_feats = HaarFeature::gen_features(width,
                                                        height,
                                                        FeatureType::HorizontalTwin);
    features.append(&mut hori_twin_feats);
    let mut vert_twin_feats = HaarFeature::gen_features(width,
                                                        height,
                                                        FeatureType::VerticalTwin);
    features.append(&mut vert_twin_feats);
    let mut hori_trip_feats = HaarFeature::gen_features(width,
                                                        height,
                                                        FeatureType::HorizontalTriplet);
    features.append(&mut hori_trip_feats);
    let mut hori_trip_feats = HaarFeature::gen_features(width,
                                                        height,
                                                        FeatureType::VerticalTriplet);
    features.append(&mut hori_trip_feats);
    let mut checkered_feats = HaarFeature::gen_features(width,
                                                        height,
                                                        FeatureType::Checkered);
    features.append(&mut checkered_feats);
    features
}

/// Performs AdaBoost to generate a strong classifier
/// The testing data should not be part of this functions parameters, but I
/// was interested in some intermediate classifier testing.
fn adaboost<'a>(features: &'a Vec<HaarFeature>,
                pos_data: &mut Vec<(IntegralImage, f64)>,
                neg_data: &mut Vec<(IntegralImage, f64)>,
                test_data: &Vec<(IntegralImage, usize)>)
                    -> Vec<(&'a HaarFeature, f64, f64, f64)>
{
    let mut strong_classifier = Vec::new();
    for _ in 0..CLASSIFIER_SIZE {
        normalize_weights(pos_data, neg_data);

        let (min_feat, min_err, threshold, polarity) =
            find_best_feature(features, pos_data, neg_data);

        update_weights(min_feat, min_err, threshold, polarity, pos_data, neg_data);
        println!("min_feature {:?}, min_err: {}, threshold {}", min_feat, min_err, threshold);
        strong_classifier.push((min_feat, min_err, threshold, polarity));
        strong_classify(&strong_classifier, test_data);
    }
    strong_classifier
}

/// Finds the feature with least amount of error and returns a reference to it
/// Threshold for classification is initialized to 0 and updated on every
/// iteration of a feature.
fn find_best_feature<'a>(features: &'a Vec<HaarFeature>,
                         pos_data: &Vec<(IntegralImage, f64)>,
                         neg_data: &Vec<(IntegralImage, f64)>)
                             -> (&'a HaarFeature, f64, f64, f64)
{
    let mut min_error = std::f64::MAX;
    let mut min_feat = &features[0];
    let mut threshold = std::f64::MAX;
    let mut polarity = 1.0;

    for feat in features.iter() {
        let  pos_res = pos_data.iter()
                              .map(|&(ref img, weight)| (feat.classify(img), weight))
                              .collect::<Vec<(isize, f64)>>();
        let neg_res = neg_data.iter()
                              .map(|&(ref img, weight)| (feat.classify(img), weight))
                              .collect::<Vec<(isize, f64)>>();

        let (feat_threshold, feat_polarity) = calculate_threshold(&pos_res, &neg_res);
        let err = calculate_error(threshold, polarity, &pos_res, &neg_res);

        if err < min_error {
            min_error = err;
            min_feat = feat;
            threshold = feat_threshold;
            polarity = feat_polarity
        }
    }
    (min_feat, min_error, threshold, polarity)
}

fn strong_classify(strong_classifier: &Vec<(&HaarFeature, f64, f64, f64)>,
                   testing: &Vec<(IntegralImage, usize)>) {
    let mut correct = 0;
    let mut detected = 0.0;
    let mut false_positives = 0;
    let mut total_faces = 0.0;
    // Test positives
    for &(ref img, tag) in testing.iter() {
        if tag == 1 {
            total_faces += 1.0;
        }
        let mut sum = 0.0;
        let mut alpha_sum = 0.0;
        for &(feat, err, threshold, sign) in strong_classifier.iter() {
            let feat_sum = feat.classify(img) as f64;
            let class = if feat_sum * sign < threshold * sign { 1.0 } else { 0.0 };

            let alpha = alpha_constant(beta_constant(err));
            sum += class * alpha;
            alpha_sum += alpha;
        }

        let prediction = if sum >= 0.5 * alpha_sum { 1 } else { 0 };

        if prediction == tag {
            correct += 1;
            if tag == 1 {
                detected += 1.0;
            }
        } else {
            if prediction == 1 && tag == 0 {
                false_positives += 1;
            }
        }
    }

    println!("{} feature classifier got {} correct, {} false positives, detected {} out of {} faces.",
             strong_classifier.len(), correct, false_positives, detected, total_faces);
}

/// Find a threshold and polarity for feature results
/// Iterates through a sorted data set, obtaining the minimal error of all
/// ways of parting the set in two.
fn calculate_threshold(pos_data: &Vec<(isize, f64)>,
                       neg_data: &Vec<(isize, f64)>) -> (f64, f64)
{
    // Group all data and add a tag for which set they were in
    let mut all_data: Vec<(isize, f64, bool)> =
        pos_data.iter().map(|val| (val.0, val.1, true))
                .chain(neg_data.iter().map(|val| (val.0, val.1, false)))
                .collect();

    // Sort by feature values
    all_data.sort_by(|a, b| (a.0).cmp(&b.0));

    // Sum of positive weights, negative weights, positive weights below
    // threshold, and negative weights below threshold
    // pos_below, and neg_below are updated as we search
    let pos_tot = pos_data.iter().fold(0.0, |accum, &(_, w)| accum + w);
    let neg_tot = neg_data.iter().fold(0.0, |accum, &(_, w)| accum + w);
    let mut pos_below = 0.0;
    let mut neg_below = 0.0;

    // init to 0th threshold values
    let mut min_err = {
        let err1 = pos_below + (neg_tot - neg_below);
        let err2 = neg_below + (pos_tot - pos_below);

        if err1 < err2 { err1 } else { err2 }
    };
    let mut threshold_index = 0;

    for i in 1..all_data.len() {
        let &(_, prev_w, prev_tag) = unsafe { all_data.get_unchecked(i - 1) };
        if prev_tag { // positive case
            pos_below += prev_w;
        } else {
            neg_below += prev_w;
        }

        let err1 = pos_below + (neg_tot - neg_below);
        let err2 = neg_below + (pos_tot - pos_below);

        let err = if err1 < err2 { err1 } else { err2 };
        if err < min_err {
            min_err = err;
            threshold_index = i;
        }
    }

    // ideally threshold would be average between first item left and right
    // of threshold. Instead my error calculation calls equals to
    let threshold = all_data.get(threshold_index).unwrap().0 as f64;
    let polarity = if threshold < 0.0 { -1.0 } else { 1.0 };

    (threshold, polarity)
}

/// Given the already classified data, calculates an error based off a
/// threshold and polarity
fn calculate_error(threshold: f64,
                   sign: f64,
                   pos_data: &Vec<(isize, f64)>,
                   neg_data: &Vec<(isize, f64)>) -> f64
{
    let mut error = 0.0;
    for &(val, weight) in pos_data.iter() {
        let val = val as f64;

        let class = if sign * val < sign * threshold { 1 } else { 0 };
        // If incorrect
        if class == 0 {
            error += weight;
        }
    }
    for &(val, weight) in neg_data.iter() {
        let val = val as f64;

        let class = if sign * val < sign * threshold { 1 } else { 0 };
        // If incorrect
        if class == 1 {
            error += weight;
        }
    }
    error
}

/// Updates the  weight values for the image samples
fn update_weights(feat: &HaarFeature,
                  error: f64,
                  threshold: f64,
                  sign: f64,
                  pos_data: &mut Vec<(IntegralImage, f64)>,
                  neg_data: &mut Vec<(IntegralImage, f64)>)
{
    for &mut (ref img, ref mut weight) in pos_data.iter_mut() {
        let val = feat.classify(img) as f64;
        let class = if sign * val >= sign * threshold { 1 } else { 0 };
        // Incorrect classification
        if class == 0 {
            *weight *= beta_constant(error);
        }
    }
    for &mut (ref img, ref mut weight) in neg_data.iter_mut() {
        let val = feat.classify(img) as f64;
        let class = if sign * val > threshold { 1 } else { 0 };
        // Incorrect classification
        if class == 1 {
            *weight *= beta_constant(error);
        }
    }
}

/// Puts data back into the range of 0 to 1
fn normalize_weights(pos_data: &mut Vec<(IntegralImage, f64)>,
                     neg_data: &mut Vec<(IntegralImage, f64)>) {
    let pos_total_weight = pos_data.iter().fold(0., |acc, &(_, w)| acc + w);
    let neg_total_weight = neg_data.iter().fold(0., |acc, &(_, w)| acc + w);
    for &mut (_, ref mut weight) in pos_data.iter_mut() {
        *weight /= pos_total_weight;
    }
    for &mut (_, ref mut weight) in neg_data.iter_mut() {
        *weight /= neg_total_weight;
    }
}

fn beta_constant(epsilon: f64) -> f64 {
    (epsilon) / (1.0 - epsilon)
}

fn alpha_constant(epsilon: f64) -> f64 {
    (1.0 / epsilon).log10()
}

fn main() {
    let (pos_images, neg_images, pos_test, neg_test) = arg_parse();

    // Convert each image to integral image and map to a weight
    let mut pos_data = pos_images.iter()
                                 .map(|img| (IntegralImage::new(&img), 1.0))
                                 .collect::<Vec<(IntegralImage, f64)>>();
    let mut neg_data = neg_images.iter()
                                 .map(|img| (IntegralImage::new(&img), 1.0))
                                 .collect::<Vec<(IntegralImage, f64)>>();

    // If test data exists use it, otherwise use half of the training set
    let test_data = if pos_test.is_some() {
        generate_test_data(pos_test.unwrap(), neg_test.unwrap())
    } else {
        let mut data = Vec::new();
        for _ in 0..pos_data.len() / 2 {
            let item = pos_data.pop().unwrap();
            data.push((item.0, 1));
        }
        for _ in 0..neg_data.len() / 2 {
            let item = neg_data.pop().unwrap();
            data.push((item.0, 0));
        }
        data
    };

    println!("Done loading training. {} pos, {} neg, {} training",
             pos_data.len(), neg_data.len(), test_data.len());

    let features = generate_features(SUBWINDOW_W as usize, SUBWINDOW_H as usize);
    println!("Generated {} features", features.len());

    println!("Performing adaboost");
    let strong_classifier = adaboost(&features, &mut pos_data, &mut neg_data, &test_data);

    println!("Classifier feature dump");
    for &(feat, err, thres, sign) in strong_classifier.iter() {
        println!("{:?} {} {} {}", feat, err, thres, sign);
    }
}
