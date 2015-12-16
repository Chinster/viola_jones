extern crate image;

mod classifiers;

use classifiers::*;
use image::{DynamicImage, GrayImage, Pixel};
use std::fs::DirEntry;
use std::path::Path;

const SUBWINDOW_W: u32 = 24;
const SUBWINDOW_H: u32 = 24;
const CLASSIFIER_SIZE: usize = 3;

/// Parses program arguments and loads image data
fn arg_parse() -> (Vec<GrayImage>, Vec<GrayImage>) {
    let usage = "viola_jones [POSITIVE] [NEGATIVE]";

    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() == 2, usage);

    let pos_path = Path::new(args.get(0).unwrap());
    let neg_path = Path::new(args.get(1).unwrap());

    let pos_images = grays(read_img_data(&pos_path));
    let neg_images = grays(read_img_data(&neg_path));

    (pos_images, neg_images)
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
    let mut checkered_feats = HaarFeature::gen_features(width,
                                                        height,
                                                        FeatureType::Checkered);
    features.append(&mut checkered_feats);
    features
}

/// Finds the feature with least amount of error and returns a reference to it
/// Threshold for classification is initialized to 0 and updated on every
/// iteration of a feature.
fn find_best_feature<'a>(features: &'a Vec<HaarFeature>,
                         pos_data: &Vec<(IntegralImage, f64)>,
                         neg_data: &Vec<(IntegralImage, f64)>)
                             -> (&'a HaarFeature, f64, f64)
{
    let mut min_error = std::f64::MAX;
    let mut min_feat = &features[0];
    let mut threshold = std::f64::MAX;

    for feat in features.iter() {
        let pos_res = pos_data.iter()
                              .map(|&(ref img, weight)| (feat.classify(img), weight))
                              .collect::<Vec<(isize, f64)>>();
        let neg_res = neg_data.iter()
                              .map(|&(ref img, weight)| (feat.classify(img), weight))
                              .collect::<Vec<(isize, f64)>>();

        let new_threshold = calculate_threshold(threshold, &pos_res, &neg_res);
        let err = calculate_error(threshold, 1.0, &pos_res, &neg_res);

        if err < min_error {
            min_error = err;
            min_feat = feat;
            threshold = new_threshold;
        }
    }
    (min_feat, min_error, threshold)
}

fn calculate_threshold(threshold: f64,
                       pos_data: &Vec<(isize, f64)>,
                       neg_data: &Vec<(isize, f64)>) -> f64
{
    10.0
    /*
    let (pos_total, pos_thres) = pos_data.iter().fold((0.0, 0.0),
        |(tot, thres), &(val, w)| {
            let new_val = if val.abs() < threshold.abs() {
                thres + w
            } else {
                thres
            };
            (tot + w, new_val)
        });
    let (neg_total, neg_thres) = neg_data.iter().fold((0.0, 0.0),
        |(tot, thres), &(val, w)| {
            let new_val = if val.abs() < threshold.abs() {
                thres + w
            } else {
                thres
            };
            (tot + w, new_val)
        });

    let val1 = pos_thres + (neg_total - neg_thres);
    let val2 = neg_thres + (pos_total - pos_thres);

    if val1 < val2 {
        val1
    } else {
        val2
    }
    */
}

/// Given the already classified data, calculates an error based off a
/// threshold
fn calculate_error(threshold: f64,
                   sign: f64,
                   pos_data: &Vec<(isize, f64)>,
                   neg_data: &Vec<(isize, f64)>) -> f64
{
    let mut error = 0.0;
    let mut correct = 0;
    for &(val, weight) in pos_data.iter() {
        let val = val as f64;

        let class = if sign * val < sign * threshold { 1 } else { 0 };
        // If incorrect
        if class == 0 {
            error += weight;
        }
        else {
            correct += 1;
        }
    }
    for &(val, weight) in neg_data.iter() {
        let val = val as f64;

        let class = if sign * val < sign * threshold { 1 } else { 0 };
        // If incorrect
        if class == 1 {
            error += weight;
        }
        else {
            correct += 1;
        }
    }
    error
}

/// Updates the  weight values for the image samples
fn update_weights(feature: &HaarFeature,
                  error: f64,
                  threshold: f64,
                  pos_data: &mut Vec<(IntegralImage, f64)>,
                  neg_data: &mut Vec<(IntegralImage, f64)>)
{
    for &mut (ref img, ref mut weight) in pos_data.iter_mut() {
        let val = if feature.classify(img).abs() as f64 > threshold { 1 } else { 0 };
        // Correct classification
        if val == 1 {
            *weight *= beta_constant(error);
        }
    }
    for &mut (ref img, ref mut weight) in neg_data.iter_mut() {
        let val = if feature.classify(img).abs() as f64 > threshold { 1 } else { 0 };
        // Correct classification
        if val == 0 {
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
    let (pos_images, neg_images) = arg_parse();

    // Convert each image to integral image and map to a weight
    let mut pos_data = pos_images.iter()
                                 .map(|img| (IntegralImage::new(&img), 1.0))
                                 .collect::<Vec<(IntegralImage, f64)>>();
    let mut neg_data = neg_images.iter()
                                 .map(|img| (IntegralImage::new(&img), 1.0))
                                 .collect::<Vec<(IntegralImage, f64)>>();
    println!("Done loading training. {} pos, {} neg", pos_data.len(), neg_data.len());

    let features = generate_features(SUBWINDOW_W as usize, SUBWINDOW_H as usize);
    println!("Generated {} features", features.len());

    /*
    let mut string_classifier = Vec::new();
    for _ in 0..CLASSIFIER_SIZE {
        normalize_weights(&mut pos_data,  &mut neg_data);

        let (min_feat, min_err, threshold) = find_best_feature(&features, &pos_data, &neg_data);

        update_weights(min_feat, min_err, threshold, &mut pos_data, &mut neg_data);
        println!("min_feature {:?}, min_err: {}, threshold {}", min_feat, min_err, threshold);
        best_feats.push((min_feat, beta_constant(min_err), threshold));

        for &(feat, _, threshold) in best_feats.iter() {
            let mut right = 0.0;
            for &(ref img, _) in pos_data.iter() {
                if (feat.classify(img).abs() as f64) > threshold {
                    right += 1.0;
                }
            }
            for &(ref img, _) in neg_data.iter() {
                if (feat.classify(img).abs() as f64) < threshold {
                    right += 1.0;
                }
            }
            println!("Got {}% correct", right / (pos_data.len() +  neg_data.len()) as f64);
        }
    }
    */
}
