use image::{GrayImage, Pixel};

#[derive(Debug)]
pub struct IntegralImage {
    width: usize,
    height: usize,
    data: Vec<usize>,
}

impl IntegralImage {
    pub fn new(image: &GrayImage) -> IntegralImage {
        let (width, height) = image.dimensions();
        IntegralImage {
            width: width as usize,
            height: height as usize,
            data: IntegralImage::integral_image(image),
        }
    }

    /*
    /// Returns the (width, height) of this image
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
    */

    /// Unsafe to prevent bounds check
    /// Converted to isize to ease result use
    pub fn get_val(&self, x: usize, y: usize) -> isize {
        unsafe {
            (*self.data.get_unchecked(y * self.width + x)) as isize
        }
    }


    /// Prints to stdout the integral image values
    pub fn print(&self) {
        let mut cnt = 0;
        for datum in self.data.iter() {
            print!("{} ", datum);
            cnt += 1;
            if cnt == self.width {
                cnt = 0;
                println!("");
            }
        }
    }

    ///  Generates an integral image that is the same size as the GrayImage
    ///  given.
    fn integral_image(image: &GrayImage) -> Vec<usize> {
        let (width, height) = image.dimensions();
        let (width, height) = (width as usize, height as usize);
        let mut out: Vec<usize> =
            image.pixels().map(|p| p.channels()[0] as usize).collect();

        // First row
        for i in 1..width {
            out[i] += out[i - 1];
        }

        // First col
        for i in 1..height {
            out[i * width] += out[(i - 1) * width];
        }

        for i in 1..height {
            for j in 1..width {
                let pixel0 = out[(i - 1) * width + (j - 1)]; // upper left
                let pixel1 = out[(i - 1) * width + j];       // upper
                let pixel2 = out[i * width + (j - 1)];       // left

                out[i * width + j] += pixel2 + pixel1 - pixel0;
            }
        }

        out
    }
}

#[derive(Debug, Copy, Clone)]
pub enum FeatureType {
    HorizontalTwin,
    VerticalTwin,
    HorizontalTriplet,
    VerticalTriplet,
    Checkered,
}

#[derive(Debug)]
pub struct HaarFeature {
    x: usize,
    y: usize,
    scalex: usize,
    scaley: usize,
    feat: FeatureType,
}

impl HaarFeature {
    /// Create a new Haar Feature where (x, y) are the features location in an image
    /// scalex and scaley is this features size
    pub fn new(x: usize, y: usize,
               scalex: usize, scaley: usize,
               feat: FeatureType) -> HaarFeature {
        HaarFeature {
            x: x,
            y: y,
            scalex: scalex,
            scaley: scaley,
            feat: feat,
        }
    }

    /// Returns a list of HaarFeatures of the given type within the given range
    /// These features would be various size and positions of a feature
    pub fn gen_features(max_width: usize,
                        max_height: usize,
                        feat: FeatureType) -> Vec<HaarFeature> {
        let (feat_width, feat_height) = HaarFeature::feature_dimensions(feat);

        let mut features = Vec::new();
        for y in 0..max_height {
            for x in 0..max_width {
                // Create all feature sizes that fit at location x, y
                //
                // for horizontal_twin_features height scales by 1 pixel
                // and width doubles each time. Leave a 1 pixel border because
                // my integral image does not have data there.
                let mut scaley = 1;
                while feat_height * scaley < max_height - y {
                    let mut scalex = 1;
                    while scalex * feat_width < max_width - x {
                        features.push(
                            HaarFeature::new(x, y, scalex, scaley, feat)
                        );
                        scalex += 1;
                    }
                    scaley += 1;
                }
            }
        }

        features
    }

    /// Return (width, height) of the given feature type
    pub fn feature_dimensions(feat: FeatureType) -> (usize, usize) {
        match feat {
            FeatureType::HorizontalTwin => (2, 1),
            FeatureType::VerticalTwin => (1, 2),
            FeatureType::HorizontalTriplet => (3, 1),
            FeatureType::VerticalTriplet => (1, 3),
            FeatureType::Checkered => (2, 2),
        }
    }

    /// Returns the region encompassed by the feature.
    /// Black rectangles are subtracted from white rectangles
    pub fn classify(&self, image: &IntegralImage) -> isize {
        let (x, y) = (self.x, self.y);
        let (width, height) = HaarFeature::feature_dimensions(self.feat);

        match self.feat {
            FeatureType::HorizontalTwin => {
                // Points from upper left to bottom right
                let p1 = image.get_val(x, y);
                let p2 = image.get_val(x + width / 2, y);
                let p3 = image.get_val(x + width, y);
                let p4 = image.get_val(x, y + height);
                let p5 = image.get_val(x + width / 2, y + height);
                let p6 = image.get_val(x + width, y + height);

                let left_rect = p5 - p2 - p4 + p1;
                let right_rect = p6 - p5 - p3 + p2;
                right_rect - left_rect
            },
            FeatureType::VerticalTwin => {
                // Points from upper left to bottom right
                let p1 = image.get_val(x, y);
                let p2 = image.get_val(x + width, y);
                let p3 = image.get_val(x, y + height / 2);
                let p4 = image.get_val(x + width, y + height / 2);
                let p5 = image.get_val(x, y + height);
                let p6 = image.get_val(x + width, y + height);

                let top_rect = p4 - p3 - p2 + p1;
                let bot_rect = p6 - p5 - p4 + p3;
                top_rect - bot_rect
            },
            FeatureType::HorizontalTriplet => {
                // Points from upper left to bottom right
                let p1 = image.get_val(x, y);
                let p2 = image.get_val(x + width / 3, y);
                let p3 = image.get_val(x + width * 2 / 3, y);
                let p4 = image.get_val(x + width, y);
                let p5 = image.get_val(x, y + height);
                let p6 = image.get_val(x + width / 3, y + height);
                let p7 = image.get_val(x + width * 2 / 3, y + height);
                let p8 = image.get_val(x + width, y + height);

                let left_rect = p6 - p5 - p2 + p1;
                let mid_rect = p7 - p6 - p3 + p2;
                let right_rect = p8 - p7 - p4 + p3;

                right_rect - mid_rect + left_rect
            },
            FeatureType::VerticalTriplet => {
                // Points from upper left to bottom right
                let p1 = image.get_val(x, y);
                let p2 = image.get_val(x + width, y);
                let p3 = image.get_val(x, y + height / 3);
                let p4 = image.get_val(x + width, y + height / 3);
                let p5 = image.get_val(x, y + height * 2 / 3);
                let p6 = image.get_val(x + width, y + height * 2 / 3);
                let p7 = image.get_val(x, y + height);
                let p8 = image.get_val(x + width, y + height);

                let top_rect = p4 - p3 - p2 + p1;
                let mid_rect = p6 - p5 - p4 + p3;
                let bot_rect = p8 - p7 - p6 + p5;

                top_rect - mid_rect + bot_rect
            },
            FeatureType::Checkered => {
                // Points from upper left to bottom right
                let p1 = image.get_val(x, y);
                let p2 = image.get_val(x + width / 2, y);
                let p3 = image.get_val(x + width, y);
                let p4 = image.get_val(x, y + height / 2);
                let p5 = image.get_val(x + width / 2, y + height / 2);
                let p6 = image.get_val(x + width, y + height / 2);
                let p7 = image.get_val(x, y + height);
                let p8 = image.get_val(x + width / 2, y + height);
                let p9 = image.get_val(x + width, y + height);

                let ul_rect = p5 - p4 - p2 + p1;
                let bl_rect = p8 - p7 - p5 + p4;
                let ur_rect = p6 - p5 - p3 + p2;
                let br_rect = p9 - p8 - p6 + p5;

                br_rect + ul_rect - bl_rect - ur_rect
            },
        }
    }
}

