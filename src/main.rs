extern crate rand;
use rand::distributions::{IndependentSample, Range};
//use rand::Rng;

struct NAdjPair {
    noun: String,
    adj: String,
    confidence: f32,
}

fn main() {
    // weights
    let mut rng = rand::thread_rng();
    let range = Range::new(0.0f32, 1.0f32);
    let mut w: (f32, f32) = (range.ind_sample(&mut rng), range.ind_sample(&mut rng));

    // create pairs
    let mut pairs: Vec<NAdjPair> = Vec::new();
    pairs.push(NAdjPair {
        noun: "door".to_string(),
        adj: "green".to_string(),
        confidence: 1.0f32,
    });
    pairs.push(NAdjPair {
        noun: "door".to_string(),
        adj: "angry".to_string(),
        confidence: -1.0f32,
    });
    pairs.push(NAdjPair {
        noun: "information".to_string(),
        adj: "mutual".to_string(),
        confidence: 1.0f32,
    });
    pairs.push(NAdjPair {
        noun: "ideas".to_string(),
        adj: "green".to_string(),
        confidence: -1.0f32,
    });
    pairs.push(NAdjPair {
        noun: "door".to_string(),
        adj: "green".to_string(),
        confidence: 1.0f32,
    });
    pairs.push(NAdjPair {
        noun: "door".to_string(),
        adj: "angry".to_string(),
        confidence: -1.0f32,
    });
    pairs.push(NAdjPair {
        noun: "information".to_string(),
        adj: "mutual".to_string(),
        confidence: 1.0f32,
    });
    pairs.push(NAdjPair {
        noun: "ideas".to_string(),
        adj: "green".to_string(),
        confidence: -1.0f32,
    });

    // for each pair
    for pair in pairs {
        // calculate mutual information
        let x: f32 = mutual_information(&pair.noun, &pair.adj);

        // dot product
        let predicted: f32 = w.0 + (x * w.1);

        // if miscategorized
        if (pair.confidence > 0.0f32) != (predicted > 0.0f32) {
            print!("miscategorized! ");
            let error: f32 = (predicted - pair.confidence) * if pair.confidence > 0.0f32 {
                1.0f32
            } else {
                -1.0f32
            };
            w.0 += error;
            w.1 += error * x;
        }

        println!(
            "{}, {} ({}): {}",
            pair.noun, pair.adj, pair.confidence, predicted
        );
    }

    println!("weights: [{}, {}]", w.0, w.1);
    // get test pairs
    // calc effectiveness on test pairs
}

fn mutual_information(noun: &String, adjective: &String) -> f32 {
    if *noun == "door".to_string() {
        return {
            if *adjective == "green".to_string() {
                0.7f32
            } else {
                -0.82f32
            }
        };
    } else if *noun == "ideas".to_string() {
        return -0.63f32;
    }

    1.0f32
}
