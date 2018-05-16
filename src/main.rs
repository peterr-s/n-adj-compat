extern crate rand;
use rand::distributions::{IndependentSample, Range};

extern crate rust2vec;
use rust2vec::{Embeddings, ReadText}; // TODO fix helper script so that this can use ReadWord2Vec instead of ReadText

use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use std::process::Command;

extern crate simd;
use simd::f32x4;

struct NAdjPair {
    noun: String,
    adj: String,
    confidence: f32,
}

#[derive(Clone)]
struct CoNLLEntry {
    lemma: String,
    pos: String,
    head: u8,
}

struct Perceptron {
    w: Vec<f32>,
}

impl Perceptron {
    fn predict(&mut self, x: Vec<f32>) -> f32 {
        self.w[0] + dot(&x, &self.w[1..])
    }

    fn train(&mut self, x: Vec<f32>, y: f32) -> f32 {
        // dot product
        let predicted: f32 = self.w[0] + dot(&x, &self.w[1..]);

        // if miscategorized
        if (y > 0.0f32) ^ (predicted > 0.0f32) {
            //  let error: f32 = (predicted - y) * if y > 0.0f32 { 1.0f32 } else { -1.0f32 };
            let error: f32 = y - predicted;

            self.w[0] += error;
            // TODO parallelize this!
            for i in 0..x.len() {
                self.w[i + 1] += error * x[i];
            }
        }

        predicted
    }
}

fn main() {
    let mut pairs: Vec<NAdjPair> = Vec::new();
    let mut rng = rand::thread_rng();

    let root_entry: CoNLLEntry = CoNLLEntry {
        lemma: "[root]".to_string(),
        pos: "[root]".to_string(),
        head: 0u8,
    };

    {
        // scope the file handles to close them when no longer needed
        // open pair file
        let pair_file: File = File::create("./pairs").expect("Error creating pair file");
        let mut pair_file: BufWriter<_> = BufWriter::new(pair_file);

        // open block text file
        let text_file: File = File::create("./text").expect("Error creating block text file");
        let mut text_file: BufWriter<_> = BufWriter::new(text_file);

        // process corpus files one at a time
        for path_container in fs::read_dir("./").unwrap() {
            let path: String = path_container.unwrap().path().display().to_string();
            if (&path).ends_with(".conll.gz") {
                // unzip file
                Command::new("gzip")
                    .arg("-df")
                    .arg(&path)
                    .status()
                    .expect("Error unzipping input file");

                // open input file
                let conll_name: String =
                    (&path).chars().take((&path).len() - 3).collect::<String>();
                let input: File = File::open(conll_name.clone()).expect("Error opening input file");
                let mut sentence: Vec<CoNLLEntry> = Vec::new();
                sentence.push(root_entry.clone()); // represents head, also makes indices easier

                // read one sentence at a time
                let mut input: BufReader<_> = BufReader::new(input);
                for line in input.lines().map(|l| match l {
                    Ok(s) => s,
                    Err(..) => String::new(),
                }) {
                    // when a sentence boundary (empty line) is hit
                    if line.len() == 0 {
                        // find all adjective/noun pairs
                        for entry in sentence.clone() {
                            // for each adjective
                            if entry.pos.starts_with("ADJ") {
                                let pair: NAdjPair = NAdjPair {
                                    noun: sentence[entry.head as usize].clone().lemma,
                                    adj: entry.lemma,
                                    confidence: 1.0f32, // DUMMY
                                };

                                // write pair to file for MI analysis
                                let pair_string: String =
                                    format!("{}\t{}\n", &(pair.noun), &(pair.adj));
                                pair_file
                                    .write_all(pair_string.as_bytes())
                                    .expect("Error writing adjective/noun pair to file");
                            }
                        }

                        // write sentence to embedding training file
                        let mut sentence_string: String = sentence
                            .iter()
                            .map(|e| {
                                let mut str: String = e.lemma.clone();
                                str.push_str(" ");
                                str
                            })
                            .collect();
                        sentence_string.push_str("\n");
                        text_file
                            .write_all(sentence_string.as_bytes())
                            .expect("Error writing sentence to file");

                        // clear sentence (except root)
                        sentence.resize(1, root_entry.clone()); // should never need to actually fill anything
                    } else {
                        // otherwise, extract the important CoNLL fields
                        let fields: Vec<String> = line.split_whitespace()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>();
                        if fields.len() >= 7 {
                            let entry: CoNLLEntry = CoNLLEntry {
                                lemma: fields[2].clone(),
                                pos: fields[3].clone(), // DEBUG: in actual corpus this is 4
                                head: match fields[6].parse::<u8>() {
                                    Ok(n) => n,
                                    Err(..) => 0,
                                },
                            };
                            sentence.push(entry);
                        } else {
                            // DEBUG
                            println!("found short line:",);
                            for field in fields {
                                println!("\t{}", field);
                            }
                        }
                    }
                }
                // rezip file
                Command::new("gzip")
                    .arg("-f")
                    .arg(&conll_name)
                    .status()
                    .expect("Error unzipping input file");
            }
        }
    } // this closes the file handles

    // run MI utility and capture output
    Command::new("compute-mi")
        .arg("-m")
        .arg("nsc")
        .arg("1,2")
        .arg("./pairs")
        .arg("./nmi")
        .status()
        .expect("Could not compute mutual information"); // do not capture output directly, use buffer since it will be massive
    {
        let pair_file: File = File::open("./nmi").expect("Could not open mutual information file");
        let pair_file: BufReader<_> = BufReader::new(pair_file);
        for line in pair_file.lines().map(|l| match l {
            Ok(s) => s,
            Err(..) => String::new(),
        }) {
            // get positive examples
            let fields: Vec<String> = line.split_whitespace()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            pairs.push(NAdjPair {
                noun: fields[0].clone(),
                adj: fields[1].clone(),
                confidence: fields[2].parse::<f32>().unwrap(),
            });
        }
    }

    // generate negative examples
    for _ in 0..pairs.len() {
        let range: Range<usize> = Range::new(0usize, pairs.len());
        let noun: String = pairs[range.ind_sample(&mut rng)].noun.clone();
        let adj: String = pairs[range.ind_sample(&mut rng)].adj.clone();
        // TODO check that the examples do not exist
        pairs.push(NAdjPair {
            noun,
            adj,
            confidence: -1.0f32,
        });
    }

    // train embeddings
    Command::new("./train_embeddings.py")
        .output()
        .expect("Could not train embeddings");
    let mut embedding_model: Embeddings;
    {
        let embedding_file: File =
            File::open("./embeddings").expect("Could not open embedding file");
        let mut embedding_file: BufReader<_> = BufReader::new(embedding_file);
        embedding_model =
            Embeddings::read_text(&mut embedding_file).expect("Could not read embedding file");

        embedding_model.normalize();
    }

    // weights
    let mut perceptron: Perceptron = Perceptron {
        w: {
            let range: Range<f32> = Range::new(0.0f32, 1.0f32);
            vec![range.ind_sample(&mut rng); (embedding_model.embed_len() * 2) + 1]
        },
    };

    // for each pair
    for pair in pairs {
        // concatenate embeddings to get feature vector
        let mut x: Vec<f32> = Vec::with_capacity(2 * embedding_model.embed_len());
        x.append(&mut match embedding_model.embedding(&(pair.adj)) {
            Some(v) => v.to_vec(),
            None => vec![0.0f32; embedding_model.embed_len()],
        });
        x.append(&mut match embedding_model.embedding(&(pair.noun)) {
            Some(v) => v.to_vec(),
            None => vec![0.0f32; embedding_model.embed_len()],
        });

        perceptron.train(x, pair.confidence);
    }

    // get test pairs
    // calc effectiveness on test pairs
    let mut test_pairs: Vec<NAdjPair> = Vec::new();
    {
        let input: BufReader<File> =
            BufReader::new(File::open("./test").expect("Could not open test pair file"));
        for line in input.lines().map(|l| l.unwrap()) {
            let fields: Vec<String> = line.split_whitespace()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            if fields.len() >= 3 {
                match fields[2].parse::<f32>() {
                    Ok(n) => test_pairs.push(NAdjPair {
                        noun: fields[0].clone(),
                        adj: fields[1].clone(),
                        confidence: n,
                    }),
                    Err(..) => (),
                }
            }
        }
    }

    let mut total: u32 = 0u32;
    let mut correct: u32 = 0u32;
    for pair in test_pairs {
        correct += {
            // concatenate embeddings to get feature vector
            let mut x: Vec<f32> = Vec::with_capacity(2 * embedding_model.embed_len());
            x.append(&mut match embedding_model.embedding(&(pair.adj)) {
                Some(v) => v.to_vec(),
                None => vec![0.0f32; embedding_model.embed_len()],
            });
            x.append(&mut match embedding_model.embedding(&(pair.noun)) {
                Some(v) => v.to_vec(),
                None => vec![0.0f32; embedding_model.embed_len()],
            });

            if (pair.confidence > 0.0f32) ^ (perceptron.predict(x) > 0.0f32) {
                println!("miscategorized {} {}", pair.adj, pair.noun);
                0
            } else {
                1
            }
        };

        total += 1;
    }
    println!("{} of {} test pairs correct", correct, total);

    // DEBUG
    print!("weights: [ ");
    for weight in perceptron.w {
        assert!(!weight.is_nan());
        print!("{} ", weight);
    }
    println!("]");
}

fn dot(mut u: &[f32], mut v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len());

    let mut sums = f32x4::splat(0.0);

    while u.len() >= 4 {
        let a = f32x4::load(u, 0);
        let b = f32x4::load(v, 0);

        sums = sums + a * b;

        u = &u[4..];
        v = &v[4..];
    }

    sums.extract(0) + sums.extract(1) + sums.extract(2) + sums.extract(3) + dot_slow(u, v)
}

fn dot_slow(u: &[f32], v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len());

    let mut sum: f32 = 0.0f32;
    for i in 0..u.len() {
        sum += u[i] * v[i];
    }

    sum
}
