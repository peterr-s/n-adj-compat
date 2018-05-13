extern crate rand;
use rand::distributions::{IndependentSample, Range};
//use rand::Rng;
//use std::env;
use std::fs;
use std::fs::File;
//use std::io::prelude::*;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::Command;

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

fn main() {
    let mut pairs: Vec<NAdjPair> = Vec::new();

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
                Command::new("sh")
                    .arg("-c gzip -d")
                    .arg(&path)
                    .output()
                    .expect("Error unzipping input file");

                // open input file
                let input: File = File::open(
                    (&path).chars().take((&path).len() - 3).collect::<String>(),
                ).expect("Error opening input file");
                let mut sentence: Vec<CoNLLEntry> = Vec::new();
                sentence.push(root_entry.clone()); // represents head, also makes indices easier

                // read one sentence at a time
                let mut input: BufReader<_> = BufReader::new(input);
                let mut line: String = String::new();
                let mut ct: usize;
                while {
                    ct = match input.read_line(&mut line) {
                        Ok(n) => n,
                        Err(..) => 0,
                    };
                    ct
                } > 0
                {
                    // when a sentence boundary (empty line) is hit
                    if ct == 1 {
                        // find all adjective/noun pairs
                        for entry in sentence.clone() {
                            // for each adjective
                            if entry.pos[..3].to_string().eq(&"ADJ".to_string()) {
                                let pair: NAdjPair = NAdjPair {
                                    noun: sentence[entry.head as usize].clone().lemma,
                                    adj: entry.lemma,
                                    confidence: 1.0f32,
                                };

                                // write pair to file for MI analysis
                                let pair_string: String =
                                    format!("{}\t{}\n", &(pair.noun), &(pair.adj));
                                pair_file
                                    .write_all(pair_string.as_bytes())
                                    .expect("Error writing adjective/noun pair to file");

                                // add pair to training data list
                                //pairs.push(pair); // can't properly populate confidence yet
                            }
                        }

                        // write sentence to embedding training file
                        let sentence_string: String = sentence
                            .iter()
                            .map(|e| {
                                let mut str: String = e.lemma.clone();
                                str.push_str(" ");
                                str
                            })
                            .collect();
                        text_file
                            .write_all(sentence_string.as_bytes())
                            .expect("Error writing sentence to file");

                        // clear sentence (except root)
                        sentence.resize(1, root_entry.clone()) // should never need to actually fill anything
                    } else {
                        // otherwise, extract the important CoNLL fields
                        let fields: Vec<String> = line.split_whitespace()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>();
                        sentence.push(CoNLLEntry {
                            lemma: fields[2].clone(),
                            pos: fields[4].clone(),
                            head: match fields[6].parse::<u8>() {
                                Ok(n) => n,
                                Err(..) => 0,
                            },
                        });
                    }
                }
            }
        }
    } // this closes the file handles

    // run MI utility and capture output
    Command::new("compute-mi")
        .arg("-m nsc ./pairs ./nmi")
        .output()
        .expect("Could not compute mutual information");
    {
        let pair_file: File = File::open("./nmi").expect("Could not open mutual information file");
        let mut pair_file: BufReader<_> = BufReader::new(pair_file);
        let mut line: String = String::new();
        let mut ct: usize;
        while {
            ct = match pair_file.read_line(&mut line) {
                Ok(n) => n,
                Err(..) => 0,
            };
            ct
        } > 0
        {
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

    // weights
    let mut rng = rand::thread_rng();
    let range = Range::new(0.0f32, 1.0f32);
    let mut w: (f32, f32) = (range.ind_sample(&mut rng), range.ind_sample(&mut rng));

    // for each pair
    for pair in pairs {
        let x: f32 = pair.confidence;

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
