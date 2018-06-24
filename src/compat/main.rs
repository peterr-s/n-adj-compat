extern crate rand;
use rand::distributions::{IndependentSample, Range};
use rand::{Rng, thread_rng};

extern crate rust2vec;
use rust2vec::{Embeddings, ReadText}; // TODO fix helper script so that this can use ReadWord2Vec instead of ReadText

extern crate tensorflow;
//use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
//use tensorflow::Status;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;
use tensorflow::Operation;
use tensorflow::OutputToken;

use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

use std::process::Command;

use std::ops::Deref;

/*extern crate itertools;
use itertools::Itertools;*/

extern crate simd;
use simd::f32x4;

struct NAdjPair {
    noun: String,
    adj: String,
    confidence: f32,
    embedding: Vec<f32>,
}

#[derive(Clone)]
struct CoNLLEntry {
    lemma: String,
    pos: String,
    head: u8,
}

fn main() {
    let mut pairs: Vec<NAdjPair> = Vec::new();
    let mut rng = rand::thread_rng();

    let root_entry: CoNLLEntry = CoNLLEntry {
        lemma: "[root]".to_string(),
        pos: "[root]".to_string(),
        head: 0u8,
    };

    /*
    // scope the file handles to close them when no longer needed
    {
        // open pair file
        let pair_file: File = File::create("./pairs").expect("Error creating pair file");
        let mut pair_file: BufWriter<_> = BufWriter::new(pair_file);

        // open block text file
        let text_file: File = File::create("./text").expect("Error creating block text file");
        let mut text_file: BufWriter<_> = BufWriter::new(text_file);

        // process corpus files one at a time
        let mut path_idx: usize = 0;
        let path_ct: usize;
        for path_container in {
            let file_list: Vec<std::fs::DirEntry> = fs::read_dir("./corpus/")
                .unwrap()
                .map(|r| r.unwrap())
                .collect();
            path_ct = file_list.len();
            file_list
        } {
            let path: String = path_container.path().display().to_string();
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
                                    confidence: 1.0f32, // dummy; will be overwritten
                                    embedding: vec![0.0f32; 600], // will also be overwritten
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
                            println!("found short line:");
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
                    .spawn()
                    .expect("Error rezipping input file");
            }

            path_idx += 1;
            print!("{} of {} files processed\r", path_idx, path_ct);
        }
        println!("");
    } // this closes the file handles
    */

    // train embeddings
    /*print!("Training embeddings\r");
    Command::new("./train_embeddings.py")
        .output()
        .expect("Could not train embeddings");*/
    print!("Reading embeddings\r");
    let mut embedding_model: Embeddings;
    {
        let embedding_file: File =
            File::open("./embeddings").expect("Could not open embedding file");
        let mut embedding_file: BufReader<_> = BufReader::new(embedding_file);
        embedding_model =
            Embeddings::read_text(&mut embedding_file).expect("Could not read embedding file");

        embedding_model.normalize();
    }
    //println!("Trained embeddings ");
    println!("Read embeddings   ");

    // run MI utility and capture output
    print!("Computing mutual information\r");
    /*Command::new("compute-mi")
        .arg("-m")
        .arg("nsc")
        .arg("1,2")
        .arg("./pairs")
        .arg("./nmi")
        .status()
        .expect("Could not compute mutual information"); // do not capture output directly, use buffer since it will be massive
    */{
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
                embedding: {
                    let mut vec: Vec<f32> = Vec::with_capacity(2 * embedding_model.embed_len());
                    vec.append(&mut match embedding_model.embedding(&(fields[0].clone())) {
                        Some(v) => v.to_vec(),
                        None => {
                            continue;
                        } // do not train on unknown words
                    });
                    vec.append(&mut match embedding_model.embedding(&(fields[1].clone())) {
                        Some(v) => v.to_vec(),
                        None => {
                            continue;
                        }
                    });
                    vec
                },
            });
        }
    }
    println!("Computed mutual information ");

    // generate negative examples
    print!("Generating negative examples\r");
    for _ in 0..pairs.len() {
        let range: Range<usize> = Range::new(0usize, pairs.len());
        let noun: String = pairs[range.ind_sample(&mut rng)].noun.clone();
        let adj: String = pairs[range.ind_sample(&mut rng)].adj.clone();
        // TODO check that the examples do not exist
        pairs.push(NAdjPair {
            noun: noun.clone(),
            adj: adj.clone(),
            confidence: -1.0f32,
            embedding: {
                let mut vec: Vec<f32> = Vec::with_capacity(2 * embedding_model.embed_len());
                vec.append(&mut match embedding_model.embedding(&(noun)) {
                    Some(v) => v.to_vec(),
                    None => {
                        continue;
                    } // do not train on unknown words
                });
                vec.append(&mut match embedding_model.embedding(&(adj)) {
                    Some(v) => v.to_vec(),
                    None => {
                        continue;
                    }
                });
                vec
            },
        });
    }
    println!("Generated negative examples ");

    // load graph
    let mut graph: Graph = Graph::new();
    let mut proto: Vec<u8> = Vec::new();
    {
        let mut graph_file: File = File::open("graph.pb").expect("Could not open graph file");
        graph_file
            .read_to_end(&mut proto)
            .expect("Could not read graph file");
    }
    graph
        .import_graph_def(&proto, &ImportGraphDefOptions::new())
        .expect("Could not import graph");

    // create session, connect to graph variables
    let mut session: Session =
        Session::new(&SessionOptions::new(), &graph).expect("Could not create session");
    let x: Operation = graph
        .operation_by_name_required("x")
        .expect("Could not find variable x in graph");
    let y: Operation = graph
        .operation_by_name_required("y")
        .expect("Could not find variable y in graph");
    let y_pred: Operation = graph
        .operation_by_name_required("y_pred")
        .expect("Could not find variable y_pred in graph");
    let init: Operation = graph
        .operation_by_name_required("init")
        .expect("Could not find init operation in graph");
    let train: Operation = graph
        .operation_by_name_required("train")
        .expect("Could not find train operation in graph");
    let loss: Operation = graph
        .operation_by_name_required("loss")
        .expect("Could not find variable loss operation in graph");

    // initialize graph
    let mut init_step: StepWithGraph = StepWithGraph::new();
    init_step.add_target(&init);
    session
        .run(&mut init_step)
        .expect("Could not initialize graph");

    // shuffle training data
    // TODO verify this is a uniform shuffle (testing indicates it is but docs do not confirm)
    thread_rng().shuffle(&mut pairs);

    // split training data into batches
    let batch_size: usize = 100;
    let batch_ct: usize = pairs.len() / batch_size; // we can probably afford to discard the last partial batch

    // train on 3/4 of data, validate on 1/4
    for batch in 0..batch_ct {
        // concatenate embeddings to get feature vector
	let x_batch: Tensor<f32>;
	let y_batch: Tensor<f32>;
	{
		let mut vec: Vec<f32>;
		x_batch = Tensor::new(&[2u64 * (embedding_model.embed_len() as u64), batch_size as u64])
		    .with_values({
			vec =
			    Vec::with_capacity(2 * embedding_model.embed_len() * batch_size);
			for e in pairs[batch * batch_size..(batch + 1) * batch_size].iter() {
			    vec.append(&mut e.embedding.clone());
			}
			vec.as_mut_slice()
		    }).unwrap();
		// concatenate confidences to get output vector
		y_batch = Tensor::new(&[1u64, batch_size as u64]).with_values({
		    vec = Vec::with_capacity(batch_size);
		    for e in pairs[batch * batch_size..(batch + 1) * batch_size].iter() {
			vec.push(e.confidence);
		    }
		    vec.as_mut_slice()
		}).unwrap();
	}

        // train step
        if batch % 4 != 0 {
            let mut train_step: StepWithGraph = StepWithGraph::new();
            train_step.add_input(&x, 0, &x_batch);
            train_step.add_input(&y, 0, &y_batch);
            train_step.add_target(&train);
            
            session.run(&mut train_step).expect("Could not run training step");
        }
        // validation step
        else {
            let mut output_step: StepWithGraph = StepWithGraph::new();
            output_step.add_input(&x, 0, &x_batch);
            output_step.add_input(&y, 0, &y_batch);
            let loss_idx: OutputToken = output_step.request_output(&loss, 0);
            
            session.run(&mut output_step).expect("Could not run validation step");

            let loss_val: Tensor<f32> = output_step.take_output(loss_idx).unwrap();
            println!("Epoch: {}\t Loss: {}", batch / 4, loss_val.deref()[0]);
        }
    }

    // get test pairs
    // calc effectiveness on test pairs
    /*let mut test_pairs: Vec<NAdjPair> = Vec::new();
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
                None => {
                    println!("Unknown adjective");
                    vec![0.0f32; embedding_model.embed_len()]
                }
            });
            x.append(&mut match embedding_model.embedding(&(pair.noun)) {
                Some(v) => v.to_vec(),
                None => {
                    println!("Unknown noun");
                    vec![0.0f32; embedding_model.embed_len()]
                }
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
*/
    // DEBUG
    /*print!("weights: [ ");
    for weight in perceptron.w {
        assert!(!weight.is_nan());
        print!("{} ", weight);
    }
    println!("]");*/
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