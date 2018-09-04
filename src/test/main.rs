use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Lines, Read, Write};

extern crate rust2vec;
use rust2vec::{Embeddings, ReadWord2Vec};

extern crate tensorflow;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Operation;
use tensorflow::OutputToken;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;

use std::ops::Deref;

extern crate serde_json;
use serde_json::Value;

#[derive(Clone)]
struct NAdjPair {
    noun: String,
    adj: String,
    valid: bool,
    embedding: Vec<f32>,
}

fn main() {
    // read settings
    let settings: Value;
    {
        let mut settings_file: File =
            File::open("./settings.json").expect("Could not open settings file");
        let mut settings_str: String = String::new();
        settings_file
            .read_to_string(&mut settings_str)
            .expect("Could not read settings file");
        settings = serde_json::from_str(&settings_str).expect("Could not parse settings");
    }

    // read embeddings
    print!("Reading embeddings\r");
    let mut embedding_model: Embeddings;
    {
        let mut embedding_file: BufReader<File> = BufReader::new(
            File::open(
                settings["embedding_file"]
                    .as_str()
                    .expect("Could not get embedding path from settings"),
            ).expect("Could not open embedding file"),
        );
        embedding_model = Embeddings::read_word2vec_binary(&mut embedding_file)
            .expect("Could not read embedding file");

        embedding_model.normalize();
    }
    println!("Read embeddings   ");

    // load data
    let mut pairs: Vec<NAdjPair> = Vec::new();
    {
        let mut input: BufReader<File> =
            BufReader::new(File::open("neg_gs").expect("Could not open negative gold standard"));
        let mut iter: std::iter::Map<Lines<_>, _> = input
            .lines()
            .map(|l| l.expect("Error reading negative sample"));
        read_samples(
            &mut pairs,
            &mut iter,
            false,
            &embedding_model,
            usize::max_value(),
        );
        input =
            BufReader::new(File::open("pos_gs").expect("Could not open positive gold standard"));
        let mut iter: std::iter::Map<Lines<_>, _> = input // type ascription doesn't want to work, so this is the only way to [implicitly] annotate the type of l in the closure
            .lines()
            .map(|l| l.expect("Error reading positive samples"));
        read_samples(
            &mut pairs,
            &mut iter,
            true,
            &embedding_model,
            usize::max_value(),
        );
    }

    // load graph
    let mut graph: Graph = Graph::new();
    let mut proto: Vec<u8> = Vec::new();
    {
        let mut graph_file: File = File::open(
            settings["graph_file"]
                .as_str()
                .expect("Could not get graph path from settings"),
        ).expect("Could not open graph file");
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
        .expect("Could not find variable \"x\" in graph");
    let y: Operation = graph
        .operation_by_name_required("y")
        .expect("Could not find variable \"y\" in graph");
    let y_pred: Operation = graph
        .operation_by_name_required("y_pred")
        .expect("Could not find variable \"y_pred\" in graph");
    let loss: Operation = graph
        .operation_by_name_required("loss")
        .expect("Could not find variable \"loss\" in graph");

    // matrix transformation only
    let is_mat: bool;
    let n_mat_idx: Operation;
    let a_mat_idx: Operation;
    {
        let n_mat_idx_opt: Option<Operation> = graph
            .operation_by_name("n_mat_rs")
            .expect("Unexpected NUL in graph file"); // why does this get thrown here if the graph is already read?
        let a_mat_idx_opt: Option<Operation> = graph
            .operation_by_name("a_mat_rs")
            .expect("Unexpected NUL in graph file");
        if n_mat_idx_opt.is_some() && a_mat_idx_opt.is_some() {
            is_mat = true;
            n_mat_idx = n_mat_idx_opt.unwrap();
            a_mat_idx = a_mat_idx_opt.unwrap();
        } else {
            is_mat = false;
            n_mat_idx = graph.operation_by_name_required("loss").unwrap(); // this is just so the compiler doesn't fight me (no way to make an op not attached to a graph; need placeholder)
            a_mat_idx = graph.operation_by_name_required("loss").unwrap();
        }
    }

    // save and load operations
    let load: Operation = graph
        .operation_by_name_required("save/restore_all")
        .expect("Could not find load operation in graph");
    let weight_path: Operation = graph
        .operation_by_name_required("save/Const")
        .expect("Could not find weight path in graph");
    let weight_path_tensor: Tensor<String> = Tensor::from("./model.ckpt".to_string());

    // define load step
    let mut load_step = StepWithGraph::new();
    load_step.add_input(&weight_path, 0, &weight_path_tensor);
    load_step.add_target(&load);

    // load weights
    session.run(&mut load_step).expect("Could not load weights");

    // rearrange data into batch(es)
    let batch_size: usize = settings["batch_size"].as_u64().unwrap_or_else(|| {
        // use _else for lazy eval, avoid printing err msg on success
        let default_batch_size: u64 = 1000;
        eprintln!(
            "Could not find batch size in settings; defaulting to {}",
            default_batch_size
        );
        default_batch_size
    }) as usize;

    // fill vec
    let test_ct: usize = pairs.len();
    let filler_ct: usize = batch_size - (test_ct % batch_size);
    pairs.append(&mut vec![
        NAdjPair {
            noun: String::new(),
            adj: String::new(),
            valid: false,
            embedding: vec![0.0f32; 600],
        };
        filler_ct
    ]);

    // for saving misclassified samples
    let mut misclassified_file: BufWriter<File> = BufWriter::new(
        File::create("./misclassified_gs").expect("Could not create misclassification file"),
    );
    misclassified_file
        .write_all(if is_mat {
            b"noun, adjective, noun mat, adj mat, confidence, validity\n"
        } else {
            b"noun, adjective, confidence, validity\n"
        })
        .expect("Could not write misclassification headers");

    let x_width: usize = embedding_model.embed_len() * 2;
    let x_size: usize = x_width * batch_size;
    let mut false_pos: usize = 0usize;
    let mut false_neg: usize = 0usize;
    let mut total_loss: f32 = 0.0f32;
    for _ in 0..(pairs.len() / batch_size) {
        // concatenate embeddings to get feature vector
        let x_batch: Tensor<f32>;
        let y_batch: Tensor<f32>;
        {
            let mut vec: Vec<f32>;
            let mut transposed: Vec<f32>;
            x_batch = Tensor::new(&[
                2u64 * (embedding_model.embed_len() as u64),
                batch_size as u64,
            ]).with_values({
                vec = Vec::with_capacity(x_size);
                for e in pairs.iter() {
                    vec.append(&mut e.embedding.clone());
                }

                transposed = Vec::with_capacity(x_size);
                unsafe {
                    transposed.set_len(x_size);
                }
                for e in 0..x_size {
                    let row: usize = e / x_width;
                    let col: usize = e % x_width;
                    transposed[(col * batch_size) + row] = vec[e];
                }

                transposed.as_mut_slice()
            })
                .unwrap();
            // train output is binary compatibility, run output is confidences as float
            y_batch = Tensor::new(&[1u64, batch_size as u64])
                .with_values({
                    // assign to vec first because of type inference in collect()
                    vec = pairs
                        .iter()
                        .map(|e| if e.valid { 1.0f32 } else { 0.0f32 })
                        .collect();
                    vec.as_mut_slice()
                })
                .unwrap();
        }

        // run graph
        let mut test_step: StepWithGraph = StepWithGraph::new();
        test_step.add_input(&x, 0, &x_batch);
        test_step.add_input(&y, 0, &y_batch);
        let y_pred_idx: OutputToken = test_step.request_output(&y_pred, 0);
        let loss_idx: OutputToken = test_step.request_output(&loss, 0);
        let n_mat_idx_idx: OutputToken = test_step.request_output(&n_mat_idx, 0);
        let a_mat_idx_idx: OutputToken = test_step.request_output(&a_mat_idx, 0);

        session
            .run(&mut test_step)
            .expect("Could not run test step");

        let y_pred_val: Tensor<f32> = test_step.take_output(y_pred_idx).unwrap();
        let loss_val: Tensor<f32> = test_step.take_output(loss_idx).unwrap();
        let n_mat_idx_val: Tensor<f32> = test_step.take_output(n_mat_idx_idx).unwrap();
        let a_mat_idx_val: Tensor<f32> = test_step.take_output(a_mat_idx_idx).unwrap();

        // get specific misclassifications and write to file
        let y_vec: Vec<f32> = y_batch.to_vec();
        let y_pred_vec: Vec<f32> = y_pred_val.to_vec();
        let n_mat_idx_vec: Vec<f32> = n_mat_idx_val.to_vec();
        let a_mat_idx_vec: Vec<f32> = a_mat_idx_val.to_vec();
        let pairs: &[NAdjPair] = &pairs;
        for i in 0..(batch_size - filler_ct) {
            let pred_valid: bool = y_pred_vec[i] > 0.5;
            let is_valid: bool = y_vec[i] > 0.5;
            if pred_valid != is_valid {
                let misclassified_string: String = if is_mat {
                    format!(
                        "{}, {}, {}, {}, {:.0}, {:.0}\n",
                        &(pairs[i].noun),
                        &(pairs[i].adj),
                        n_mat_idx_vec[i],
                        a_mat_idx_vec[i],
                        y_pred_vec[i],
                        y_vec[i],
                    )
                } else {
                    format!(
                        "{}, {}, {}, {}\n",
                        &(pairs[i].noun),
                        &(pairs[i].adj),
                        y_pred_vec[i],
                        y_vec[i],
                    )
                };
                misclassified_file
                    .write_all(misclassified_string.as_bytes())
                    .expect("Could not write misclassified pair to file");

                // y is column major with positive confidences first
                match pred_valid {
                    true => false_pos += 1,  // type 1 error
                    false => false_neg += 1, // type 2 error
                };
            }
        }
        total_loss += loss_val.deref()[0];
    }

    // evaluate
    println!("{} samples evaluated", test_ct);
    let fp_rate: f32 = (false_pos as f32) / (test_ct as f32);
    let fn_rate: f32 = (false_neg as f32) / (test_ct as f32);
    let accuracy: f32 = 1.0f32 - fp_rate - fn_rate;
    let loss: f32 = total_loss * (pairs.len() as f32) / (test_ct as f32);
    //    (((test_ct * batch_size) as f32) * total_loss) / ((pairs.len() * pairs.len()) as f32);
    println!(
        "{:1.3} accuracy ({:1.3} fp, {:1.3} fn), {:1.3} loss",
        accuracy, fp_rate, fn_rate, loss
    );
    println!(
        "{} filler, {} false positives, {} false negatives",
        filler_ct, false_pos, false_neg
    );
}

fn read_samples<I>(
    pairs: &mut Vec<NAdjPair>,
    iter: &mut I,
    valid: bool,
    embedding_model: &Embeddings,
    max_length: usize,
) where
    I: Iterator<Item = String>,
{
    while pairs.len() < max_length {
        if let Some(line) = iter.next() {
            // get samples
            let fields: Vec<String> = line
                .split_whitespace()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            pairs.push(NAdjPair {
                noun: fields[0].clone(),
                adj: fields[1].clone(),
                valid,
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
        } else {
            return;
        }
    }
}
