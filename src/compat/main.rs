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

extern crate rand;
use rand::{thread_rng, Rng};

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

use std::ops::Deref;

struct NAdjPair {
    noun: String,
    adj: String,
    valid: bool,
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

    // read embeddings
    print!("Reading embeddings\r");
    let mut embedding_model: Embeddings;
    {
        let embedding_file: File =
            File::open("./embeddings").expect("Could not open embedding file");
        let mut embedding_file: BufReader<_> = BufReader::new(embedding_file);
        embedding_model = Embeddings::read_word2vec_binary(&mut embedding_file)
            .expect("Could not read embedding file");

        embedding_model.normalize();
    }
    println!("Read embeddings   ");

    // read MI values
    print!("Reading positive samples\r");
    read_samples(&mut pairs, "./pos", true, &embedding_model);
    println!("Reading negative samples\r");
    read_samples(&mut pairs, "./neg", false, &embedding_model);
    println!("Read all examples        ");

    // load graph
    let mut graph: Graph = Graph::new();
    let mut proto: Vec<u8> = Vec::new();
    {
        let mut graph_file: File = File::open("./graph.pb").expect("Could not open graph file");
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
    let init: Operation = graph
        .operation_by_name_required("init")
        .expect("Could not find init operation in graph");
    let train: Operation = graph
        .operation_by_name_required("train")
        .expect("Could not find train operation in graph");
    let loss: Operation = graph
        .operation_by_name_required("loss")
        .expect("Could not find variable \"loss\" in graph");
    let accuracy: Operation = graph
        .operation_by_name_required("accuracy")
        .expect("Could not find variable \"accuracy\" in graph");

    // save and load operations
    let save: Operation = graph
        .operation_by_name_required("save/control_dependency")
        .expect("Could not find save operation in graph");
    let load: Operation = graph
        .operation_by_name_required("save/restore_all")
        .expect("Could not find load operation in graph");
    let weight_path: Operation = graph
        .operation_by_name_required("save/Const")
        .expect("Could not find weight path in graph");
    let weight_path_tensor: Tensor<String> = Tensor::from("./model.ckpt".to_string());

    // define save step here for use anywhere
    let mut save_step = StepWithGraph::new();
    save_step.add_input(&weight_path, 0, &weight_path_tensor);
    save_step.add_target(&save);

    // define load step
    let mut load_step = StepWithGraph::new();
    load_step.add_input(&weight_path, 0, &weight_path_tensor);
    load_step.add_target(&load);

    // initialize graph
    let mut init_step: StepWithGraph = StepWithGraph::new();
    init_step.add_target(&init);
    session
        .run(&mut init_step)
        .expect("Could not initialize graph");

    // split training data into batches
    let batch_size: usize = 1000;
    let epoch_ct: usize = 3;
    let batch_ct: usize = pairs.len() / batch_size; // we can probably afford to discard the last partial batch
    let x_width: usize = embedding_model.embed_len() * 2;
    let x_size: usize = x_width * batch_size;
    let mut x_batches: Vec<Tensor<f32>> = Vec::with_capacity(batch_ct);
    let mut y_batches: Vec<Tensor<f32>> = Vec::with_capacity(batch_ct);

    // train each epoch on complete set
    for epoch in 0..epoch_ct {
        // shuffle training data
        // TODO verify this is a uniform shuffle (testing indicates it is but docs do not confirm)
        thread_rng().shuffle(&mut pairs);

        // generate batches
        println!("Split data into {} batches", batch_ct);
        for batch in 0..batch_ct {
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
                    for e in pairs[batch * batch_size..(batch + 1) * batch_size].iter() {
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
                // train output is a one-hot over {true, false}, run output is confidences as float
                y_batch = Tensor::new(&[2u64, batch_size as u64])
                    .with_values({
                        vec = Vec::with_capacity(2 * batch_size);
                        unsafe {
                            vec.set_len(2 * batch_size);
                        }
                        for i in 0..batch_size {
                            if pairs[i].valid {
                                vec[i] = 1.0f32;
                                vec[i + batch_size] = 0.0f32;
                            } else {
                                vec[i] = 0.0f32;
                                vec[i + batch_size] = 1.0f32;
                            }
                        }
                        vec.as_mut_slice()
                    })
                    .unwrap();
            }

            // add x and y to batches for perceptron training
            x_batches.push(x_batch);
            y_batches.push(y_batch);
        }
        println!("Trained MI prediction on all data");

        // train on all batches
        {
            // save misclassified examples
            let misclassified_file: File =
                File::create("./misclassified").expect("Could not create misclassification file");
            let mut misclassified_file: BufWriter<_> = BufWriter::new(misclassified_file);
            misclassified_file
                .write_all(b"noun, adjective, pred true, pred false, true, false\n")
                .expect("Could not write misclassification headers");

            for batch in 0..batch_ct {
                let x_batch: Tensor<f32> = x_batches.pop().unwrap();
                let y_batch: Tensor<f32> = y_batches.pop().unwrap();

                // train step (3/4 of all batches)
                if batch % 4 != 0 {
                    let mut train_step: StepWithGraph = StepWithGraph::new();
                    train_step.add_input(&x, 0, &x_batch);
                    train_step.add_input(&y, 0, &y_batch);
                    train_step.add_target(&train);

                    session
                        .run(&mut train_step)
                        .expect("Could not run training step");
                }
                // validation step (every 4th batch)
                else {
                    let mut output_step: StepWithGraph = StepWithGraph::new();
                    output_step.add_input(&x, 0, &x_batch);
                    output_step.add_input(&y, 0, &y_batch);
                    let y_pred_idx: OutputToken = output_step.request_output(&y_pred, 0);
                    let loss_idx: OutputToken = output_step.request_output(&loss, 0);
                    let accuracy_idx: OutputToken = output_step.request_output(&accuracy, 0);

                    session
                        .run(&mut output_step)
                        .expect("Could not run validation step");

                    let y_pred_val: Tensor<f32> = output_step.take_output(y_pred_idx).unwrap();
                    let loss_val: Tensor<f32> = output_step.take_output(loss_idx).unwrap();
                    let accuracy_val: Tensor<f32> = output_step.take_output(accuracy_idx).unwrap();
                    println!(
                        "Epoch: {}\t Batch: {}\tLoss: {}  \tAccuracy: {}",
                        epoch,
                        batch / 4,
                        loss_val.deref()[0],
                        accuracy_val.deref()[0]
                    );

                    // get specific misclassifications and write to file
                    let y_vec: Vec<f32> = y_batch.to_vec();
                    let y_pred_vec: Vec<f32> = y_pred_val.to_vec();
                    let pairs: &[NAdjPair] = &pairs[batch * batch_size..(batch + 1) * batch_size];
                    for i in 0..batch_size {
                        if (y_pred_vec[i] > y_pred_vec[i + batch_size])
                            != (y_vec[i] > y_vec[i + batch_size])
                        {
                            let misclassified_string: String = format!(
                                "{}, {}, {}, {}, {}, {}\n",
                                &(pairs[i].noun),
                                &(pairs[i].adj),
                                y_pred_vec[i],
                                y_pred_vec[i + batch_size],
                                y_vec[i],
                                y_vec[i + batch_size]
                            );
                            misclassified_file
                                .write_all(misclassified_string.as_bytes())
                                .expect("Could not write misclassified pair to file");
                        }
                    }
                }
            }
        }
    }
    println!("Trained compatibility prediction on all data");

    // save weights
    session.run(&mut save_step).expect("Could not save weights");

    // load weights
    /*session
        .run(&mut load_step)
        .expect("Could not load weights");*/}

fn read_samples(pairs: &mut Vec<NAdjPair>, path: &str, valid: bool, embedding_model: &Embeddings) {
    let pair_file: File = File::open(path).expect("Could not open sample file");
    let pair_file: BufReader<_> = BufReader::new(pair_file);
    for line in pair_file.lines().map(|l| match l {
        Ok(s) => s,
        Err(..) => String::new(),
    }) {
        // get samples
        let fields: Vec<String> = line.split_whitespace()
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
    }
}
