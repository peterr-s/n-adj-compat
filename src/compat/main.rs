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

extern crate serde_json;
use serde_json::Value;

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

use std::process::{Command, Child, Stdio};

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
        let embedding_file: File = File::open(
            settings["embedding_file"]
                .as_str()
                .expect("Could not get embedding path from settings"),
        ).expect("Could not open embedding file");
        let mut embedding_file: BufReader<_> = BufReader::new(embedding_file);
        embedding_model = Embeddings::read_word2vec_binary(&mut embedding_file)
            .expect("Could not read embedding file");

        embedding_model.normalize();
    }
    println!("Read embeddings   ");

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
    let batch_size: usize = settings["batch_size"].as_u64().unwrap_or_else(|| {
        // use _else for lazy eval, avoid printing err msg on success
        let default_batch_size: u64 = 1000;
        eprintln!(
            "Could not find batch size in settings; defaulting to {}",
            default_batch_size
        );
        default_batch_size
    }) as usize;
    let batch_size_f: f32 = batch_size as f32;
    let epoch_ct: usize = settings["epoch_ct"].as_u64().unwrap_or_else(|| {
        let default_epoch_ct: u64 = 3;
        eprintln!(
            "Could not find epoch ct in settings; defaulting to {}",
            default_epoch_ct
        );
        default_epoch_ct
    }) as usize;
    let x_width: usize = embedding_model.embed_len() * 2;
    let x_size: usize = x_width * batch_size;
    let mut train_loss: Tensor<f32> = Tensor::new(&[1]).with_values(&[0.0f32]).unwrap();

    // for saving misclassified samples
    let misclassified_file: File =
        File::create("./misclassified").expect("Could not create misclassification file");
    let mut misclassified_file: BufWriter<_> = BufWriter::new(misclassified_file);

    // keep these as named variables to be iterated over on every reshuffle
    let mut pos_files: Vec<&str> = Vec::new();
    let mut neg_files: Vec<&str> = Vec::new();
    for sample_file in settings["sample_files"]
        .as_array()
        .expect("Could not get sample files from settings")
        .iter()
    {
        let path: &str = sample_file["path"]
            .as_str()
            .expect("Could not get sample file path from settings");
        if sample_file["is_positive"]
            .as_bool()
            .expect("Could not get sample file validity from settings")
        {
            pos_files.push(path);
        } else {
            neg_files.push(path);
        }
    }

    // get pipe buffer size from settings
    /*let buf_size: usize = settings["buf_size"]
        .as_u64()
        .expect("Could not get buffer size from settings") as usize;*/
    const BUF_SIZE: usize = 51200usize; // can't read from file and still declare as array

    // train each epoch on complete set
    for epoch in 0..epoch_ct {
        // shuffle negative and positive samples separately so they are implicitly annotated
        let mut cat: Child = Command::new("cat")
            .args(&neg_files)
            .stdout(Stdio::piped())
            .spawn()
            .expect("Could not concatenate negative sample files");
        let mut shuf: Child = Command::new("shuf")
            .arg("-o")
            .arg("./neg_shuffled")
            .stdin(Stdio::piped())
            .spawn()
            .expect("Could not shuffle negative samples");
        if let Some(ref mut cat_out) = cat.stdout {
            if let Some(ref mut shuf_in) = shuf.stdin {
                /*let mut buf: [u8; BUF_SIZE] = [0u8; BUF_SIZE]; // use 10 kB buffer
                while 0 < cat_out.read(&mut buf).expect("Could not read from negative cat output") {
                    shuf_in.write(&mut buf).expect("Could not write to negative shuf input");
                }*/
                
                // write the remainder of the file after the last full chunk
                let mut buf: Vec<u8> = Vec::with_capacity(BUF_SIZE);
                cat_out.read_to_end(&mut buf).expect("Could not read end of negative cat output");
                shuf_in.write_all(&mut buf).expect("Could not write end of negative shuf input");
            }
        }
        shuf.wait().expect("Could not finish shuffling negative samples");
        cat = Command::new("cat")
            .args(&pos_files)
            .stdout(Stdio::piped())
            .spawn()
            .expect("Could not concatenate positive sample files");
        shuf = Command::new("shuf")
            .arg("-o")
            .arg("./pos_shuffled")
            .stdin(Stdio::piped())
            .spawn()
            .expect("Could not shuffle positive samples");
        if let Some(ref mut cat_out) = cat.stdout {
            if let Some(ref mut shuf_in) = shuf.stdin {
                /*let mut buf: [u8; BUF_SIZE] = [0u8; BUF_SIZE]; // use 10 kB buffer
                while 0 < cat_out.read(&mut buf).expect("Could not read from negative cat output") {
                    shuf_in.write(&mut buf).expect("Could not write to negative shuf input");
                }*/
                
                // write the remainder of the file after the last full chunk
                let mut buf: Vec<u8> = Vec::with_capacity(BUF_SIZE);
                cat_out.read_to_end(&mut buf).expect("Could not read end of positive cat output");
                shuf_in.write_all(&mut buf).expect("Could not write end of positive shuf input");
            }
        }
        shuf.wait().expect("Could not finish shuffling positive samples");

        // create iterators over the shuffled data so it can be read on the fly
        let neg_file: File =
            File::open("./neg_shuffled").expect("Could not open shuffled negatives");
        let neg_file: BufReader<File> = BufReader::new(neg_file);
        let mut neg_iter = neg_file
            .lines()
            .map(|l| l.expect("Error reading negative sample"));
        let pos_file: File =
            File::open("./pos_shuffled").expect("Could not open shuffled positives");
        let pos_file: BufReader<File> = BufReader::new(pos_file);
        let mut pos_iter = pos_file
            .lines()
            .map(|l| l.expect("Error reading positive sample"));

        // while there are enough samples left to fill a batch
        let mut batch: usize = 0usize;
        let mut pairs: Vec<NAdjPair>;
        while {
            batch += 1;

            pairs = Vec::with_capacity(batch_size);
            read_samples(
                &mut pairs,
                &mut neg_iter,
                false,
                &embedding_model,
                batch_size / 2,
            );
            read_samples(
                &mut pairs,
                &mut pos_iter,
                true,
                &embedding_model,
                batch_size,
            );
            if pairs.len() < batch_size {
                read_samples(
                    &mut pairs,
                    &mut neg_iter,
                    false,
                    &embedding_model,
                    batch_size,
                );
            }
            pairs.len() == batch_size
        } {
            // shuffle batch to ramdomize negatives and positives
            // TODO verify this is a uniform shuffle (testing indicates it is but docs do not confirm)
            thread_rng().shuffle(&mut pairs);

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

            // save misclassified examples
            misclassified_file
                .write_all(b"noun, adjective, confidence, validity\n")
                .expect("Could not write misclassification headers");

            // train step (3/4 of all batches)
            if batch % 4 != 0 {
                let mut train_step: StepWithGraph = StepWithGraph::new();
                train_step.add_input(&x, 0, &x_batch);
                train_step.add_input(&y, 0, &y_batch);
                train_step.add_target(&train);
                let loss_idx: OutputToken = train_step.request_output(&loss, 0);

                session
                    .run(&mut train_step)
                    .expect("Could not run training step");

                train_loss = train_step.take_output(loss_idx).unwrap();
            }
            // validation step (every 4th batch)
            else {
                let mut validation_step: StepWithGraph = StepWithGraph::new();
                validation_step.add_input(&x, 0, &x_batch);
                validation_step.add_input(&y, 0, &y_batch);
                let y_pred_idx: OutputToken = validation_step.request_output(&y_pred, 0);
                let loss_idx: OutputToken = validation_step.request_output(&loss, 0);
                let accuracy_idx: OutputToken = validation_step.request_output(&accuracy, 0);

                session
                    .run(&mut validation_step)
                    .expect("Could not run validation step");

                let y_pred_val: Tensor<f32> = validation_step.take_output(y_pred_idx).unwrap();
                let loss_val: Tensor<f32> = validation_step.take_output(loss_idx).unwrap();
                let accuracy_val: Tensor<f32> = validation_step.take_output(accuracy_idx).unwrap();

                // get specific misclassifications and write to file
                let y_vec: Vec<f32> = y_batch.to_vec();
                let y_pred_vec: Vec<f32> = y_pred_val.to_vec();
                let pairs: &[NAdjPair] = &pairs;
                let mut false_pos: usize = 0usize;
                let mut false_neg: usize = 0usize;
                for i in 0..batch_size {
                    let pred_valid: bool = y_pred_vec[i] > 0.5;
                    let is_valid: bool = y_vec[i] > 0.5;
                    if pred_valid != is_valid {
                        let misclassified_string: String = format!(
                            "{}, {}, {}, {}\n",
                            &(pairs[i].noun),
                            &(pairs[i].adj),
                            y_pred_vec[i],
                            y_vec[i],
                        );
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

                // print batch results
                print!(
                        "Epoch: {}\t Batch: {}\tTrain loss: {2:1.4}\tVal loss: {3:1.4}\tAccuracy: {4:1.4} ({5:1.4} t1, {6:1.4} t2)\r",
                        epoch,
                        batch,
                        train_loss.deref()[0],
                        loss_val.deref()[0],
                        accuracy_val.deref()[0],
                        (false_pos as f32) / batch_size_f,
                        (false_neg as f32) / batch_size_f
                    );
                match std::io::stdout().flush() // stdout is not implicitly flushed on carriage return and this is not a bottleneck
                {
                    Ok(_) => (),
                    Err(_) => (),
                };
            }
        }
        println!("");
    }
    println!("Trained compatibility prediction on all data");

    // save weights
    session.run(&mut save_step).expect("Could not save weights");

    // load weights
    /*session
        .run(&mut load_step)
        .expect("Could not load weights");*/}

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
        }
    }
}
