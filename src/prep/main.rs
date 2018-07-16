use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

use std::process::Command;

use std::collections::HashMap;
use std::slice::Iter;

extern crate rust2vec;
use rust2vec::{Embeddings, ReadWord2Vec, WordSimilarity};

extern crate rand;
use rand::distributions::{IndependentSample, Range};

extern crate bloom;
use bloom::{BloomFilter, ASMS};

extern crate serde_json;
use serde_json::Value;

struct NAdjPair {
    noun: String,
    adj: String,
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

    let root_entry: CoNLLEntry = CoNLLEntry {
        lemma: "[root]".to_string(),
        pos: "[root]".to_string(),
        head: 0u8,
    };

    // read embeddings
    print!("Reading embeddings\r");
    let mut embedding_model: Embeddings;
    {
        let embedding_path: &str = settings["embedding_file"]
            .as_str()
            .expect("Could not get embedding path from settings");
        let embedding_file: File = match File::open(embedding_path) {
            Ok(file) => file,
            Err(..) => {
                eprintln!("Could not open embedding file; making a dummy model");
                Command::new("./train_embeddings")
                    .status()
                    .expect("Could not create dummy embeddings");
                println!("Working with dummy embedding file; you will need to train embeddings and rerun this module");
                File::open(embedding_path).expect("Could not open dummy embedding file")
            }
        };
        let mut embedding_file: BufReader<_> = BufReader::new(embedding_file);
        embedding_model = Embeddings::read_word2vec_binary(&mut embedding_file)
            .expect("Could not read embedding file; consider deleting");

        embedding_model.normalize();
    }
    println!("Read embeddings   ");

    let mut pairs_pos: Vec<NAdjPair> = Vec::new();
    {
        // open block text file
        let text_file: File = File::create(
            settings["text_file"]
                .as_str()
                .expect("Could not get block text path from file"),
        ).expect("Error creating block text file");
        let mut text_file: BufWriter<_> = BufWriter::new(text_file);
        // process corpus files one at a time
        let mut path_idx: usize = 0;
        let path_ct: usize;
        for path_container in {
            let file_list: Vec<std::fs::DirEntry> = fs::read_dir(
                settings["corpus_dir"]
                    .as_str()
                    .expect("Could not get corpus directory from file"),
            ).unwrap()
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
                                // make sure at least one is present in the embedding model (will
                                // be useful for negative sampling if not as a pair)
                                let noun: String = sentence[entry.head as usize].clone().lemma;
                                let adj: String = entry.lemma;
                                if embedding_model.embedding(&noun).is_some()
                                    || embedding_model.embedding(&adj).is_some()
                                {
                                    let pair: NAdjPair = NAdjPair { noun, adj };

                                    pairs_pos.push(pair);
                                }
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
                        let fields: Vec<String> = line
                            .split_whitespace()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>();
                        if fields.len() >= 7 {
                            let entry: CoNLLEntry = CoNLLEntry {
                                lemma: fields[2].clone(),
                                pos: fields[4].clone(),
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
    }

    let mut filter: BloomFilter = BloomFilter::with_rate(0.05f32, pairs_pos.len() as u32 * 20u32);

    let mut rng = rand::thread_rng();
    {
        // go through all combinations of similar words
        let similar_max: usize = settings["neg_neighbors"]
            .as_u64()
            .expect("Could not get negative neighbor count from file")
            as usize;
        /*let range: Range<usize> = Range::new(0usize, {
            let mut sum: usize = 0usize;
            for i in 0..similar_ct {
                sum += i * i;
            }
            sum
        });*/
        let range: Range<usize> = Range::new(0usize, similar_max);

        print!("Processing positive samples\r");

        // open pair file
        let pair_file: File = File::create("./pos").expect("Error creating pair file");
        let mut pair_file: BufWriter<_> = BufWriter::new(pair_file);

        // keep track of top 10 lists for faster lookups
        let mut neighbor_map: HashMap<String, Vec<WordSimilarity>> = HashMap::new();

        let target: usize = pairs_pos.len();
        let mut ct: usize = 0usize;
        let mut iter: Iter<NAdjPair> = pairs_pos.iter();
        while let Some(pair) = iter.next() {
            // don't bother with unknown words
            if let (Some(_), Some(_)) = (
                embedding_model.embedding(&(pair.noun)),
                embedding_model.embedding(&(pair.adj)),
            ) {
                // write pair to file for MI analysis
                let pair_string: String = format!("{}\t{}\n", &(pair.noun), &(pair.adj));
                pair_file
                    .write_all(pair_string.as_bytes())
                    .expect("Error writing adjective/noun pair to file");

                let similar_ct: usize = range.ind_sample(&mut rng);

                // removing here to use the vec directly; the map will be "updated" with the same value afterward anyway
                let similar_nouns: Vec<WordSimilarity> = match neighbor_map.remove(&(pair.noun)) {
                    Some(v) => v,
                    None => match embedding_model.similarity(&(pair.noun), similar_ct) {
                        Some(list) => list,
                        None => Vec::new(),
                    },
                };
                let similar_adjs: Vec<WordSimilarity> = match neighbor_map.remove(&(pair.adj)) {
                    Some(v) => v,
                    None => match embedding_model.similarity(&(pair.adj), similar_ct) {
                        Some(list) => list,
                        None => Vec::new(),
                    },
                };

                // scope this so that the handles are closed before we add back to the map
                {
                    let mut similar_noun_iter: Iter<WordSimilarity> = similar_nouns.iter();
                    while let Some(noun) = similar_noun_iter.next() {
                        let mut similar_adj_iter: Iter<WordSimilarity> = similar_adjs.iter();
                        while let Some(adj) = similar_adj_iter.next() {
                            // add the combinations to the filter
                            let concatenation: String = format!("{} {}", &(adj.word), &(noun.word));
                            filter.insert(&concatenation);
                        }
                    }

                    let concatenation: String = format!("{} {}", &(pair.adj), &(pair.noun));
                    filter.insert(&concatenation);
                }

                neighbor_map.insert(pair.noun.clone(), similar_nouns);
                neighbor_map.insert(pair.adj.clone(), similar_adjs);
            }
            ct += 1;
            print!("Processed {} of {} positive samples\r", ct, target);
        }
        println!("");
    }

    // generate negative samples
    let target: usize = pairs_pos.len();
    let mut pairs_neg: Vec<NAdjPair> = Vec::with_capacity(target);
    print!("Generating negative samples\r");
    for ct in 0..target {
        let range: Range<usize> = Range::new(0usize, pairs_pos.len());

        // keep trying new combinations until an invalid one is found
        let mut noun: String;
        let mut adj: String;
        while {
            // do
            noun = pairs_pos[range.ind_sample(&mut rng)].noun.clone();
            adj = pairs_pos[range.ind_sample(&mut rng)].adj.clone();
            let concatenation: String = format!("{} {}", &adj, &noun);

            // while
            filter.contains(&concatenation)
        } {}

        pairs_neg.push(NAdjPair {
            noun: noun.clone(),
            adj: adj.clone(),
        });

        print!("Generated {} of {} negative samples\r", ct, target);
    }
    println!("");

    {
        // open pair file
        let pair_file: File = File::create("./neg").expect("Error creating pair file");
        let mut pair_file: BufWriter<_> = BufWriter::new(pair_file);

        let mut iter: Iter<NAdjPair> = pairs_neg.iter();
        while let Some(pair) = iter.next() {
            // write pair to file for MI analysis
            let pair_string: String = format!("{}\t{}\n", &(pair.noun), &(pair.adj));
            pair_file
                .write_all(pair_string.as_bytes())
                .expect("Error writing adjective/noun pair to file");
        }
    }
    println!("Wrote negative samples to file");
}
