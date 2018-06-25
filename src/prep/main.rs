use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use std::process::Command;

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
    let root_entry: CoNLLEntry = CoNLLEntry {
        lemma: "[root]".to_string(),
        pos: "[root]".to_string(),
        head: 0u8,
    };

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
            let conll_name: String = (&path).chars().take((&path).len() - 3).collect::<String>();
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
