extern crate rust2vec;
use rust2vec::{Embeddings, ReadText, WriteWord2Vec}; // TODO fix helper script so that this can use ReadWord2Vec instead of ReadText

extern crate serde_json;
use serde_json::Value;

use std::fs::File;
use std::io::{BufReader, BufWriter, Read};

use std::process::Command;

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
    let embedding_path: &str = settings["embedding_file"]
        .as_str()
        .expect("Could not get embedding path from settings");

    // train
    Command::new("./train_embeddings.py")
        .arg(
            settings["text_file"]
                .as_str()
                .expect("Could not get block text path from settings"),
        ).status()
        .expect("Error training embeddings");

    // read as text
    print!("Reading embeddings\r");
    let mut embedding_model: Embeddings;
    {
        let embedding_file: File =
            File::open(embedding_path).expect("Could not open embedding file");
        let mut embedding_file: BufReader<_> = BufReader::new(embedding_file);
        embedding_model =
            Embeddings::read_text(&mut embedding_file).expect("Could not read embedding file");
    }
    println!("Read embeddings   ");

    println!("Normalizing embeddings\r");
    embedding_model.normalize();
    println!("Normalized embeddings ");

    // write back as binary
    print!("Writing embeddings\r");
    {
        let embedding_file: File =
            File::create(embedding_path).expect("Could not open embedding file");
        let mut embedding_file: BufWriter<_> = BufWriter::new(embedding_file);
        embedding_model
            .write_word2vec_binary(&mut embedding_file)
            .expect("Could not write embedding file");
    }
    println!("Wrote embeddings  ");
}
