extern crate rust2vec;
use rust2vec::{Embeddings, ReadText, WriteWord2Vec}; // TODO fix helper script so that this can use ReadWord2Vec instead of ReadText

use std::fs::File;
use std::io::{BufReader, BufWriter};

use std::process::Command;

fn main() {
    // train
    Command::new("./train_embeddings.py")
        .status()
        .expect("Error training embeddings");

    // read as text
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
    println!("Read embeddings   ");

    // write back as binary
    print!("Writing embeddings\r");
    {
        let embedding_file: File =
            File::create("./embeddings").expect("Could not open embedding file");
        let mut embedding_file: BufWriter<_> = BufWriter::new(embedding_file);
        embedding_model
            .write_word2vec_binary(&mut embedding_file)
            .expect("Could not write embedding file");
    }
    println!("Wrote embeddings  ");
}
