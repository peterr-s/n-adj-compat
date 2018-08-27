use std::collections::HashSet;
use std::collections::VecDeque;

use std::fs::{File, OpenOptions};

use std::io::{BufRead, BufReader, BufWriter, Lines, Write};

use std::iter::FromIterator;

extern crate rand;
use rand::distributions::{IndependentSample, Range};

extern crate console;
use console::{Key, Term};

fn main() -> () {
    // get all nouns and adjectives
    let mut nouns: HashSet<String> = HashSet::new();
    let mut adjs: HashSet<String> = HashSet::new();
    {
        let input: BufReader<_> =
            BufReader::new(File::open("pos").expect("Failed to open input file"));
        let mut lines: Lines<_> = input.lines();
        while let Some(Ok(line)) = lines.next() {
            let fields: Vec<&str> = Vec::from_iter(line.split_whitespace());
            assert!(fields.len() == 2, "Error splitting line: {}", line);
            nouns.insert(fields[0].to_string());
            adjs.insert(fields[1].to_string());
        }
    }
    let mut noun_vec: Vec<String> = Vec::from_iter(nouns.iter().map(|e| e.clone()));
    let mut adj_vec: Vec<String> = Vec::from_iter(adjs.iter().map(|e| e.clone()));

    // create RNG
    let mut rng = rand::thread_rng();
    let mut n_range: Range<usize> = Range::new(0usize, noun_vec.len());
    let mut a_range: Range<usize> = Range::new(0usize, adj_vec.len());

    // open files for writing
    let mut pos_gs: BufWriter<_> = BufWriter::new(
        OpenOptions::new()
        .append(true) // implies .write()
        .create(true)
        .open("pos_gs")
        .expect("Failed to open positive gold standard file for writing"),
    );
    let mut neg_gs: BufWriter<_> = BufWriter::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open("neg_gs")
            .expect("Failed to open negative gold standard file for writing"),
    );

    // process input until user types EOF/Ctrl+D
    let mut noun: String;
    let mut adj: String;
    let undo_buf_len = 10;
    let mut queue: VecDeque<(String, bool)> = VecDeque::with_capacity(undo_buf_len + 1);
    loop {
        // generate pair
        noun = noun_vec[n_range.ind_sample(&mut rng)].clone();
        adj = adj_vec[a_range.ind_sample(&mut rng)].clone();
        print!("{} {}\n", adj, noun);

        while match Term::stdout().read_key() {
            Ok(Key::Char('h')) => {
                // accept
                println!("Accepted");

                queue.push_back((format!("{} {}\n", noun, adj), true));
                if queue.len() > undo_buf_len {
                    pos_gs
                        .write_all(queue.pop_front().unwrap().0.as_bytes())
                        .expect("Could not write to positive sample file");
                }
                false
            }
            Ok(Key::Char('l')) => {
                // reject
                println!("Rejected");

                queue.push_back((format!("{} {}\n", noun, adj), false));
                if queue.len() > undo_buf_len {
                    neg_gs
                        .write_all(queue.pop_front().unwrap().0.as_bytes())
                        .expect("Could not write to negative sample file");
                }
                false
            }
            Ok(Key::Char('j')) => {
                // drop adjective
                if adjs.remove(&adj) {
                    println!("Removed {}", adj);
                }

                adj_vec = Vec::from_iter(adjs.iter().map(|e| e.clone()));
                a_range = Range::new(0usize, adj_vec.len());

                // pick new adjective and retry
                adj = adj_vec[a_range.ind_sample(&mut rng)].clone();
                print!("{} {}\n", adj, noun);
                true
            }
            Ok(Key::Char('k')) => {
                // drop noun
                if nouns.remove(&noun) {
                    println!("Removed {}", noun);
                }

                noun_vec = Vec::from_iter(nouns.iter().map(|e| e.clone()));
                n_range = Range::new(0usize, noun_vec.len());

                // pick new noun and retry
                noun = noun_vec[n_range.ind_sample(&mut rng)].clone();
                print!("{} {}\n", adj, noun);
                true
            }
            Ok(Key::Char('u')) => {
                // undo
                match queue.pop_back() {
                    Some(e) => println!("Undid {:?}", e),
                    None => println!("Nothing to undo"),
                }

                // retry current pair
                true
            }
            Ok(Key::Char('s')) => {
                // skip
                println!("Skipped");

                false
            }
            Ok(Key::Char('\u{4}')) => {
                // clear queues and exit
                let mut iter = queue.iter();
                while let Some(e) = iter.next() {
                    if e.1 {
                        pos_gs
                            .write_all(e.0.as_bytes())
                            .expect("Could not write to positive sample file");
                    } else {
                        neg_gs
                            .write_all(e.0.as_bytes())
                            .expect("Could not write to negative sample file");
                    }
                }

                return;
            }
            Ok(k) => {
                eprintln!("Invalid input ({:?}); try again", k);

                true
            }
            Err(..) => {
                eprintln!("Console error");

                return;
            }
        } {}

        println!("");
    }
}
