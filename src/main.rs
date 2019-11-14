use std::collections::HashMap;
use std::collections::VecDeque;

use std::fs::File;

use std::io::prelude::*;
use std::env;

use quick_xml::Reader;
use quick_xml::events::Event;

//use std::error::Error;


fn get_xml_contents(file_name : String, dag: &mut HashMap<String, Node>) {
    let mut file = File::open(&file_name[..]).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    
    let mut reader = Reader::from_str(&contents);
    reader.trim_text(true);

    let mut buf = Vec::new();

    let mut in_variable_tag = false;
    let mut in_name_tag = false;
    let mut in_definition_tag = false;
    let mut in_for_tag = false;
    let mut in_given_tag = false;
    let mut in_table_tag = false;
    let mut current_var = String::new();

    loop {
        match reader.read_event(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name() {
                    b"VARIABLE" => in_variable_tag = true,
                    b"NAME" => in_name_tag = true,
                    b"DEFINITION" => in_definition_tag = true,
                    b"FOR" => in_for_tag = true,
                    b"GIVEN" => in_given_tag = true,
                    b"TABLE" => in_table_tag = true,
                    _ => (),
                }
            },
            Ok(Event::End(ref e)) => {
                match e.name() {
                    b"VARIABLE" => in_variable_tag = false,
                    b"NAME" => in_name_tag = false,
                    b"DEFINITION" => {
                        in_definition_tag = false;
                        current_var = String::new();
                    },
                    b"FOR" => in_for_tag = false,
                    b"GIVEN" => in_given_tag = false,
                    b"TABLE" => in_table_tag = false,
                    _ => (),
                }
            },
            Ok(Event::Text(e)) => {
                if in_variable_tag && in_name_tag {
                    dag.insert(e.unescape_and_decode(&reader).unwrap(), Node {
                        children: Vec::new(), 
                        parents: Vec::new(),
                        cpts: Vec::new(),
                    });     
                } else if in_definition_tag {
                    if in_for_tag {
                        current_var = e.unescape_and_decode(&reader).unwrap();
                    } else if in_given_tag {
                        let parent = e.unescape_and_decode(&reader).unwrap();

                        match dag.get_mut(&current_var) {
                            Some(node) => node.parents.push(parent.clone()),
                            None => ()
                        };

                        match dag.get_mut(&parent) {
                            Some(node) => node.children.push(current_var.clone()),
                            None => ()
                        };
                    } else if in_table_tag {
                        match dag.get_mut(&current_var) {
                            Some(node) => {
                                let cpts = e.unescape_and_decode(&reader).unwrap();
                                let mut nums: Vec<f64> = cpts.split_whitespace().map(|s| s.parse::<f64>().unwrap()).collect::<Vec<f64>>();
                                node.cpts.append(&mut nums);
                            },
                            None => ()
                        };
                    }  
                }
            },
            Ok(Event::Eof) => break, // exits the loop when reaching end of file
            Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
            _ => (), // There are several other `Event`s we do not consider here
        }

        // if we don't keep a borrow elsewhere, we can clear the buffer to keep memory usage low
        buf.clear();
    }
}

struct Config {
    file_name: String,
    query: String,
    evidences: HashMap<String, usize>,
}

#[derive(Debug)]
struct Node {
    children: Vec<String>, 
    parents: Vec<String>,
    cpts: Vec<f64>,
}

fn visit(variable: &String, ordered_nodes: &mut VecDeque<String>, dag: &HashMap<String, Node>) -> () {
    if ordered_nodes.contains(variable){
        return ();
    }
    
    match dag.get(variable) {
        Some(node) => {
            for m in node.children.iter() {
                visit(m, ordered_nodes, dag);
            }

            ordered_nodes.push_front(variable.to_string());
        },
        None => panic!("This shouldn't happen")
    };
}

fn get_topological_ordering(dag: &HashMap<String, Node>) -> VecDeque<String> {
    
    let mut ordered_nodes: VecDeque<String> = VecDeque::new();
    for variable in dag.keys() {
        visit(variable, &mut ordered_nodes, dag);
    }
    ordered_nodes
}

fn get_config(mut args: std::env::Args) -> Config {
    args.next();
    
    let file_name = args.next().expect("Didn't get a file");

    let mut evidences: HashMap<String, usize> = HashMap::new();
    
    let query = args.next().expect("Didn't get a query");

    loop {
        let evidence_variable = match args.next() {
            Some(arg) => arg,
            None => break,
        };

        let evidence_value = match args.next() {
            Some(arg) => arg.parse::<bool>().expect("Didn't get a variable value"),
            None => panic!("Didn't get a variable value"),
        };

        evidences.insert(evidence_variable, !evidence_value as usize);
    }

    Config {file_name, query, evidences}
}

fn enumerate_all(ordered_nodes: &mut VecDeque<String>, evidences: &HashMap<String, usize>, dag: &HashMap<String, Node>) -> f64 {
    match ordered_nodes.pop_front() {
        Some(y) => {
            let node = dag.get(&y).unwrap();

            let index = node.parents.iter().rev().enumerate().map(|(i, p)| evidences.get(p).unwrap() * (usize::pow(2, i as u32))).sum::<usize>();

            match evidences.get(&y) {

                Some(y_val) => {
                    node.cpts[index * 2 + y_val] * enumerate_all(ordered_nodes, evidences, dag)
                },
                None => {
                    let mut temp: f64 = 0.0;
                    for y_val in &[0, 1] {
                        let mut evidences_y = evidences.clone();
                        evidences_y.insert(y.clone(), *y_val);
                        let mut c_ordered_nodes = ordered_nodes.clone();
                        temp += node.cpts[index * 2 + y_val] * enumerate_all(&mut c_ordered_nodes, &evidences_y, dag);
                    }
                    temp
                },
            }
        },
        None => 1.0,
    }
}


fn enumeration_ask(query: String, evidences: &mut HashMap<String, usize>, dag: &HashMap<String, Node>) -> Vec<f64> {
    let ordered_nodes = get_topological_ordering(dag);

    let mut distribution: Vec<f64> = Vec::new();
    for xi in &[0, 1] {
        let mut evidences_xi = evidences.clone();
        evidences_xi.insert(query.clone(), *xi);
        let mut c_ordered_nodes = ordered_nodes.clone();
        distribution.push(enumerate_all(&mut c_ordered_nodes, &evidences_xi, dag));
    }
    distribution
}


fn main() {

    let mut config = get_config(env::args());
     
    // Directed Acyclic Graph
    let mut dag: HashMap<String, Node> = HashMap::new();

    get_xml_contents(config.file_name, &mut dag);

    if dag.contains_key(&config.query) {
        println!("Query variable exists");
    } else {
        println!("Query variable doesn't exist.");
    }
    
    println!("Ans: {:?}", enumeration_ask(config.query, &mut config.evidences, &dag));
}
