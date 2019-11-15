use std::collections::{HashMap, VecDeque};

use std::fmt;
use std::fs::File;

use std::env;
use std::io::prelude::*;

use quick_xml::events::Event;
use quick_xml::Reader;

use rand::distributions::WeightedIndex;
use rand::prelude::*;

use std::time::Instant;

//use std::error::Error;

const TRUE: usize = 0;
const FALSE: usize = 1;

enum Status {
    ExactInference,
    ApproxInference,
}

fn get_xml_contents(file_name: &String, net: &mut BayesNet) {
    let mut file = File::open(file_name).unwrap();
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
            Ok(Event::Start(ref e)) => match e.name() {
                b"VARIABLE" => in_variable_tag = true,
                b"NAME" => in_name_tag = true,
                b"DEFINITION" => in_definition_tag = true,
                b"FOR" => in_for_tag = true,
                b"GIVEN" => in_given_tag = true,
                b"TABLE" => in_table_tag = true,
                _ => (),
            },
            Ok(Event::End(ref e)) => match e.name() {
                b"VARIABLE" => in_variable_tag = false,
                b"NAME" => in_name_tag = false,
                b"DEFINITION" => {
                    in_definition_tag = false;
                    current_var = String::new();
                }
                b"FOR" => in_for_tag = false,
                b"GIVEN" => in_given_tag = false,
                b"TABLE" => in_table_tag = false,
                _ => (),
            },
            Ok(Event::Text(e)) => {
                if in_variable_tag && in_name_tag {
                    let new_variable = e.unescape_and_decode(&reader).unwrap();

                    net.add_variable(new_variable);
                } else if in_definition_tag {
                    if in_for_tag {
                        current_var = e.unescape_and_decode(&reader).unwrap();
                    } else if in_given_tag {
                        let parent_var = e.unescape_and_decode(&reader).unwrap();

                        net.add_dependency(&current_var, &parent_var);
                    } else if in_table_tag {
                        let cps = e.unescape_and_decode(&reader).unwrap();
                        let mut nums: Vec<f64> = cps
                            .split_whitespace()
                            .map(|s| s.parse::<f64>().unwrap())
                            .collect::<Vec<f64>>();

                        net.add_cps(&current_var, &mut nums);
                    }
                }
            }
            Ok(Event::Eof) => break, // exits the loop when reaching end of file
            Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
            _ => (), // There are several other `Event`s we do not consider here
        }

        // if we don't keep a borrow elsewhere, we can clear the buffer to keep memory usage low
        buf.clear();
    }
}

struct BayesNet {
    dag: HashMap<String, Node>,
    ordered_nodes: VecDeque<String>,
}

impl BayesNet {
    fn new() -> BayesNet {
        BayesNet {
            dag: HashMap::new(),
            ordered_nodes: VecDeque::new(),
        }
    }

    fn visit(dag: &HashMap<String, Node>, variable: &String, ordered_nodes: &mut VecDeque<String>) {
        if ordered_nodes.contains(variable) {
            return ();
        }
        let node = dag.get(variable).unwrap();
        for m in node.children.iter() {
            BayesNet::visit(dag, m, ordered_nodes);
        }
        ordered_nodes.push_front(variable.to_string());
    }

    fn order_variables(&mut self) {
        if self.ordered_nodes.is_empty() {
            for variable in self.dag.keys() {
                BayesNet::visit(&self.dag, variable, &mut self.ordered_nodes);
            }
        }
    }

    fn add_variable(&mut self, var: String) {
        self.dag.insert(
            var,
            Node {
                children: Vec::new(),
                parents: Vec::new(),
                cps: Vec::new(),
            },
        );
    }

    fn add_dependency(&mut self, var: &String, condition: &String) {
        match self.dag.get_mut(var) {
            Some(node) => node.parents.push(condition.to_string()),
            None => (),
        };

        match self.dag.get_mut(condition) {
            Some(node) => node.children.push(var.to_string()),
            None => (),
        };
    }

    fn is_variable_valid(&self, var: &String) -> bool {
        self.dag.contains_key(var)
    }

    fn add_cps(&mut self, var: &String, cps: &mut Vec<f64>) {
        match self.dag.get_mut(var) {
            Some(node) => node.cps.append(cps),
            None => (),
        };
    }

    fn get_cpt_row(&self, var: &String, conditions: &HashMap<String, usize>) -> &[f64] {
        let node = self.dag.get(var).unwrap();
        let index: usize = node
            .parents
            .iter()
            .rev()
            .enumerate()
            .map(|(i, p)| conditions.get(p).unwrap() * (1 << i + 1))
            .sum();

        &node.cps[index..(index + 2)]
    }
}

#[derive(Debug)]
struct Node {
    children: Vec<String>,
    parents: Vec<String>,
    cps: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Config {
    file_name: String,
    query: String,
    num_samples: u32,
    evidences: HashMap<String, usize>,
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P( {} |", self.query.clone())?;
        for (k, v) in self.evidences.iter() {
            write!(f, " {0} = {1}", *k, *v == 0)?;
        }
        write!(f, " )")
    }
}

fn get_config(mut args: std::env::Args) -> Config {
    args.next();

    let num_samples = args.next().unwrap().parse::<u32>().unwrap();

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

    Config {
        file_name,
        query,
        num_samples,
        evidences,
    }
}

fn enumerate_all(
    ordered_nodes: &mut VecDeque<String>,
    evidences: &HashMap<String, usize>,
    net: &BayesNet,
) -> f64 {
    match ordered_nodes.pop_front() {
        Some(y) => {
            let cpt_row = net.get_cpt_row(&y, evidences);

            match evidences.get(&y) {
                Some(y_val) => cpt_row[*y_val] * enumerate_all(ordered_nodes, evidences, net),
                None => {
                    let mut temp = 0.0;
                    for y_val in &[TRUE, FALSE] {
                        let mut evidences_y = evidences.clone();
                        evidences_y.insert(y.clone(), *y_val);
                        let mut c_ordered_nodes = ordered_nodes.clone();
                        temp += cpt_row[*y_val]
                            * enumerate_all(&mut c_ordered_nodes, &evidences_y, net);
                    }
                    temp
                }
            }
        }
        None => 1.0,
    }
}

fn enumeration_ask(
    query: &String,
    evidences: &mut HashMap<String, usize>,
    net: &BayesNet,
) -> Vec<f64> {
    let mut distribution: Vec<f64> = Vec::new();
    for xi in &[TRUE, FALSE] {
        let mut evidences_xi = evidences.clone();
        evidences_xi.insert(query.to_string(), *xi);
        let mut c_ordered_nodes = net.ordered_nodes.clone();
        distribution.push(enumerate_all(&mut c_ordered_nodes, &evidences_xi, net));
    }

    // Normalization
    let sum: f64 = distribution.iter().sum();
    distribution.iter().map(|x| x / sum).collect()
}

fn prior_sample(net: &BayesNet) -> HashMap<String, usize> {
    let mut rng = rand::thread_rng();

    let choices = [TRUE, FALSE];

    let mut sample: HashMap<String, usize> = HashMap::new();
    for xi in net.ordered_nodes.iter() {
        let cpt_row = net.get_cpt_row(xi, &sample);

        let dist = WeightedIndex::new(cpt_row).unwrap();

        sample.insert(xi.to_string(), choices[dist.sample(&mut rng)]);
    }
    sample
}

fn rejection_sampling(
    query: &String,
    evidences: &mut HashMap<String, usize>,
    net: &BayesNet,
    num_samples: u32,
) -> Vec<f64> {
    let mut counts: Vec<usize> = vec![0, 0];

    for _ in 0..num_samples {
        let sample = prior_sample(net);
        let is_consistent = evidences.iter().all(|(k1, v1)| {
            let v = sample.get(k1).unwrap();
            *v == *v1
        });
        if is_consistent {
            counts[*sample.get(query).unwrap()] += 1;
        }
    }

    // Normalization
    let sum: usize = counts.iter().sum();
    counts.iter().map(|x| (*x as f64) / (sum as f64)).collect()
}

fn main() {
    let mut config = get_config(env::args());

    let mut net = BayesNet::new();

    get_xml_contents(&config.file_name, &mut net);

    net.order_variables();

    if net.is_variable_valid(&config.query) {
        println!("Query variable exists");
    } else {
        println!("Query variable doesn't exist.");
    }

    let now = Instant::now();
    println!(
        "{0}\nExact Inference Ans: {1:?}",
        config.clone(),
        enumeration_ask(&config.query, &mut config.evidences, &net)
    );
    println!("{} seconds elapsed", now.elapsed().as_secs_f64());

    let now = Instant::now();
    println!(
        "{0}\nRejection Sampling Ans: {1:?}",
        config.clone(),
        rejection_sampling(
            &config.query,
            &mut config.evidences,
            &net,
            config.num_samples
        )
    );
    println!("{} seconds elapsed", now.elapsed().as_secs_f64());
}
