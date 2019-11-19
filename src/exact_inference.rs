use crate::bayes_net::BayesNet;
use crate::CHOICES;
use std::collections::{HashMap, VecDeque};

fn enumerate_all(
    ordered_nodes: &mut VecDeque<&String>,
    evidences: &HashMap<String, usize>,
    net: &BayesNet,
) -> f64 {
    match ordered_nodes.pop_front() {
        Some(y) => {
            let cpt_row = net.get_cpt_row(y, evidences);

            match evidences.get(y) {
                Some(y_val) => cpt_row[*y_val] * enumerate_all(ordered_nodes, evidences, net),
                None => {
                    let mut temp = 0.0;
                    for y_val in CHOICES {
                        let mut evidences_y = evidences.clone();
                        evidences_y.insert(y.to_string(), *y_val);
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

pub fn enumeration_ask(
    query: &String,
    evidences: &HashMap<String, usize>,
    net: &BayesNet,
) -> Vec<f64> {
    let mut distribution: Vec<f64> = Vec::new();
    for xi in CHOICES {
        let mut evidences_xi = evidences.clone();
        evidences_xi.insert(query.to_string(), *xi);
        let mut c_ordered_nodes = net.get_ordered_variables().collect();
        distribution.push(enumerate_all(&mut c_ordered_nodes, &evidences_xi, net));
    }
    // Normalization
    let sum: f64 = distribution.iter().sum();
    distribution.iter().map(|x| x / sum).collect()
}
