use rand::distributions::{Uniform, WeightedIndex};
use rand::prelude::*;
use std::collections::HashMap;

use crate::bayes_net::BayesNet;
use crate::CHOICES;

pub fn likelihood_weighting(
    query: &String,
    evidence: &HashMap<String, usize>,
    net: &BayesNet,
    n: u32,
) -> Vec<f64> {
    let mut big_w: Vec<f64> = vec![0.0, 0.0];
    for _ in 1..n {
        let (x, w) = weighted_sample(net, evidence);
        let index = x.get(query).unwrap();
        big_w[*index] += w;
    }
    let sum: f64 = big_w.iter().sum();
    big_w.iter().map(|x| *x / sum).collect()
}

fn weighted_sample(
    net: &BayesNet,
    evidence: &HashMap<String, usize>,
) -> (HashMap<String, usize>, f64) {
    let mut rng = rand::thread_rng();
    let die = Uniform::from(0..=1);
    let mut w = 1.0;
    let mut x = evidence.clone();

    for key in net.get_ordered_variables() {
        if evidence.contains_key(key) {
            continue;
        } else {
            x.insert(key.to_string(), die.sample(&mut rng) as usize);
        }
    }
    for x_i in net.get_ordered_variables() {
        match evidence.get(x_i) {
            Some(xi) => {
                w *= net.get_cpt_row(x_i, &x)[*xi];
            }
            None => {
                let dist = WeightedIndex::new(net.get_cpt_row(x_i, &x)).unwrap();
                x.insert(x_i.to_string(), CHOICES[dist.sample(&mut rng)]);
            }
        }
    }
    (x, w)
}
