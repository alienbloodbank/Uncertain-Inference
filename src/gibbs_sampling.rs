use crate::bayes_net::BayesNet;
use crate::CHOICES;
use rand::distributions::{Uniform, WeightedIndex};
use rand::prelude::*;
use std::collections::HashMap;

pub fn gibbs_ask(
    query: &String,
    evidences: &HashMap<String, usize>,
    net: &BayesNet,
    num_samples: u32,
) -> Vec<f64> {
    let mut counts: Vec<f64> = vec![0.0, 0.0];
    let mut rng = rand::thread_rng();
    let die = Uniform::from(0..=1);

    // Getting the list of unobserved
    // non-evidence variables and initializing a random state
    let mut sample: HashMap<String, usize> = HashMap::new();
    let unobserved = net
        .get_ordered_variables()
        .filter(|&x| {
            if let Some(val) = evidences.get(x) {
                sample.insert(x.to_string(), *val);
                false
            } else {
                sample.insert(x.to_string(), die.sample(&mut rng) as usize);
                true
            }
        })
        .collect::<Vec<&String>>();
    for _ in 0..num_samples {
        for &zi in unobserved.iter() {
            // Get Markov Blanket and sample from it
            let markov_blanket_dist = net.get_markov_blanket_cps(zi, &mut sample);
            let dist = WeightedIndex::new(markov_blanket_dist.as_slice()).unwrap();
            sample.insert(zi.to_string(), CHOICES[dist.sample(&mut rng)]);
            counts[*sample.get(query).unwrap()] += 1.0;
        }
    }
    // Normalization
    let sum: f64 = counts.iter().sum();
    counts.iter().map(|x| *x / sum).collect()
}
