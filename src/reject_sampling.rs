use crate::bayes_net::BayesNet;
use crate::CHOICES;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::collections::HashMap;

fn prior_sample(net: &BayesNet) -> HashMap<String, usize> {
    let mut rng = rand::thread_rng();
    let mut sample: HashMap<String, usize> = HashMap::new();

    for xi in net.get_ordered_variables() {
        let cpt_row = net.get_cpt_row(xi, &sample);
        let dist = WeightedIndex::new(cpt_row).unwrap();
        sample.insert(xi.to_string(), CHOICES[dist.sample(&mut rng)]);
    }
    sample
}

pub fn rejection_sampling(
    query: &String,
    evidences: &HashMap<String, usize>,
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
