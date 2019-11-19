use crate::bayes_net::BayesNet;
use crate::CHOICES;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::collections::HashMap;

fn prior_sample(net: &BayesNet, sample: &mut HashMap<String, usize>) {
    let mut rng = rand::thread_rng();

    for xi in net.get_ordered_variables() {
        let cpt_row = net.get_cpt_row(xi, &sample);
        let dist = WeightedIndex::new(cpt_row).unwrap();
        sample.insert(xi.to_string(), CHOICES[dist.sample(&mut rng)]);
    }
}

pub fn rejection_sampling(
    query: &String,
    evidences: &HashMap<String, usize>,
    net: &BayesNet,
    num_samples: u32,
) -> Vec<f64> {
    let mut counts: Vec<f64> = vec![0.0, 0.0];
    let mut sample: HashMap<String, usize> = HashMap::new();
    for _ in 0..num_samples {
        prior_sample(net, &mut sample);
        let is_consistent = evidences.iter().all(|(k1, v1)| {
            let v = sample.get(k1).unwrap();
            *v == *v1
        });
        if is_consistent {
            counts[*sample.get(query).unwrap()] += 1.0;
        }
    }
    // Normalization
    let sum: f64 = counts.iter().sum();
    counts.iter().map(|x| *x / sum).collect()
}
