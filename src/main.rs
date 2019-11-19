/* CSC 442: Intro to AI
 * Spring 2019
 * Project 3: Uncertain Inference
 * Authors: Soubhik Ghosh (netId: sghosh13)
 *          Andrew Sexton (netId: asexton2)
 */

mod bayes_net;
mod exact_inference;
mod gibbs_sampling;
mod like_weighting;
mod reject_sampling;
mod xml_parser;

use bayes_net::*;
use exact_inference::enumeration_ask;
use gibbs_sampling::gibbs_ask;
use like_weighting::likelihood_weighting;
use reject_sampling::rejection_sampling;
use xml_parser::get_xml_contents;
use std::env;
use std::time::Instant;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Status {
    ExactInference,
    ApproxInference(u32),
}

enum Outcome {
    TRUE = 0,
    FALSE = 1,
}

const CHOICES: &[usize] = &[Outcome::TRUE as usize, Outcome::FALSE as usize];

mod test_inference {
    use super::*;

    pub fn exact_inference<F>(config: &Config, net: &BayesNet, test_name: &str, f: F)
        where
            F: Fn(&String, &HashMap<String, usize>, &BayesNet) -> Vec<f64>,
    {
        let now = Instant::now();
        println!(
            "\n{0}\n{1} Ans: {2:?}",
            test_name,
            config,
            f(&config.query, &config.evidences, net)
        );
        println!("{} seconds elapsed", now.elapsed().as_secs_f64());
    }

    pub fn approx_inference<F>(config: &Config, net: &BayesNet, test_name: &str, f: F)
        where
            F: Fn(&String, &HashMap<String, usize>, &BayesNet, u32) -> Vec<f64>,
    {
        let now = Instant::now();
        if let Status::ApproxInference(num_samples) = config.inference_type {
            println!(
                "\n{0}\n{1} Ans: {2:?}",
                test_name,
                config,
                f(&config.query, &config.evidences, net, num_samples)
            );
        }
        println!("{} seconds elapsed", now.elapsed().as_secs_f64());
    }
}

fn main() {
    let config = Config::new(env::args());

    let mut net = BayesNet::new();

    get_xml_contents(&config.file_name, &mut net);

    if net.is_variable_valid(&config.query) {
        println!("Query variable exist");
    } else {
        panic!("Query variable doesn't exist.");
    }

    if config.evidences.keys().all(|e| net.is_variable_valid(e)) {
        println!("Evidence list is valid");
    } else {
        panic!("Evidence list is not valid");
    }

    // Topological sorting
    net.order_variables();

    match config.inference_type {
        Status::ExactInference => {
            /* Exact Inference Test */
            test_inference::exact_inference(
                &config,
                &net,
                "Inference by Enumeration",
                enumeration_ask,
            );
        }
        Status::ApproxInference(_) => {
            /* Rejection Sampling Test */
            test_inference::approx_inference(
                &config,
                &net,
                "Rejection Sampling",
                rejection_sampling,
            );

            /* Likelihood Weighting Test */
            test_inference::approx_inference(
                &config,
                &net,
                "Likelihood Weighting",
                likelihood_weighting,
            );

            /* Gibbs Sampling Test */
            test_inference::approx_inference(&config, &net, "Gibbs Sampling", gibbs_ask);
        }
    };
}