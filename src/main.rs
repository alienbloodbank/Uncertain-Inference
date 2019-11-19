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

use bayes_net::{get_config, BayesNet};
use exact_inference::enumeration_ask;
use gibbs_sampling::gibbs_ask;
use like_weighting::likelihood_weighting;
use reject_sampling::rejection_sampling;
use std::env;
use std::time::Instant;
use xml_parser::get_xml_contents;

enum Outcome {
    TRUE = 0,
    FALSE = 1,
}

const CHOICES: &[usize] = &[Outcome::TRUE as usize, Outcome::FALSE as usize];

fn main() {
    let config = get_config(env::args());
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

    /* Exact Inference Test */
    let now = Instant::now();
    println!(
        "\n{0}\nExact Inference Ans: {1:?}",
        config.clone(),
        enumeration_ask(&config.query, &config.evidences, &net)
    );
    println!("{} seconds elapsed", now.elapsed().as_secs_f64());

    /* Rejection Sampling Test */
    let now = Instant::now();
    println!(
        "\n{0}\nRejection Sampling Ans: {1:?}",
        config.clone(),
        rejection_sampling(&config.query, &config.evidences, &net, config.num_samples)
    );
    println!("{} seconds elapsed", now.elapsed().as_secs_f64());

    /* Likelihood Weighting Test */
    let now = Instant::now();
    println!(
        "\n{0}\nLikelihood Weighting Ans: {1:?}",
        config.clone(),
        likelihood_weighting(&config.query, &config.evidences, &net, config.num_samples)
    );
    println!("{} seconds elapsed", now.elapsed().as_secs_f64());

    /* Gibbs Sampling Test */
    let now = Instant::now();
    println!(
        "\n{0}\nGibbs Sampling Ans: {1:?}",
        config.clone(),
        gibbs_ask(&config.query, &config.evidences, &net, config.num_samples)
    );
    println!("{} seconds elapsed", now.elapsed().as_secs_f64());
}
