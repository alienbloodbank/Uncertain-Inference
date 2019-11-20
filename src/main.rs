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
mod inf_test;

use bayes_net::BayesNet;
use exact_inference::enumeration_ask;
use gibbs_sampling::gibbs_ask;
use like_weighting::likelihood_weighting;
use reject_sampling::rejection_sampling;
use xml_parser::init_net_from_xmlbif;
use inf_test::test_inf;

use std::collections::HashMap;
use std::env;
use std::fmt;
use std::time::Instant;

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

#[derive(Debug, Clone)]
pub struct Config {
    file_name: String,
    query: String,
    inference_type: Status,
    evidences: HashMap<String, usize>,
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P( {} ", self.query.clone())?;
        if self.evidences.is_empty() {
            return write!(f, ")");
        }
        write!(f, "|")?;
        for (k, v) in self.evidences.iter() {
            write!(f, " {0} = {1}", *k, *v == 0)?;
        }
        write!(f, " )")
    }
}

impl Config {
    fn new(mut args: std::env::Args) -> Self {
        args.next();

        let inference_type_decider = args.next().unwrap();

        let (file_name, inference_type) = match inference_type_decider.parse::<u32>() {
            Ok(num_samples) => {
                let file_name = args.next().expect("Didn't get a file");
                (file_name, Status::ApproxInference(num_samples))
            }
            Err(_) => {
                let file_name = inference_type_decider;
                (file_name, Status::ExactInference)
            }
        };

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
            inference_type,
            evidences,
        }
    }
}

mod test_inference {
    use super::*;

    pub fn exact_inference<F>(config: &Config, net: &BayesNet, test_name: &str, f: F)
    where
        F: Fn(&String, &HashMap<String, usize>, &BayesNet) -> Vec<f64>,
    {
        let now = Instant::now();
        println!(
            "\n{0}\n{1} = {2:?}",
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
            let result = f(&config.query, &config.evidences, net, num_samples);
            if result[0].is_nan() {
                println!("Insufficient counts to form distribution. Try increasing number of samples.");
            } else {
                println!(
                    "\n{0}\n{1} = {2:?}",
                    test_name,
                    config,
                    result
                );
            }
        }
        println!("{} seconds elapsed", now.elapsed().as_secs_f64());
    }
}


fn main() {
    let mut config = Config::new(env::args());

    let mut net = BayesNet::new();

    init_net_from_xmlbif(&config.file_name, &mut net);

    if net.is_variable_valid(&config.query) {
        println!("Query variable exist");
    } else {
        eprintln!("Query variable doesn't exist.\nExiting...");
        std::process::exit(1);
    }

    if config.evidences.keys().all(|e| net.is_variable_valid(e)) {
        println!("Evidence list is valid");
    } else {
        eprintln!("Evidence list is not valid.\nExiting...");
        std::process::exit(1);
    }

    // Topological sorting
    net.order_variables();

    let now = Instant::now();
    test_inf("rejection", &config, &net);
    println!("{} seconds elapsed", now.elapsed().as_secs_f64());

    let now = Instant::now();
    test_inf("likelihood", &config, &net);
    println!("{} seconds elapsed", now.elapsed().as_secs_f64());
    let now = Instant::now();
    test_inf("gibbs", &config, &net);
    println!("{} seconds elapsed", now.elapsed().as_secs_f64());

//    match config.inference_type {
//        Status::ExactInference => {
//            /* Exact Inference Test */
//            test_inference::exact_inference(
//                &config,
//                &net,
//                "Inference by Enumeration",
//                enumeration_ask,
//            );
//        }
//        Status::ApproxInference(_) => {
//            /* Rejection Sampling Test */
//            test_inference::approx_inference(
//                &config,
//                &net,
//                "Rejection Sampling",
//                rejection_sampling,
//            );
//
//            /* Likelihood Weighting Test */
//            test_inference::approx_inference(
//                &config,
//                &net,
//                "Likelihood Weighting",
//                likelihood_weighting,
//            );
//
//            /* Gibbs Sampling Test */
//            test_inference::approx_inference(&config, &net, "Gibbs Sampling", gibbs_ask);
//        }
//    };
}
