use crate::bayes_net::BayesNet;
use crate::exact_inference::enumeration_ask;
use crate::gibbs_sampling::gibbs_ask;
use crate::like_weighting::likelihood_weighting;
use crate::reject_sampling::rejection_sampling;
use crate::Config;
use std::time::Instant;

const STEP_SIZE: u32 = 10;
const MEAN_SIZE: u32 = 30;

pub fn test_inf(test: &str, config: &Config, net: &BayesNet) {
    let exact_inf = enumeration_ask(&config.query, &config.evidences, &net);
    println!("\nExact Inference distribution = {:?}", exact_inf);
    let mut num_samples: u32 = 1;

    let now = Instant::now();
    loop {
        let mut result_ave: Vec<f64> = Vec::new();
        for _ in 0..MEAN_SIZE {
            match test {
                "rejection" => {
                    let result =
                        rejection_sampling(&config.query, &config.evidences, &net, num_samples);
                    result_ave.push(result[0]);
                }
                "likelihood" => {
                    let result =
                        likelihood_weighting(&config.query, &config.evidences, &net, num_samples);
                    result_ave.push(result[0]);
                }
                "gibbs" => {
                    let result = gibbs_ask(&config.query, &config.evidences, &net, num_samples);
                    result_ave.push(result[0]);
                }
                _ => unreachable!(),
            }
        }
        let mean: f64 = result_ave.iter().sum::<f64>() / (MEAN_SIZE as f64);
        let diff: f64 = (exact_inf[0] - mean).abs();
        if diff < 0.01 * exact_inf[0] {
            println!(
                "{} samples necessary to be within 1% of exact value",
                num_samples
            );
            println!(
                "{} sampling distribution = {:?}",
                test,
                vec![mean, 1.0 - mean]
            );
            break;
        }
        num_samples += STEP_SIZE;
    }
    println!("{} seconds elapsed", now.elapsed().as_secs_f64());
}
