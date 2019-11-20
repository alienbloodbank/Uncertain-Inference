use crate::bayes_net::BayesNet;
use crate::exact_inference::enumeration_ask;
use crate::gibbs_sampling::gibbs_ask;
use crate::like_weighting::likelihood_weighting;
use crate::reject_sampling::rejection_sampling;
use crate::Config;

pub fn test_inf(test: &str, config: &Config, net: &BayesNet) {
    let exact_inf = enumeration_ask(&config.query, &config.evidences, &net);
    println!("\nExact Inference probability: {:?}", exact_inf);
    let mut num_samples: u32 = 1;

    loop {
        let mut result_ave: Vec<f64> = Vec::new();
        for _ in 0..30 {
            match test {
                "rejection" => {
                    let result = rejection_sampling(&config.query, &config.evidences, &net, num_samples);
                    result_ave.push(result[0]);
                },
                "likelihood" => {
                    let result = likelihood_weighting(&config.query, &config.evidences, &net, num_samples);
                    result_ave.push(result[0]);
                },
                "gibbs" => {
                    let result = gibbs_ask(&config.query, &config.evidences, &net, num_samples);
                    result_ave.push(result[0]);
                },
                _ => unreachable!()
            }
        }
        let mean: f64 = result_ave.iter().sum::<f64>() / 30.0;
        let diff: f64 = (exact_inf[0] - mean).abs();
        if diff < 0.01 {
            println!("{} sampling mean of True: {}", test, mean);
            println!("Probability cutoff at {} samples", num_samples);
            break;
        }
        num_samples += 1;
    }
}
