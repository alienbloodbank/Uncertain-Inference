use crate::CHOICES;
use std::collections::{HashMap, VecDeque};
use std::fmt;

pub struct BayesNet {
    pub dag: HashMap<String, Node>,
    pub ordered_nodes: VecDeque<String>,
}

#[derive(Debug)]
pub struct Node {
    pub children: Vec<String>,
    pub parents: Vec<String>,
    pub cps: Vec<f64>,
}

impl BayesNet {
    pub fn new() -> BayesNet {
        BayesNet {
            dag: HashMap::new(),
            ordered_nodes: VecDeque::new(),
        }
    }

    pub fn visit(
        dag: &HashMap<String, Node>,
        variable: &String,
        ordered_nodes: &mut VecDeque<String>,
    ) {
        if ordered_nodes.contains(variable) {
            return ();
        }
        let node = dag.get(variable).unwrap();
        for m in node.children.iter() {
            BayesNet::visit(dag, m, ordered_nodes);
        }
        ordered_nodes.push_front(variable.to_string());
    }

    pub fn order_variables(&mut self) {
        if self.ordered_nodes.is_empty() {
            for variable in self.dag.keys() {
                BayesNet::visit(&self.dag, variable, &mut self.ordered_nodes);
            }
        }
    }

    pub fn add_variable(&mut self, var: String) {
        self.dag.insert(
            var,
            Node {
                children: Vec::new(),
                parents: Vec::new(),
                cps: Vec::new(),
            },
        );
    }

    pub fn add_dependency(&mut self, var: &String, condition: &String) {
        match self.dag.get_mut(var) {
            Some(node) => node.parents.push(condition.to_string()),
            None => (),
        };

        match self.dag.get_mut(condition) {
            Some(node) => node.children.push(var.to_string()),
            None => (),
        };
    }

    pub fn is_variable_valid(&self, var: &String) -> bool {
        self.dag.contains_key(var)
    }

    pub fn get_ordered_variables(&self) -> impl Iterator<Item = &String> {
        self.ordered_nodes.iter()
    }

    pub fn add_cps(&mut self, var: &String, cps: &mut Vec<f64>) {
        match self.dag.get_mut(var) {
            Some(node) => node.cps.append(cps),
            None => (),
        };
    }

    pub fn get_cpt_row(&self, var: &String, observed: &HashMap<String, usize>) -> &[f64] {
        let node = self.dag.get(var).unwrap();
        let index: usize = node
            .parents
            .iter()
            .rev()
            .enumerate()
            .map(|(i, p)| observed.get(p).unwrap() * (1 << i + 1))
            .sum();

        &node.cps[index..(index + 2)]
    }

    pub fn get_markov_blanket_cps(
        &self,
        var: &String,
        observed: &mut HashMap<String, usize>,
    ) -> Vec<f64> {
        let mut markov_blanket_dist = vec![0.0, 0.0];
        markov_blanket_dist.copy_from_slice(self.get_cpt_row(var, observed));
        let node = self.dag.get(var).unwrap();
        // This might not be correct
        let _ = node
            .children
            .iter()
            .map(|c| {
                *observed.get_mut(var).unwrap() = CHOICES[0];
                let child_cpt_row_true = self.get_cpt_row(c, observed);
                *observed.get_mut(var).unwrap() = CHOICES[1];
                let child_cpt_row_false = self.get_cpt_row(c, observed);
                let c_val = observed.get(c).unwrap();
                vec![child_cpt_row_true[*c_val], child_cpt_row_false[*c_val]]
            })
            .fold(markov_blanket_dist.as_mut_slice(), |acc, x| {
                acc[0] *= x[0];
                acc[1] *= x[1];
                acc
            });

        // unnormalized
        markov_blanket_dist
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub file_name: String,
    pub query: String,
    pub num_samples: u32,
    pub evidences: HashMap<String, usize>,
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

pub fn get_config(mut args: std::env::Args) -> Config {
    args.next();

    let num_samples = args.next().unwrap().parse::<u32>().unwrap();
    let file_name = args.next().expect("Didn't get a file");
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
        num_samples,
        evidences,
    }
}
