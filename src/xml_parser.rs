use crate::bayes_net::BayesNet;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs::File;
use std::io::prelude::*;

pub fn get_xml_contents(file_name: &str, net: &mut BayesNet) {
    let mut file = File::open(file_name).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let mut reader = Reader::from_str(&contents);
    reader.trim_text(true);

    let mut buf = Vec::new();

    let mut in_variable_tag = false;
    let mut in_name_tag = false;
    let mut in_definition_tag = false;
    let mut in_for_tag = false;
    let mut in_given_tag = false;
    let mut in_table_tag = false;
    let mut current_var = String::new();

    let mut cps: Vec<f64> = Vec::new();

    loop {
        match reader.read_event(&mut buf) {
            Ok(Event::Start(ref e)) => match e.name() {
                b"VARIABLE" => in_variable_tag = true,
                b"NAME" => in_name_tag = true,
                b"DEFINITION" => in_definition_tag = true,
                b"FOR" => in_for_tag = true,
                b"GIVEN" => in_given_tag = true,
                b"TABLE" => {
                    in_table_tag = true;
                    cps.clear();
                }
                _ => (),
            },
            Ok(Event::End(ref e)) => match e.name() {
                b"VARIABLE" => in_variable_tag = false,
                b"NAME" => in_name_tag = false,
                b"DEFINITION" => {
                    in_definition_tag = false;
                    current_var = String::new();
                }
                b"FOR" => in_for_tag = false,
                b"GIVEN" => in_given_tag = false,
                b"TABLE" => {
                    in_table_tag = false;
                    net.add_cps(&current_var, cps.clone());
                }
                _ => (),
            },
            Ok(Event::Text(e)) => {
                if in_variable_tag && in_name_tag {
                    let new_variable = e.unescape_and_decode(&reader).unwrap();

                    net.add_variable(new_variable);
                } else if in_definition_tag {
                    if in_for_tag {
                        current_var = e.unescape_and_decode(&reader).unwrap();
                    } else if in_given_tag {
                        let parent_var = e.unescape_and_decode(&reader).unwrap();

                        net.add_dependency(&current_var, &parent_var);
                    } else if in_table_tag {
                        let mut nums: Vec<f64> = e
                            .unescape_and_decode(&reader)
                            .unwrap()
                            .split_whitespace()
                            .map(|s| s.parse::<f64>().unwrap())
                            .collect::<Vec<f64>>();

                        cps.append(&mut nums);
                    }
                }
            }
            Ok(Event::Eof) => break, // exits the loop when reaching end of file
            Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
            _ => (), // There are several other `Event`s we do not consider here
        }

        // if we don't keep a borrow elsewhere, we can clear the buffer to keep memory usage low
        buf.clear();
    }
}