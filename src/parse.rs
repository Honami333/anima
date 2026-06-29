use std::fs;

use std::collections::HashSet;
use std::collections::HashMap;


#[derive(Debug, Clone)]
pub struct ParseData {
    pub char_to_id: HashMap<char, usize>,
    pub id_to_char: HashMap<usize, char>,
    pub tokenized_text: Vec<usize>,
    pub len: usize,
}

#[derive(Debug, Clone)]
pub struct MatrixData {
    pub inputs: Vec<f32>,
    pub targets: Vec<u32>,
}

pub fn compact_matrix_data(matrix_data: &mut MatrixData, data: &ParseData, context_size: usize) {
    for i in 0..(data.tokenized_text.len() - context_size) {
        let window = &data.tokenized_text[i..i + context_size];

        for &token in window {
            for char_id in 0..data.len {
                if char_id == token {
                    matrix_data.inputs.push(1.0_f32);
                } else {
                    matrix_data.inputs.push(0.0_f32);
                };
            };
        };

        let next_char_id = data.tokenized_text[i + context_size];

        matrix_data.targets.push(next_char_id as u32);
    };
}

pub fn parse_learning_file() -> Result<ParseData, Box<dyn std::error::Error>> {
    let text = fs::read_to_string("src/dataset.txt")?;

    let mut unique_chars  = HashSet::new();

    for ch in text.chars() {
        unique_chars.insert(ch);
    };

    let mut char_to_id = HashMap::new();
    let mut id_to_char = HashMap::new();

    for (i, ch) in unique_chars.iter().enumerate() {
        char_to_id.insert(*ch, i);
        id_to_char.insert(i, *ch);
    };

    let mut tokenized_text = Vec::new();

    for ch in text.chars() {
        let id = char_to_id.get(&ch);

        if let Some(id) = id {
            tokenized_text.push(*id);
        };
    };

    let data = ParseData {
        char_to_id,
        id_to_char,
        tokenized_text,
        len: unique_chars.len(),
    };

    Ok(data)
}