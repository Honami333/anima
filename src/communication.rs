use std::io;

use candle_core::Tensor;
use candle_nn::ops::softmax;

use rand::distributions::{Distribution, WeightedIndex};

use crate::TensorCompact;
use crate::all_layer_process;

use crate::parse;
use crate::parse::MatrixData;

use crate::CONTEXT_SIZE;
use crate::TEMPERATURE;

use crate::learning;

const START_TEXT: &str = "---- Приветствую, напишите свое сообщение для карманного ИИ или введите специальные команды ----";
const LEARNING_TEXT: &str = "---- Начинаю обучение! ----";
const EXIT_TEXT: &str = "---- Прощайте! ----";

const LEARNING_OK_FINISH_TEXT: &str = "---- Обучение полностью завершено! ----";
const LEARNING_ERR_FINISH_TEXT: &str = "---- Критическая ошибка в обучении! ----";

const TOKENIZER_LIB_OK_FINISH_TEXT: &str = "---- Сборка токенов для обучения завершена! ----";
const TOKENIZER_LIB_ERR_FINISH_TEXT: &str = "---- Критическая ошибка в сборке токенов! ----";
const TOKENIZER_LIB_PARSE_ERR_FINISH_TEXT: &str = "---- Критическая ошибка в парсинге токенов! ----";

const TOKENIZER_LIB_PARSE_COMMAND: &str = "anima ~$ token_parse";
const TOKENIZER_LIB_COMMAND: &str = "anima ~$ collect_token";
const LEARNING_COMMAND: &str = "anima ~$ learning";
const EXIT_COMMAND: &str = "anima ~$ exit";

pub fn start_loop_consol(
    matrix_data: &mut Option<MatrixData>,
    compact: &mut Option<TensorCompact>
) -> anyhow::Result<()> {
    println!("{START_TEXT}");

    load_vec_data(matrix_data);
    
    loop {
        let mut input_text = String::new();
        io::stdin().read_line(&mut input_text)?;
        let trimmed = input_text.trim();

        if trimmed == EXIT_COMMAND {
            println!("{EXIT_TEXT}");
            break;
        };

        if trimmed.is_empty() {
            continue;
        };

        if trimmed == LEARNING_COMMAND && let Some(data) = matrix_data {
            println!("{LEARNING_TEXT}");
            
            match learning(data) {
                Ok(comp) => {
                    println!("{LEARNING_OK_FINISH_TEXT}");
                    *compact = Some(comp);
                },
                Err(error) => {
                    println!("{LEARNING_ERR_FINISH_TEXT} {:?}", error);
                }
            };

            continue;
        };

        if trimmed == TOKENIZER_LIB_PARSE_COMMAND {
            if let Err(error) = parse::parse_tokenizers() {
                println!("{TOKENIZER_LIB_PARSE_ERR_FINISH_TEXT} {:?}", error);
            };

            continue;
        };

        if trimmed == TOKENIZER_LIB_COMMAND {
            load_vec_data(matrix_data);
            continue;
        };

        let matrix = matrix_data.clone().unwrap();

        if let Some(compact) = &compact {
            let word_count = input_text.split_whitespace().count() * (CONTEXT_SIZE + crate::BRAIN_SIZE);

            let reply = generate_reply(&matrix, compact, word_count, trimmed);

            if let Ok(rep) = reply {
                print!("anima: {rep}");
            };

            println!();
        };
    };

    Ok(())
}

fn load_vec_data(matrix_data: &mut Option<MatrixData>) {
    match parse::collections_vec_data(CONTEXT_SIZE) {
        Ok(data) => {
            println!("{TOKENIZER_LIB_OK_FINISH_TEXT}");
            *matrix_data = Some(data);
        },
        Err(error) => {
            println!("{TOKENIZER_LIB_ERR_FINISH_TEXT}: {:?}", error);
        }
    };
}

pub fn generate_reply(
    matrix_data: &parse::MatrixData,
    compact: &TensorCompact,
    word_count: usize,
    input_text: &str
) -> anyhow::Result<String> {
    let bos_id = matrix_data.tokenizer.token_to_id(crate::TOK_BOS).unwrap_or(0);
    let pad_id = matrix_data.tokenizer.token_to_id(crate::TOK_PAD).unwrap_or(1);
    let user_id = matrix_data.tokenizer.token_to_id(crate::TOK_USER).unwrap_or(4);
    let end_user_id = matrix_data.tokenizer.token_to_id(crate::TOK_END_USER).unwrap_or(5);
    let anima_id = matrix_data.tokenizer.token_to_id(crate::TOK_ANIMA).unwrap_or(6);

    let encoding = matrix_data.tokenizer.encode(input_text, true)
        .map_err(anyhow::Error::msg)?;
    let user_text_ids = encoding.get_ids();

    let mut token_ids = Vec::new();
    token_ids.push(bos_id);
    token_ids.push(user_id);
    token_ids.extend_from_slice(user_text_ids);
    token_ids.push(end_user_id);
    token_ids.push(anima_id);

    let mut generated_tokens = Vec::new();

    for _ in 0..word_count {
        if token_ids.len() < CONTEXT_SIZE {
            while token_ids.len() < CONTEXT_SIZE {
                token_ids.insert(0, pad_id);
            };
        } else {
            let len = token_ids.len();
            token_ids = token_ids[len - CONTEXT_SIZE..len].to_vec();
        };

        let input_f32: Vec<f32> = token_ids.iter().map(|&x| x as f32).collect();
        let input_tensor = Tensor::from_slice(&input_f32, (1, CONTEXT_SIZE), &compact.device)?;

        let input_compact = TensorCompact {
            device: compact.device.clone(),
            matrix: input_tensor,
            embeddings: compact.embeddings.clone(),
            layers: compact.layers.clone(),
        };

        let logits = all_layer_process(&input_compact)
            .map_err(anyhow::Error::msg)?;

        let scaled_logits = (&logits / TEMPERATURE)?;

        let probs  = softmax(&scaled_logits, candle_core::D::Minus1)?;

        let probs_vec = probs.to_vec2::<f32>()?[0].clone();

        let mut rng = rand::thread_rng();
        let dist = WeightedIndex::new(&probs_vec)?;
        let next_token_id = dist.sample(&mut rng) as u32;

        if matches!(
            matrix_data.tokenizer.id_to_token(next_token_id),
            Some(t) if t == crate::TOK_EOS || t == crate::TOK_END_ANIMA) {
                
            break;
        };

        generated_tokens.push(next_token_id);
        token_ids.push(next_token_id);
    };
    
    let decoded_text = matrix_data.tokenizer.decode(&generated_tokens, true)
        .map_err(anyhow::Error::msg)?;

    Ok(decoded_text)
}