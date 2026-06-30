use std::fs;

use tokenizers::{Tokenizer, TokenizerBuilder};
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::AddedToken;

use crate::DICTIONARY_SIZE;
use crate::DATASET_PATH;
use crate::TOKENIZER_LIB_PATH;


#[derive(Debug, Clone)]
pub struct MatrixData {
    pub tokenizer: Tokenizer,
    pub inputs: Vec<f32>,
    pub targets: Vec<u32>,
    pub vocab_size: usize,
}

pub fn collections_vec_data(
    context_size: usize,
) -> anyhow::Result<MatrixData> {
    let tokenizer = Tokenizer::from_file(TOKENIZER_LIB_PATH)
        .map_err(anyhow::Error::msg)?;

    let text = fs::read_to_string(DATASET_PATH)?;

    let encoding = tokenizer.encode(text, true)
        .map_err(anyhow::Error::msg)?;
    let token_ids = encoding.get_ids();

    let vocab_size = tokenizer.get_vocab_size(true);

    let mut matrix_data = MatrixData {
        tokenizer,
        inputs: Vec::new(),
        targets: Vec::new(),
        vocab_size
    };

    if token_ids.len() <= context_size {
        return Err(anyhow::anyhow!(
            "Ошибка: Текст в dataset.txt слишком короткий ({}) для контекста CONTEXT_SIZE ({})! Добавь больше текста.", 
            token_ids.len(), 
            context_size
        ));
    }

    for i in 0..(token_ids.len() - context_size) {
        let window = &token_ids[i..i + context_size];

        for &token_id in window {
            matrix_data.inputs.push(token_id as f32);
        };

        let next_token_id = token_ids[i + context_size];
        matrix_data.targets.push(next_token_id);
    };
    
    Ok(matrix_data)
}

pub fn parse_tokenizers() -> anyhow::Result<()> {
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()
        .map_err(anyhow::Error::msg)?;

    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(DICTIONARY_SIZE)
        .special_tokens(vec![
            AddedToken::from(String::from(crate::TOK_BOS), true),
            AddedToken::from(String::from(crate::TOK_PAD), true),
            AddedToken::from(String::from(crate::TOK_EOS), true),
            AddedToken::from(String::from(crate::TOK_UNK), true),
            AddedToken::from(String::from(crate::TOK_USER), true),
            AddedToken::from(String::from(crate::TOK_ANIMA), true),
            AddedToken::from(String::from(crate::TOK_END_USER), true),
            AddedToken::from(String::from(crate::TOK_END_ANIMA), true),
        ])
        .build();

    tokenizer.train_from_files(&mut trainer, vec![DATASET_PATH.to_string()])
        .map_err(anyhow::Error::msg)?;

    tokenizer.save(TOKENIZER_LIB_PATH, true)
        .map_err(anyhow::Error::msg)?;

    println!("Токенизатор успешно обучен и сохранен!");

    Ok(())
}