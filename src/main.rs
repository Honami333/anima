use candle_core::{Device, Tensor, Var};
use candle_nn::{Optimizer, AdamW};

use candle_nn::loss;
use candle_nn::ops::softmax;

use crate::parse::MatrixData;

mod communication;
pub mod parse;

pub const DATASET_PATH: &str = "src/dataset.txt";
pub const TOKENIZER_LIB_PATH: &str = "tokenizer_lib.json";

pub const TOK_BOS: &str = "<s>";
pub const TOK_PAD: &str = "<pad>";
pub const TOK_EOS: &str = "</s>";
pub const TOK_UNK: &str = "<unk>";
pub const TOK_USER: &str = "<user>";
pub const TOK_ANIMA: &str = "<anima>";
pub const TOK_END_USER: &str = "</user>";
pub const TOK_END_ANIMA: &str = "</anima>";

const LEARNING_TEXT: &str = "Результат обучения:";
const BAR_WIDTH: usize = 30;

const GENERATIONS: u32 = 8000;

pub const CONTEXT_SIZE: usize = 24;

pub const TEMPERATURE: f64 = 0.15;

pub const DICTIONARY_SIZE: usize = 350;

const ADAM_CONFIG: f64 = 0.0005;

const LAYER_COUNT: usize = 4;
pub const BRAIN_SIZE: usize = 64;


fn main() {
    let mut matrix_data = None;

    let mut compact = None;

    let _ = communication::start_loop_consol(&mut matrix_data, &mut compact);
}

pub fn learning(matrix_data: &parse::MatrixData) -> anyhow::Result<TensorCompact> {
    let device = Device::new_cuda(0)?;
    println!("Устройство успешно создано: {:?}", device);
    
    let compact = create_compact_tensor(&device, matrix_data)?;

    let true_tensor = Tensor::from_slice(&matrix_data.targets, matrix_data.targets.len(), &device)?;

    learning_process(&compact, &true_tensor)?;
    print_result_learning(&compact, matrix_data, LEARNING_TEXT)?;

    Ok(compact)
}

#[derive(Debug, Clone)]
pub struct TensorCompact {
    device: Device,
    matrix: Tensor,
    embeddings: Var,
    layers: Vec<Layer>
}

#[derive(Debug, Clone)]
pub struct Layer {
    var: Var,
    bias: Var,
}

fn create_compact_tensor(
    device: &Device,
    matrix_data: &MatrixData,
) -> anyhow::Result<TensorCompact> {
    let vocab_size = matrix_data.vocab_size;
    let inputs_array: &[f32] = &matrix_data.inputs;
    
    let row = matrix_data.targets.len();

    let shape = (row, CONTEXT_SIZE);
    let matrix = Tensor::from_slice(inputs_array, shape, device)?;

    let shape = (vocab_size, BRAIN_SIZE);
    let embeddings = Var::randn(0.0_f32, 0.2_f32, shape, device)?;

    let mut layers = Vec::new();

    let layer1 = Layer {
        var: create_var(device, (BRAIN_SIZE, BRAIN_SIZE))?,
        bias: create_bias(device, (1, BRAIN_SIZE))?,
    };

    layers.push(layer1);

    for _ in 1..(LAYER_COUNT - 1) {
        layers.push( Layer {
            var: create_var(device, (BRAIN_SIZE, BRAIN_SIZE))?,
            bias: create_bias(device, (1, BRAIN_SIZE))?,
        });
    };

    let final_layer = Layer {
        var: create_var(device, (BRAIN_SIZE, vocab_size))?,
        bias: create_bias(device, (1, vocab_size))?,
    };

    layers.push(final_layer);

    let compact = TensorCompact {
        device: device.clone(),
        matrix,
        embeddings,
        layers
    };

    Ok(compact)
}

fn create_var(device: &Device, paramset: (usize, usize)) -> anyhow::Result<Var> {
    let var = Var::randn(0.0_f32, 0.1_f32, paramset, device)?;

    Ok(var)
}

fn create_bias(device: &Device, paramset: (usize, usize)) -> anyhow::Result<Var> {
    let bias_tensorshape = Tensor::zeros(paramset, candle_core::DType::F32, device)?;
    let bias = Var::from_tensor(&bias_tensorshape)?;

    Ok(bias)
}

fn learning_process(compact: &TensorCompact, true_tensor: &Tensor) -> anyhow::Result<()> {
    let mut adam_vec = Vec::new();

    for layer in compact.layers.iter() {
        adam_vec.push(layer.var.clone());
        adam_vec.push(layer.bias.clone());
    };

    let adam_config = candle_nn::ParamsAdamW {
        lr: ADAM_CONFIG,
        ..candle_nn::ParamsAdamW::default()
    };

    let mut adam = AdamW::new(adam_vec, adam_config)?;

    print!("Прогрес: [");
    for i in 0..GENERATIONS {
        let matrix = all_layer_process(compact)?;

        let assembly_tensor = loss::cross_entropy(&matrix, true_tensor)?;

        let grads = assembly_tensor.backward()?;

        adam.step(&grads)?;

        let progress = (i + 1) as f32 / GENERATIONS as f32;

        let filled_length = (progress * BAR_WIDTH as f32).round() as usize;

        let bar = format!(
            "{}{}",
            "█".repeat(filled_length),
            "░".repeat(BAR_WIDTH - filled_length)
        );

        print!(
            "\rПрогресс: [{}] {:.1}% ({} / {})", 
            bar, 
            progress * 100.0, 
            i + 1, 
            GENERATIONS
        );
    };
    println!("] Готово!\n");

    Ok(())
}

fn all_layer_process(compact: &TensorCompact) -> anyhow::Result<Tensor> {
    let rows = compact.matrix.dim(0)?; 

    let flat_tokens = compact.matrix.reshape(rows * CONTEXT_SIZE)?.to_dtype(candle_core::DType::U32)?;

    let mut matrix = compact.embeddings
        .as_tensor()
        .index_select(&flat_tokens, 0)?
        .reshape((rows, CONTEXT_SIZE, BRAIN_SIZE))?;

    let layer_len = compact.layers.len();

    for (i, layer) in compact.layers.iter().enumerate() {
        if i == layer_len - 1 {
            let last_token_matrix = matrix
                .narrow(1, CONTEXT_SIZE - 1, 1)?
                .squeeze(1)?
                .contiguous()?;

            let new_layer = layer_marger(
                &last_token_matrix,
                &layer.var,
                &layer.bias
            )?;

            matrix = new_layer;
        } else {
            let matrix_t = matrix.transpose(1, 2)?;

            let attention_scores = matrix.matmul(&matrix_t)?;

            let attention_probs = softmax(&attention_scores, candle_core::D::Minus1)?;
            
            let mixed_matrix = attention_probs.matmul(&matrix)?;

            let flat_mixed = mixed_matrix.reshape((rows * CONTEXT_SIZE, BRAIN_SIZE))?;

            let new_layer = layer_marger(
                &flat_mixed,
                &layer.var,
                &layer.bias
            )?;

            let new_layer3d = new_layer.reshape((rows, CONTEXT_SIZE, BRAIN_SIZE))?;

            let activated = new_layer3d.relu()?;

            matrix = activated.add(&matrix)?;
        };
    };

    Ok(matrix)
}

fn layer_marger(matrix: &Tensor, var: &Var, bias: &Var) -> anyhow::Result<Tensor> {
    let layer = matrix.matmul(var.as_tensor())?;

    let layer_offset = layer.broadcast_add(bias.as_tensor())?;

    Ok(layer_offset)
}

fn print_result_learning(
    compact: &TensorCompact,
    matrix_data: &MatrixData,
    final_text: &str
) -> anyhow::Result<()> {
    let matrix = all_layer_process(compact)?;

    println!("{matrix}\n");

    let argmax = matrix.argmax(candle_core::D::Minus1)?;
    let predicted_ids = argmax.to_vec1::<u32>()?;

    println!("{final_text}");
    
    let decoded_text = matrix_data.tokenizer.decode(&predicted_ids, true);

    match decoded_text {
        Ok(text) => println!("{text}"),
        Err(error) => println!("{error}"),
    };

    println!();

    Ok(())
}