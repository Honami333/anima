use candle_core::{Device, Tensor, Var};
use candle_nn::{Optimizer, SGD};

use candle_nn::loss;

mod parse;

const LEARNING_TEXT: &str = "Результат обучения:";

const GENERATIONS: u32 = 1000;
const SGD_CONFIG: f64 = 0.15;

const CONTEXT_SIZE: usize = 5;

const START_SIZE: usize = 5;

const LAYER_COUNT: usize = 3;
const BRAIN_SIZE: usize = 1024;


fn main() {
    let parse_data = match parse::parse_learning_file() {
        Ok(data) => Some(data),
        Err(error) => {
            println!("{:?}", error);
            None
        }
    };

    let mut matrix_data = parse::MatrixData {
        inputs: Vec::new(),
        targets: Vec::new(),
    };

    let parse_data = parse_data.unwrap();

    parse::compact_matrix_data(&mut matrix_data, &parse_data, CONTEXT_SIZE);

    if let Err(error) = learning(&matrix_data, &parse_data) {
        println!("{:?}", error);
    };
}

fn learning(matrix_data: &parse::MatrixData, parse_data: &parse::ParseData) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let vocab_size = parse_data.len;

    let compact = create_compact_tensor(&device, &matrix_data.inputs, vocab_size)?;

    let true_tensor = Tensor::from_slice(&matrix_data.targets, matrix_data.targets.len(), &device)?;

    learning_process(&compact, &true_tensor)?;
    print_result_learning(&compact, parse_data, LEARNING_TEXT)?;

    Ok(())
}

#[derive(Debug, Clone)]
struct TensorCompact {
    matrix: Tensor,
    layers: Vec<Layer>
}

#[derive(Debug, Clone)]
struct Layer {
    var: Var,
    bias: Var,
}

fn create_compact_tensor(
    device: &Device,
    matrix_data: &[f32],
    vocab_size: usize,
) -> Result<TensorCompact, Box<dyn std::error::Error>> {
    let rows = matrix_data.len() / (START_SIZE * vocab_size);
    let shape = (rows, START_SIZE * vocab_size);
    let matrix = Tensor::from_slice(matrix_data, shape, device)?;

    let mut layers = Vec::new();

    let layer1 = Layer {
        var: create_var(device, (START_SIZE * vocab_size, BRAIN_SIZE))?,
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
        matrix,
        layers
    };

    Ok(compact)
}

fn create_var(device: &Device, paramset: (usize, usize)) -> Result<Var, Box<dyn std::error::Error>> {
    let var = Var::randn(0.0_f32, 0.1_f32, paramset, device)?;

    Ok(var)
}

fn create_bias(device: &Device, paramset: (usize, usize)) -> Result<Var, Box<dyn std::error::Error>> {
    let bias_tensorshape = Tensor::zeros(paramset, candle_core::DType::F32, device)?;
    let bias = Var::from_tensor(&bias_tensorshape)?;

    Ok(bias)
}

fn learning_process(compact: &TensorCompact, true_tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let mut sgd_vec = Vec::new();

    for layer in compact.layers.iter() {
        sgd_vec.push(layer.var.clone());
        sgd_vec.push(layer.bias.clone());
    };

    let mut sgd = SGD::new(sgd_vec, SGD_CONFIG)?;

    for i in 0..GENERATIONS {
        let matrix = all_layer_process(compact)?;

        let assembly_tensor = loss::cross_entropy(&matrix, true_tensor)?;

        let grads = assembly_tensor.backward()?;

        sgd.step(&grads)?;

        println!("{i}");
    };

    Ok(())
}

fn all_layer_process(compact: &TensorCompact) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut matrix = compact.matrix.clone();
    let layer_len = compact.layers.len();

    for (i, layer) in compact.layers.iter().enumerate() {
        let new_layer = layer_marger(
            &matrix,
            &layer.var,
            &layer.bias
        )?;

        if i == layer_len - 1 {
            matrix = new_layer;
        } else {
            matrix = new_layer.relu()?;
        };
    };

    Ok(matrix)
}

fn layer_marger(matrix: &Tensor, var: &Var, bias: &Var) -> Result<Tensor, Box<dyn std::error::Error>> {
    let layer = matrix.matmul(var.as_tensor())?;

    let layer_offset = layer.broadcast_add(bias.as_tensor())?;

    Ok(layer_offset)
}

fn print_result_learning(
    compact: &TensorCompact,
    parse_data: &parse::ParseData,
    final_text: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let matrix = all_layer_process(compact)?;

    println!("{matrix}");

    let argmax = matrix.argmax(candle_core::D::Minus1)?;
    let predicted_ids = argmax.to_vec1::<u32>()?;

    println!("{final_text}");
    for id in predicted_ids {
        if let Some(ch) = parse_data.id_to_char.get(&(id as usize)) {
            print!("{}", ch);
        };
    };
    println!();

    Ok(())
}