use candle_core::{Device, Tensor, Var};
use candle_nn::{Optimizer, SGD};

use candle_nn::ops::sigmoid;

const TRUE_MATRIX: [f32; 12] = [
    0.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 0.0
];

const LEARNING_TEXT: &str = "Результат обучения:";

const MATRIX_DATA: [f32; 36] = [
    0.0, 0.0, 0.0,    1.0, 0.0, 0.0,     0.0, 1.0, 0.0,      1.0, 1.0, 0.0,
    0.0, 0.0, 0.5,    1.0, 0.0, 0.5,     0.0, 1.0, 0.5,      1.0, 1.0, 0.5,
    0.0, 0.0, 1.0,    1.0, 0.0, 1.0,     0.0, 1.0, 1.0,      1.0, 1.0, 1.0,
];

const MATRIX_DATA_SHAPE: (usize, usize) = (12, 3);
const TRUE_MATRIX_SHAPE: (usize, usize) = (12, 1);

const VAR_ONE_SHAPE: (usize, usize) = (3, 10);
const VAR_TWO_SHAPE: (usize, usize) = (10, 1);

const BIAS_ONE_SHAPE: (usize, usize) = (1, 10);
const BIAS_TWO_SHAPE: (usize, usize) = (1, 1);

const GENERATIONS: u32 = 100000;
const SGD_CONFIG: f64 = 0.2;


fn main() {
    if let Err(error) = learning() {
        println!("{:?}", error);
    };
}

fn learning() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let compact = create_compact_tensor(&device)?;

    let true_tensor = Tensor::from_slice(&TRUE_MATRIX, TRUE_MATRIX_SHAPE, &device)?;

    learning_process(&compact, &true_tensor)?;
    print_result_learning(&compact, LEARNING_TEXT)?;

    Ok(())
}

#[derive(Debug, Clone)]
struct TensorCompact {
    matrix: Tensor,
    var: VarCompact, 
    bias: VarCompact,
}

#[derive(Debug, Clone)]
struct VarCompact {
    one: Var,
    two: Var,
}

fn create_compact_tensor(device: &Device) -> Result<TensorCompact, Box<dyn std::error::Error>> {
    let matrix = Tensor::from_slice(&MATRIX_DATA, MATRIX_DATA_SHAPE, device)?;

    let var = VarCompact {
        one: create_var(device, VAR_ONE_SHAPE)?,
        two: create_var(device, VAR_TWO_SHAPE)?,
    };

    let bias = VarCompact {
        one: create_bias(device, BIAS_ONE_SHAPE)?,
        two: create_bias(device, BIAS_TWO_SHAPE)?,
    };

    let compact = TensorCompact {
        matrix: matrix.clone(),
        var: var.clone(),
        bias: bias.clone(),
    };

    Ok(compact)
}

fn create_var(device: &Device, paramset: (usize, usize)) -> Result<Var, Box<dyn std::error::Error>> {
    let var = Var::randn(0.0_f32, 1.0_f32, paramset, device)?;

    Ok(var)
}

fn create_bias(device: &Device, paramset: (usize, usize)) -> Result<Var, Box<dyn std::error::Error>> {
    let bias_tensorshape = Tensor::zeros(paramset, candle_core::DType::F32, device)?;
    let bias = Var::from_tensor(&bias_tensorshape)?;

    Ok(bias)
}

fn learning_process(compact: &TensorCompact, true_tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let sgd_vec = vec![
        compact.var.one.clone(),
        compact.var.two.clone(),

        compact.bias.one.clone(),
        compact.bias.two.clone()
    ];

    let mut sgd = SGD::new(sgd_vec, SGD_CONFIG)?;

    for _ in 0..GENERATIONS {
        let layer_1_sigmoid = shadow_layer_marger(
            &compact.matrix,
            &compact.var.one,
            &compact.bias.one
        )?;

        let layer_2_sigmoid = final_layer_marger(
            &layer_1_sigmoid,
            &compact.var.two,
            &compact.bias.two
        )?;

        let try_tensor = layer_2_sigmoid.sub(true_tensor)?;

        let sqr_tensor = try_tensor.sqr()?;

        let assembly_tensor = sqr_tensor.mean_all()?;

        let grads = assembly_tensor.backward()?;

        sgd.step(&grads)?;
    };

    Ok(())
}

fn shadow_layer_marger(matrix: &Tensor, var: &Var, bias: &Var) -> Result<Tensor, Box<dyn std::error::Error>> {
    let layer = matrix.matmul(var.as_tensor())?;

    let layer_offset = layer.broadcast_add(bias.as_tensor())?;

    let layer_sigmoid = sigmoid(&layer_offset)?;

    Ok(layer_sigmoid)
}

fn final_layer_marger(layer: &Tensor, var: &Var, bias: &Var) -> Result<Tensor, Box<dyn std::error::Error>> {
    let layer = layer.matmul(var.as_tensor())?;

    let layer_offset = layer.broadcast_add(bias.as_tensor())?;

    let layer_sigmoid = sigmoid(&layer_offset)?;

    Ok(layer_sigmoid)
}

fn print_result_learning(compact: &TensorCompact, final_text: &str) -> Result<(), Box<dyn std::error::Error>> {
    let layer_1_sigmoid = shadow_layer_marger(
        &compact.matrix,
        &compact.var.one,
        &compact.bias.one
    )?;

    let layer_2_sigmoid = final_layer_marger(
        &layer_1_sigmoid,
        &compact.var.two,
        &compact.bias.two
    )?;

    let flat_tensor = layer_2_sigmoid.flatten_all()?;

    let velues = flat_tensor.to_vec1::<f32>()?;

    println!("{final_text}");
    for val in velues {
        println!("{:.4}", val);
    };

    Ok(())
}