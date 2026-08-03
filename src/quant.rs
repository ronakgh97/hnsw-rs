//! Implementation will be based on this [Turboquant](https://arxiv.org/pdf/2504.19874)

#[inline(always)]
/// Quantizes a vector of `f32` values into `i8` values using a simple linear online quantization scheme.
/// Returns the scale factor used for quantization.
pub fn quant(vector_f32: &[f32], vector_i8: &mut [i8]) -> f32 {
    assert_eq!(vector_f32.len(), vector_i8.len());
    // find max_abs
    let max = vector_f32.iter().map(|x| x.abs()).fold(0.0, f32::max); // llvm will optimize, i swear

    // return on 0
    if max == 0.0 {
        vector_i8.fill(0);
        return 0.0;
    }

    // find scale
    const MULTIPLIER: f32 = 0.007874015;
    let scale = max * MULTIPLIER; // not sure, what i'm doing

    // i8 = round(f32/scale)
    for (x, q) in vector_f32.iter().zip(vector_i8.iter_mut()) {
        *q = (x / scale).round().clamp(-127.0, 127.0) as i8;
    }

    scale
}

#[inline(always)]
/// Transforms a vector of `i8` values back into `f32` values using the provided `scale` factor.
/// Uses it with [`quant`] to recover the original values `(with some loss)`.
pub fn dequant(vector_i8: &[i8], vector_f32: &mut [f32], scale: f32) {
    assert_eq!(vector_i8.len(), vector_f32.len());

    for (q, x) in vector_i8.iter().zip(vector_f32.iter_mut()) {
        *x = *q as f32 * scale;
    }
}

#[test]
fn test_quant_loss() {
    use crate::utils::VectorGen;

    let vg = VectorGen::generate(2, 1024, None); // random seed

    let mut vector1_i8 = vec![0i8; 1024];
    let mut vector2_i8 = vec![0i8; 1024];

    let vector1_f32 = vg.get(0);
    let vector2_f32 = vg.get(1);

    let scale1 = quant(vector1_f32, &mut vector1_i8);
    let scale2 = quant(vector2_f32, &mut vector2_i8);

    let mut vector1_dequant = vec![0.0; 1024];
    let mut vector2_dequant = vec![0.0; 1024];

    dequant(&vector1_i8, &mut vector1_dequant, scale1);
    dequant(&vector2_i8, &mut vector2_dequant, scale2);

    let max_error = vector1_f32
        .iter()
        .zip(&vector1_dequant)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max error = {}", max_error);

    let mae = vector1_f32
        .iter()
        .zip(&vector1_dequant)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / vector1_f32.len() as f32;

    println!("MAE = {}", mae);

    let mse = vector1_f32
        .iter()
        .zip(&vector1_dequant)
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        / vector1_f32.len() as f32;

    println!("MSE = {}", mse);
}
