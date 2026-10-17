//! Kernel di dequant e matvec.
//!
//! Placeholder: qui andranno int4/int8 group-wise come in Colibrì `quant.h` / `idot.h`.

use colibri_core::Result;

/// Dequant grezzo: interpreta `packed` come f32 LE (solo per test / scaffold).
pub fn dequant_f32_le(packed: &[u8], out: &mut [f32]) -> Result<()> {
    let n = out.len().min(packed.len() / 4);
    for i in 0..n {
        let j = i * 4;
        out[i] = f32::from_le_bytes([
            packed[j],
            packed[j + 1],
            packed[j + 2],
            packed[j + 3],
        ]);
    }
    Ok(())
}

/// y = W x  (W row-major, f32), stub seriale.
pub fn matvec_f32(w: &[f32], x: &[f32], y: &mut [f32], cols: usize) {
    let rows = y.len();
    assert_eq!(w.len(), rows * cols);
    assert_eq!(x.len(), cols);
    for r in 0..rows {
        let mut acc = 0.0f32;
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        y[r] = acc;
    }
}
