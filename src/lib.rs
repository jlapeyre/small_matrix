pub mod matmul;
pub use crate::matmul::{matmul_4x4, matmul_4x4_literal, matmul_4x4_faer,
                      matmul_4x4_array_view, matmul_nxn, matmul_4x4_checked};
