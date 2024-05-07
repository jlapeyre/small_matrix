pub mod matmul;
pub use crate::matmul::{
    matmul_2x2_row_major,
    matmul_2x2_row_major_unrolled,
    matmul_4x4_row_major,
    matmul_4x4_col_major,
    matmul_4x4_checked_row_major,
    matmul_nxn_row_major,
    matmul_nxn_col_major,
};

pub mod ndarray;
pub mod faer;
pub mod rust_array;
pub mod num_complex;

pub mod alternative;
pub use crate::alternative::{
    matmul_4x4_array_view_col_major,
    matmul_4x4_array_view_row_major,
    matmul_nxn_array_view_row_major,
    matmul_nxn_array_view_col_major,
    matmul_4x4_faer_row_major,
    matmul_4x4_faer_col_major,
};

pub mod gates;
