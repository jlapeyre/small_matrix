use num_complex::{Complex64};
use ndarray::{aview2, aview_mut2, ArrayViewMut2, Array2, arr2};
use small_matrix_rust::num_complex::{CMatrix4x4, ONE, ZERO, IM};

use small_matrix_rust::matmul::{
    matmul_4x4_row_major,
    matmul_4x4_col_major,
    matmul_nxn_col_major,
};

fn get_arrays() -> [CMatrix4x4; 3] {
    let a = [[ONE; 4]; 4];
    let b = [[IM; 4]; 4];
    let c = [[ZERO; 4]; 4];
    [a, b, c]
}

fn rust_array() -> CMatrix4x4{
    let [a, b, mut c] = get_arrays();
    matmul_4x4_col_major(&mut c, &a, &b);
    c
}

fn ndarray() -> Array2<Complex64>{
    let [a, b, mut c] = get_arrays();
    let a1 = aview2(&a);
    let b1 = aview2(&b);
    let c1 = aview_mut2(&mut c);
    matmul_4x4_col_major(c1, a1, b1);
    arr2(&c)
}

fn mixed() -> Array2<Complex64>{
    let [a, b, mut c] = get_arrays();
    let b1 = aview2(&b);
    let c1 = aview_mut2(&mut c);
    matmul_4x4_row_major(c1, a, b1);
    arr2(&c)
}

fn main() {
    let c1 = rust_array();
    println!("{:?}\n", &c1);
    let c2 = ndarray();
    println!("{:?}\n", &c1);
    let d = c2 - aview2(&c1);
    println!("{:?}\n", d.map(|x| x.norm_sqr()).sum());
    let c3 = mixed();
    let d = c3 - aview2(&c1);
    println!("{:?}\n", d.map(|x| x.norm_sqr()).sum());
}
