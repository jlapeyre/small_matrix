use num_complex::Complex64;
// use ndarray::{ArrayView2, ArrayViewMut2};
use crate::matmul::{Index2, Assign, MyNrows, MyZero};

//
// rust array implementations
//

impl<A> MyNrows for [[A; 4]; 4] {
    #[inline(always)]
    fn nrows(&self) -> usize {
        4
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        4
    }
}

impl<A> MyNrows for  &[[A; 4]; 4] {
    #[inline(always)]
    fn nrows(&self) -> usize {
        4
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        4
    }
}

impl<A> MyNrows for  &mut [[A; 4]; 4] {
    #[inline(always)]
    fn nrows(&self) -> usize {
        4
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        4
    }
}

impl<T: Copy> Index2<T> for [[T; 4]; 4] {
    #[inline(always)]
    fn index2(&self, a: usize, b: usize) -> T {
        self[a][b]
    }
}

impl<T: Copy> Index2<T> for &[[T; 4]; 4] {
    #[inline(always)]
    fn index2(&self, a: usize, b: usize) -> T {
        self[a][b]
    }
}

impl<T> Assign<T> for &mut [[T; 4]; 4] {
    fn assign(&mut self, a: usize, b: usize, val: T) {
        self[a][b] = val;
    }
}
