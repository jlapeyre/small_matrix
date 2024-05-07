use num_complex::Complex64;
use ndarray::{ArrayView2, ArrayViewMut2};

use crate::matmul::{Index2, Assign, MyNrows, MyZero};

//
// ndarray implementations
//

impl MyZero for Complex64 {
    type Output = Complex64;
    #[inline(always)]
    fn my_zero() -> Complex64 {
        Complex64::new(0.0, 0.0)
    }
}

impl<A> MyNrows for ArrayView2<'_, A> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols()
    }
}

impl<A> MyNrows for ArrayViewMut2<'_, A> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols()
    }
}

impl<T: Copy> Index2<T> for ArrayView2<'_, T> {
    #[inline(always)]
    fn index2(&self, a: usize, b: usize) -> T {
        self[[a, b]]
    }
}

impl<T> Assign<T> for ArrayViewMut2<'_, T> {
    fn assign(&mut self, a: usize, b: usize, val: T) {
        self[[a, b]] = val;
    }
}
