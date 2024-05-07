use faer::prelude::*;
use faer_entity::IdentityGroup;
use faer::{Entity};

use crate::matmul::{Index2, Assign, MyNrows, MyZero};

//
// faer implementations
//

impl MyZero for c64 {
    type Output = c64;
    #[inline(always)]
    fn my_zero() -> c64 {
        c64::new(0.0, 0.0)
    }
}

impl<E: Entity > MyNrows for MatRef<'_, E> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols()
    }
}

impl<E: Entity > MyNrows for MatMut<'_, E> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols()
    }
}

impl<T: Copy + Entity<Group = IdentityGroup, Unit=T>> Index2<T> for MatRef<'_, T> {
    #[inline(always)]
    fn index2(&self, a: usize, b: usize) -> T {
        self[(a, b)]
    }
}

impl<T: Entity<Group = IdentityGroup, Unit=T>> Assign<T> for MatMut<'_, T> {
    fn assign(&mut self, a: usize, b: usize, val: T) {
        self[(a, b)] = val;
    }
}
