use ndarray::prelude::*;
use ndarray::Data;
use num_traits::{FromPrimitive, NumOps};
use super::super::{QuantileExt, QuantileExt1d};
use super::super::interpolate::Nearest;
use super::{Edges, Bins};

pub trait BinsBuilder<T>
    where
        T: Ord
{
    fn from_array<S>(array: ArrayBase<S, Ix1>) -> Self
        where
            S: Data<Elem=T>;

    fn build(&self) -> Bins<T>;
}

pub struct EquiSpaced<T> {
    n_bins: usize,
    min: T,
    max: T,
}

pub struct Sqrt<T> {
    builder: EquiSpaced<T>,
}

pub struct Rice<T> {
    builder: EquiSpaced<T>,
}

pub struct Sturges<T> {
    builder: EquiSpaced<T>,
}

pub struct FreedmanDiaconis<T> {
    builder: EquiSpaced<T>,
}

impl<T> EquiSpaced<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn new(n_bins: usize, min: T, max: T) -> Self
    {
        Self { n_bins, min, max }
    }

    fn build(&self) -> Bins<T> {
        let edges = match self.n_bins {
            0 => Edges::from(vec![]),
            1 => {
                Edges::from(
                    vec![self.min.clone(), self.max.clone()]
                )
            },
            _ => {
                let range = self.max.clone() - self.min.clone();
                let step = range / T::from_usize(self.n_bins).unwrap();
                let mut edges: Vec<T> = vec![];
                for i in 0..(self.n_bins+1) {
                    let edge = self.min.clone() + T::from_usize(i).unwrap()*step.clone();
                    edges.push(edge);
                }
                Edges::from(edges)
            },
        };
        Bins::new(edges)
    }
}

impl<T> BinsBuilder<T> for Sqrt<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array<S>(a: ArrayBase<S, Ix1>) -> Self
        where
            S: Data<Elem=T>,
    {
        let n_elems = a.len();
        let n_bins = (n_elems as f64).sqrt().round() as usize;
        let min = a.min().clone();
        let max = a.max().clone();
        let builder = EquiSpaced::new(n_bins, min, max);
        Self { builder }
    }

    fn build(&self) -> Bins<T> {
        self.builder.build()
    }
}

impl<T> BinsBuilder<T> for Rice<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array<S>(a: ArrayBase<S, Ix1>) -> Self
        where
            S: Data<Elem=T>,
    {
        let n_elems = a.len();
        let n_bins = (2.*n_elems as f64).powf(1./3.).round() as usize;
        let min = a.min().clone();
        let max = a.max().clone();
        let builder = EquiSpaced::new(n_bins, min, max);
        Self { builder }
    }

    fn build(&self) -> Bins<T> {
        self.builder.build()
    }
}

impl<T> BinsBuilder<T> for Sturges<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array<S>(a: ArrayBase<S, Ix1>) -> Self
        where
            S: Data<Elem=T>,
    {
        let n_elems = a.len();
        let n_bins = (n_elems as f64).log2().round() as usize + 1;
        let min = a.min().clone();
        let max = a.max().clone();
        let builder = EquiSpaced::new(n_bins, min, max);
        Self { builder }
    }

    fn build(&self) -> Bins<T> {
        self.builder.build()
    }
}

impl<T> BinsBuilder<T> for FreedmanDiaconis<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array<S>(a: ArrayBase<S, Ix1>) -> Self
        where
            S: Data<Elem=T>,
    {
        let n_bins = a.len();

        let mut a_copy = a.to_owned();
        let first_quartile = a_copy.quantile_mut::<Nearest>(0.25);
        let third_quartile = a_copy.quantile_mut::<Nearest>(0.75);
        let iqr = third_quartile - first_quartile;

        let bin_width = FreedmanDiaconis::bin_width(n_bins, iqr);
        let min = a_copy.min().clone();
        let max = a_copy.max().clone();
        let mut max_edge = min.clone();
        while max_edge < max {
            max_edge = max_edge + bin_width.clone();
        }
        let builder = EquiSpaced::new(n_bins, min, max_edge);
        Self { builder }
    }

    fn build(&self) -> Bins<T> {
        self.builder.build()
    }
}

impl<T> FreedmanDiaconis<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn bin_width(n_bins: usize, iqr: T) -> T
    {
        let denominator = (n_bins as f64).powf(1. / 3.);
        let bin_width = T::from_usize(2).unwrap() * iqr / T::from_f64(denominator).unwrap();
        bin_width
    }
}
