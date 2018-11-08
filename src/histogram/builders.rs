use ndarray::prelude::*;
use ndarray::Data;
use num_traits::{FromPrimitive, NumOps};
use super::super::{QuantileExt, QuantileExt1d};
use super::super::interpolate::Nearest;
use super::{Edges, Bins};

pub trait BinsBuildingStrategy<T>
    where
        T: Ord
{
    fn from_array(array: ArrayView1<T>) -> Self;

    fn build(&self) -> Bins<T>;

    fn n_bins(&self) -> usize;
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

enum SturgesOrFD<T> {
    Sturges(Sturges<T>),
    FreedmanDiaconis(FreedmanDiaconis<T>),
}

pub struct Auto<T> {
    builder: SturgesOrFD<T>,
}

impl<T> EquiSpaced<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    pub fn new(n_bins: usize, min: T, max: T) -> Self
    {
        Self { n_bins, min, max }
    }

    pub fn build(&self) -> Bins<T> {
        let edges = match self.n_bins {
            0 => Edges::from(vec![]),
            1 => {
                Edges::from(
                    vec![self.min.clone(), self.max.clone()]
                )
            },
            _ => {
                let bin_width = self.bin_width();
                let mut edges: Vec<T> = vec![];
                for i in 0..(self.n_bins+1) {
                    let edge = self.min.clone() + T::from_usize(i).unwrap()*bin_width.clone();
                    edges.push(edge);
                }
                Edges::from(edges)
            },
        };
        Bins::new(edges)
    }

    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    pub fn bin_width(&self) -> T {
        let range = self.max.clone() - self.min.clone();
        let bin_width = range / T::from_usize(self.n_bins).unwrap();
        bin_width
    }
}

impl<T> BinsBuildingStrategy<T> for Sqrt<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array(a: ArrayView1<T>) -> Self
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

    fn n_bins(&self) -> usize {
        self.builder.n_bins()
    }
}

impl<T> Sqrt<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy<T> for Rice<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array(a: ArrayView1<T>) -> Self
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

    fn n_bins(&self) -> usize {
        self.builder.n_bins()
    }
}

impl<T> Rice<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy<T> for Sturges<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array(a: ArrayView1<T>) -> Self
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

    fn n_bins(&self) -> usize {
        self.builder.n_bins()
    }
}

impl<T> Sturges<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy<T> for FreedmanDiaconis<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array(a: ArrayView1<T>) -> Self
    {
        let n_bins = a.len();

        let mut a_copy = a.to_owned();
        let first_quartile = a_copy.quantile_mut::<Nearest>(0.25);
        let third_quartile = a_copy.quantile_mut::<Nearest>(0.75);
        let iqr = third_quartile - first_quartile;

        let bin_width = FreedmanDiaconis::compute_bin_width(n_bins, iqr);
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

    fn n_bins(&self) -> usize {
        self.builder.n_bins()
    }
}

impl<T> FreedmanDiaconis<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn compute_bin_width(n_bins: usize, iqr: T) -> T
    {
        let denominator = (n_bins as f64).powf(1. / 3.);
        let bin_width = T::from_usize(2).unwrap() * iqr / T::from_f64(denominator).unwrap();
        bin_width
    }

    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy<T> for Auto<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array(a: ArrayView1<T>) -> Self
    {
        let fd_builder = FreedmanDiaconis::from_array(a.view());
        let sturges_builder = Sturges::from_array(a.view());
        let builder = {
            if fd_builder.bin_width() > sturges_builder.bin_width() {
                SturgesOrFD::Sturges(sturges_builder)
            } else {
                SturgesOrFD::FreedmanDiaconis(fd_builder)
            }
        };
        Self { builder }
    }

    fn build(&self) -> Bins<T> {
        // Ugly
        match &self.builder {
            SturgesOrFD::FreedmanDiaconis(b) => b.build(),
            SturgesOrFD::Sturges(b) => b.build(),
        }
    }

    fn n_bins(&self) -> usize {
        // Ugly
        match &self.builder {
            SturgesOrFD::FreedmanDiaconis(b) => b.n_bins(),
            SturgesOrFD::Sturges(b) => b.n_bins(),
        }
    }
}

impl<T> Auto<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    pub fn bin_width(&self) -> T {
        // Ugly
        match &self.builder {
            SturgesOrFD::FreedmanDiaconis(b) => b.bin_width(),
            SturgesOrFD::Sturges(b) => b.bin_width(),
        }
    }
}
