/// Wrapper around `Array1` that makes sure the elements are in ascending order.
struct Edges<A: Ord> {
    edges: Array1<A>,
}

impl<A: Ord> From<Array1<A>> for Edges<A> {
    fn from(mut edges: Array1<A>) -> Self {
        // sort the array in-place
        Edges { edges }
    }
}

impl<A: Ord> Edges<A> {
    fn view(&self) -> ArrayView1<A> {
        self.edges.view()
    }

    /// Returns the index of the bin containing the given value,
    /// or `None` if none of the bins contain the value.
    fn bin_index(&self, value: &A) -> Option<usize> {
        // binary search for the correct bin
    }

    /// Returns the range of the bin containing the given value.
    fn bin_range(&self, value: &A) -> Option<Range<A>>
    where
        A: Clone,
    {
        let i = self.bin_index(value);
        Range { start: self.edges[i].clone(), end: self.edges[i + 1].clone() }
    }
}

struct HistogramCounts<A: Ord> {
    counts: ArrayD<usize>,
    edges: Vec<Edges<A>>,
}

struct HistogramDensity<A: Ord> {
    density: ArrayD<A>,
    edges: Vec<Edges<A>>,
}

impl<A: Ord> HistogramCounts<A> {
    pub fn new(edges: Vec<Edges<A>>) -> Self {
        let counts = ArrayD::zeros(edges.iter().map(|e| e.len() - 1).collect::<Vec<_>>());
        HistogramCounts { counts, edges }
    }

    pub fn add_observation(observation: ArrayView1<A>) -> Result<(), NotFound> {
        let bin = observation
            .iter()
            .zip(&self.edges)
            .map(|(v, e)| e.bin_index(v).ok_or(NotFound))
            .collect::<Result<Vec<_>, _>>()?;
        self.counts[bin] += 1;
    }
}