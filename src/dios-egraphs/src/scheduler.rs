use egg::{Extractor, Id, Rewrite, RewriteScheduler};

use crate::{
    cost::VecCostFn,
    tracking::TrackRewrites,
    veclang::{EGraph, VecLang},
};

/// Stores a map from rule name (string) and iteration number (usize)
/// to the number of times that rule was applied in that iteration.
pub struct LoggingScheduler {
    root: Id,
}

impl LoggingScheduler {
    pub fn new(root: Id) -> Self {
        LoggingScheduler { root }
    }
}

impl RewriteScheduler<VecLang, TrackRewrites> for LoggingScheduler {
    fn apply_rewrite(
        &mut self,
        _iteration: usize,
        egraph: &mut EGraph,
        rewrite: &Rewrite<VecLang, TrackRewrites>,
        matches: Vec<egg::SearchMatches<VecLang>>,
    ) -> usize {
        let (bef_cost, _) = {
            let extractor = Extractor::new(&egraph, VecCostFn { egraph });
            extractor.find_best(self.root)
        };
        let bef_size = egraph.total_size();
        let bef_classes = egraph.number_of_classes();

        // apply the rule
        let applications = rewrite.apply(egraph, &matches);
        // for id in &applications {
        //     // egraph[*id].data.insert(rewrite.name().to_string());
        //     egraph[*id].data.push(rewrite.name().to_string());
        // }

        let extractor = Extractor::new(&egraph, VecCostFn { egraph });
        let (aft_cost, _) = extractor.find_best(self.root);
        let aft_size = egraph.total_size();
        let aft_classes = egraph.number_of_classes();

        let diff_classes = if aft_classes < bef_classes {
            format!("-{}", bef_classes - aft_classes)
        } else {
            format!("{}", aft_classes - bef_classes)
        };

        eprintln!(
            "Rewrite {} cost {} total {} eclasses {}",
            rewrite.name(),
            bef_cost - aft_cost,
            aft_size - bef_size,
            diff_classes
        );

        // return the number of applications
        applications.len()
    }
}

#[derive(Default)]
pub struct LoggingData;

impl<L: egg::Language, N: egg::Analysis<L>> egg::IterationData<L, N> for LoggingData {
    fn make(_runner: &egg::Runner<L, N, Self>) -> Self {
        eprintln!("iter");
        LoggingData
    }
}
