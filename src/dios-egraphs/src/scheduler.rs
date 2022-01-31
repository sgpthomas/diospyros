use std::{collections::HashMap, iter::FromIterator};

use egg::{Extractor, Id, RecExpr, Rewrite, RewriteScheduler};

use crate::{
    cost::VecCostFn,
    tracking::TrackRewrites,
    tree::{get_rewrites_used, print_rewrites_used},
    veclang::{EGraph, VecLang},
};

/// Stores a map from rule name (string) and iteration number (usize)
/// to the number of times that rule was applied in that iteration.
pub struct LoggingScheduler {
    root: Id,
    #[allow(unused)]
    init_prog: RecExpr<VecLang>,
    pub counts: HashMap<String, HashMap<String, u64>>,
}

impl LoggingScheduler {
    pub fn new(root: Id, init_prog: RecExpr<VecLang>) -> Self {
        LoggingScheduler {
            root,
            init_prog,
            counts: HashMap::default(),
        }
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
        egraph.rebuild();
        let (bef_cost, _bef_prog) = {
            let extractor = Extractor::new(
                &egraph,
                VecCostFn { // egraph
	    },
            );
            extractor.find_best(self.root)
        };
        let bef_size = egraph.total_size();
        let bef_classes: i64 = egraph.number_of_classes() as i64;

        // =*=*= apply the rule =*=*=
        let applications = rewrite.apply(egraph, &matches);

        egraph.rebuild();
        eprintln!("apps({})= {:?}", rewrite.name.as_str(), applications);
        eprintln!("{{");
        for id in &applications {
            let extractor = Extractor::new(
                &egraph,
                VecCostFn { // egraph
	    },
            );
            let (_, l) = extractor.find_best(*id);
            // let ex = &mut egraph.clone().explain_existance(&l);
            // let ex_tree = &ex.explanation_trees;

            eprintln!("{}, {}", id, l.pretty(80));
            // if is_rewrite_used(rewrite.name.as_str(), &ex_tree) {
            // }

            // if let Some(&VecLang::VecAdd(_)) = l.as_ref().last() {
            //     eprintln!("* {}, {}", id, l.pretty(80));
            //     eprintln!("{}", &ex.get_flat_string());
            // }
        }
        eprintln!("}}");

        let extractor = Extractor::new(
            &egraph,
            VecCostFn { // egraph
	},
        );
        let (aft_cost, aft_prog) = extractor.find_best(self.root);
        let aft_size = egraph.total_size();
        let aft_classes: i64 = egraph.number_of_classes() as i64;

        let rules = get_rewrites_used(
            &egraph
                .clone()
                .explain_equivalence(&self.init_prog, &aft_prog)
                .explanation_trees,
        );
        self.counts
            .entry(rewrite.name.to_string())
            .and_modify(|hm| {
                for r in &rules {
                    hm.entry(r.to_string()).and_modify(|n| *n += 1).or_insert(1);
                }
            })
            .or_insert_with(|| HashMap::from_iter(rules.iter().cloned().map(|r| (r, 1))));

        let diff_cost = bef_cost - aft_cost;
        if diff_cost != 0.0 {
            eprintln!("~~~~~~~~~~~~~~~~~~~~~~~~");
            print_rewrites_used("  ", &rules);

            eprintln!(
                "Rewrite {} cost {} total ({} - {}) eclasses {}",
                rewrite.name,
                bef_cost - aft_cost,
                aft_size,
                bef_size,
                aft_classes - bef_classes
            );
            eprintln!("~~~~~~~~~~~~~~~~~~~~~~~~");
        }

        // return the number of applications
        applications.len()
    }

    // fn log(&self) {
    //     eprintln!("{:#?}", self.counts);
    // }
}

#[derive(Default)]
pub struct LoggingData;

impl<L: egg::Language, N: egg::Analysis<L>> egg::IterationData<L, N> for LoggingData {
    fn make(_runner: &egg::Runner<L, N, Self>) -> Self {
        // eprintln!("==^= iter {} =^==", runner.iterations.len() + 1);
        LoggingData
    }
}
