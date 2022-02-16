use std::{collections::HashMap, io::stdout, io::Write};

use egg::{Extractor, Id, RecExpr, Rewrite, RewriteScheduler};

use crate::{
    cost::{cost_differential, VecCostFn},
    tracking::TrackRewrites,
    // tree::{get_rewrites_used, print_rewrites_used},
    veclang::{EGraph, VecLang},
};

/// Stores a map from rule name (string) and iteration number (usize)
/// to the number of times that rule was applied in that iteration.
pub struct LoggingScheduler {
    root: Id,
    #[allow(unused)]
    init_prog: RecExpr<VecLang>,
    pub counts: HashMap<String, HashMap<String, u64>>,
    out_file: Box<dyn Write>,
}

impl LoggingScheduler {
    pub fn new(root: Id, init_prog: RecExpr<VecLang>) -> Self {
        LoggingScheduler {
            root,
            init_prog,
            counts: HashMap::default(),
            out_file: Box::new(stdout()),
        }
    }

    pub fn new_w_write(root: Id, init_prog: RecExpr<VecLang>, out_file: Box<dyn Write>) -> Self {
        LoggingScheduler {
            root,
            init_prog,
            counts: HashMap::default(),
            out_file,
        }
    }

    pub fn write_headers(&mut self) {
        let headers: [&str; 11] = [
            "iteration",
            "name",
            "cd",
            "n_apps",
            "bef_size",
            "bef_class",
            "bef_cost",
            "aft_size",
            "aft_class",
            "aft_cost",
            "diff_cost",
        ];
        writeln!(self.out_file, "{}", headers.join(",")).unwrap();
    }

    pub fn write_row(
        &mut self,
        iteration: usize,
        name: &str,
        cd: f64,
        n_apps: usize,
        bef_size: usize,
        bef_class: i64,
        bef_cost: f64,
        aft_size: usize,
        aft_class: i64,
        aft_cost: f64,
        diff_cost: f64,
    ) {
        writeln!(
            self.out_file,
            "{},{},{},{},{},{},{},{},{},{},{}",
            iteration,
            name,
            cd,
            n_apps,
            bef_size,
            bef_class,
            bef_cost,
            aft_size,
            aft_class,
            aft_cost,
            diff_cost
        )
        .unwrap()
    }
}

impl RewriteScheduler<VecLang, TrackRewrites> for LoggingScheduler {
    fn apply_rewrite(
        &mut self,
        iteration: usize,
        egraph: &mut EGraph,
        rewrite: &Rewrite<VecLang, TrackRewrites>,
        matches: Vec<egg::SearchMatches<VecLang>>,
    ) -> usize {
        egraph.rebuild();
        let (bef_cost, _bef_prog) = {
            let extractor = Extractor::new(&egraph, VecCostFn {});
            extractor.find_best(self.root)
        };
        let bef_size = egraph.total_size();
        let bef_class: i64 = egraph.number_of_classes() as i64;

        // =*=*= apply the rule =*=*=
        let applications = rewrite.apply(egraph, &matches);

        // rebuild the graph so that we can examine the state
        // of the graph after the rule application.
        egraph.rebuild();

        let extractor = Extractor::new(&egraph, VecCostFn {});
        let (aft_cost, _aft_prog) = extractor.find_best(self.root);
        let aft_size = egraph.total_size();
        let aft_class: i64 = egraph.number_of_classes() as i64;

        // let rules = get_rewrites_used(
        //     &egraph
        //         .clone()
        //         .explain_equivalence(&self.init_prog, &aft_prog)
        //         .explanation_trees,
        // );
        // self.counts
        //     .entry(rewrite.name.to_string())
        //     .and_modify(|hm| {
        //         for r in &rules {
        //             hm.entry(r.to_string()).and_modify(|n| *n += 1).or_insert(1);
        //         }
        //     })
        //     .or_insert_with(|| HashMap::from_iter(rules.iter().cloned().map(|r| (r, 1))));

        // print_rewrites_used(&mut self.out_file, " -", &rules);

        self.write_row(
            iteration,
            rewrite.name.as_str(),
            cost_differential(rewrite),
            applications.len(),
            bef_size,
            bef_class,
            bef_cost,
            aft_size,
            aft_class,
            aft_cost,
            bef_cost - aft_cost,
        );

        // return the number of applications
        applications.len()
    }
}

#[derive(Default)]
pub struct LoggingData;

impl<L: egg::Language, N: egg::Analysis<L>> egg::IterationData<L, N> for LoggingData {
    fn make(_runner: &egg::Runner<L, N, Self>) -> Self {
        // eprintln!("==^= iter {} =^==", runner.iterations.len() + 1);
        LoggingData
    }
}
