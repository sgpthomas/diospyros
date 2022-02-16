use std::{fs::File, io::BufWriter};

use egg::{BackoffScheduler, Extractor, RecExpr, Runner, SimpleScheduler};

use crate::{
    cost::VecCostFn,
    opts,
    rules::LoggingRunner,
    scheduler::{LoggingData, LoggingScheduler},
    tracking::TrackRewrites,
    veclang::{DiosRwrite, EGraph, VecLang},
};

fn report(runner: &Runner<VecLang, TrackRewrites, LoggingData>) {
    let search_time: f64 = runner.iterations.iter().map(|i| i.search_time).sum();
    let apply_time: f64 = runner.iterations.iter().map(|i| i.apply_time).sum();
    let rebuild_time: f64 = runner.iterations.iter().map(|i| i.rebuild_time).sum();
    let total_time: f64 = runner.iterations.iter().map(|i| i.total_time).sum();

    let iters = runner.iterations.len();
    let rebuilds: usize = runner.iterations.iter().map(|i| i.n_rebuilds).sum();

    let eg = &runner.egraph;
    eprintln!("  Runner report");
    eprintln!("  =============");
    eprintln!(
        "    Stop reason: {:?}",
        runner.stop_reason.as_ref().unwrap()
    );
    eprintln!("    Iterations: {}", iters);
    eprintln!(
        "    Egraph size: {} nodes, {} classes, {} memo",
        eg.total_number_of_nodes(),
        eg.number_of_classes(),
        eg.total_size()
    );
    eprintln!(
        "    Rebuilds: {}, {:.2} per iter",
        rebuilds,
        (rebuilds as f64) / (iters as f64)
    );
    eprintln!("    Total time: {}", total_time);
    eprintln!(
        "      Search:  ({:.2}) {}",
        search_time / total_time,
        search_time
    );
    eprintln!(
        "      Apply:   ({:.2}) {}",
        apply_time / total_time,
        apply_time
    );
    eprintln!(
        "      Rebuild: ({:.2}) {}",
        rebuild_time / total_time,
        rebuild_time
    );
}

pub fn init_egraph() -> EGraph {
    let mut init_egraph: EGraph =
        EGraph::new(TrackRewrites::default()).with_explanations_disabled();
    init_egraph.add(VecLang::Num(0));
    init_egraph
}

pub fn do_eqsat(
    rules: &[DiosRwrite],
    egraph: EGraph,
    prog: &RecExpr<VecLang>,
    opts: &opts::Opts,
) -> (f64, RecExpr<VecLang>, EGraph) {
    let mut runner = LoggingRunner::new(Default::default())
        .with_egraph(egraph)
        .with_expr(&prog)
        .with_node_limit(10_000_000)
        .with_time_limit(std::time::Duration::from_secs(opts.timeout as u64));

    // select the scheduler
    runner = match opts.scheduler {
        opts::SchedulerOpt::Simple => runner.with_scheduler(SimpleScheduler),
        opts::SchedulerOpt::Backoff => runner.with_scheduler(BackoffScheduler::default()),
        opts::SchedulerOpt::Logging => {
            if let Some(filename) = &opts.instrument {
                let write = Box::new(BufWriter::with_capacity(
                    1024,
                    File::create(filename).unwrap(),
                ));
                let mut sched = LoggingScheduler::new_w_write(runner.roots[0], prog.clone(), write);
                sched.write_headers();
                runner.with_scheduler(sched)
            } else {
                let sched = LoggingScheduler::new(runner.roots[0], prog.clone());
                runner.with_scheduler(sched)
            }
        }
    };

    eprintln!("Starting run with {} rules", rules.len());
    runner = runner.run(rules);

    eprintln!("Egraph size: {}", runner.egraph.total_size());
    report(&runner);

    // print reason to STDERR.
    eprintln!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );

    // Extract the resulting program
    let (eg, root) = (runner.egraph.clone(), runner.roots[0].clone());
    let extractor = Extractor::new(&eg, VecCostFn {});
    let (cost, prog) = extractor.find_best(root);
    (cost, prog, runner.egraph)
}
