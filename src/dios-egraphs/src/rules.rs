use egg::{rewrite as rw, BackoffScheduler, Extractor, RecExpr, Runner, SimpleScheduler};

use crate::{
    cost::VecCostFn,
    external::{external_rules, retain_cost_effective_rules, smart_select_rules},
    handwritten::handwritten_rules,
    opts,
    patterns::gen_patterns,
    scheduler::{LoggingData, LoggingScheduler},
    tracking::TrackRewrites,
    veclang::{DiosRwrite, EGraph, VecLang},
};

#[allow(unused)]
fn filter_applicable_rules(rules: &mut Vec<DiosRwrite>, prog: &RecExpr<VecLang>) {
    let prog_str: String = prog.pretty(80);
    let ops_to_filter = vec!["neg", "sqrt", "/"];
    let unused_ops: Vec<&&str> = ops_to_filter
        .iter()
        .filter(|&op| !prog_str.contains(op))
        .collect();

    let mut dropped = "".to_string();
    rules.retain(|r| {
        let drop = unused_ops.iter().any(|&op| {
            let rule_sr = format!("{:?}", r);
            rule_sr.contains(op)
        });
        if drop {
            dropped = format!("{} {}", dropped, r.name)
        };
        !drop
    });
    if dropped != "" {
        eprintln!("Dropping inapplicable rules:{}", dropped);
    }
}

#[allow(unused)]
fn retain_rules_by_name(rules: &mut Vec<DiosRwrite>, names: &[&str]) {
    rules.retain(|rewrite| names.contains(&rewrite.name.as_str()))
}

fn report(runner: &Runner<VecLang, TrackRewrites, LoggingData>) {
    let search_time: f64 = runner.iterations.iter().map(|i| i.search_time).sum();
    let apply_time: f64 = runner.iterations.iter().map(|i| i.apply_time).sum();
    let rebuild_time: f64 = runner.iterations.iter().map(|i| i.rebuild_time).sum();
    let total_time: f64 = runner.iterations.iter().map(|i| i.total_time).sum();

    let iters = runner.iterations.len();
    let rebuilds: usize = runner.iterations.iter().map(|i| i.n_rebuilds).sum();

    let eg = &runner.egraph;
    eprintln!("Runner report");
    eprintln!("=============");
    eprintln!("  Stop reason: {:?}", runner.stop_reason.as_ref().unwrap());
    eprintln!("  Iterations: {}", iters);
    eprintln!(
        "  Egraph size: {} nodes, {} classes, {} memo",
        eg.total_number_of_nodes(),
        eg.number_of_classes(),
        eg.total_size()
    );
    eprintln!(
        "  Rebuilds: {}, {:.2} per iter",
        rebuilds,
        (rebuilds as f64) / (iters as f64)
    );
    eprintln!("  Total time: {}", total_time);
    eprintln!(
        "    Search:  ({:.2}) {}",
        search_time / total_time,
        search_time
    );
    eprintln!(
        "    Apply:   ({:.2}) {}",
        apply_time / total_time,
        apply_time
    );
    eprintln!(
        "    Rebuild: ({:.2}) {}",
        rebuild_time / total_time,
        rebuild_time
    );
}

pub type LoggingRunner = Runner<VecLang, TrackRewrites, LoggingData>;

/// Run the rewrite rules over the input program and return the best (cost, program)
pub fn run(prog: &RecExpr<VecLang>, opts: &opts::Opts) -> (f64, RecExpr<VecLang>) {
    // let use_only_ruler = true;

    let mut rules: Vec<DiosRwrite> = vec![];

    // add handwritten rules
    if opts.handwritten {
        rules.extend(handwritten_rules(opts.no_ac, opts.no_vec));
    }

    // add external rules
    if let Some(filename) = &opts.rules {
        rules.extend(external_rules(filename));
    }

    // filter out rules that have a cost differential lower than cutoff
    if let Some(cutoff) = opts.cost_filter {
        rules = retain_cost_effective_rules(&rules, opts.no_dup_vars, |x| x > cutoff);
    }

    // if we are using sub programs, generate patterns from the program
    // and then use smart_select_rules to generate rules.
    if opts.sub_prog {
        let mut fancy_rules: Vec<DiosRwrite> = vec![];
        for (i, p) in gen_patterns(prog).into_iter().enumerate() {
            if let Some(r) = smart_select_rules(&rules, &p) {
                let name = format!("fancy_{}", i);
                fancy_rules.push(rw!(name; p => r));
            }
        }
        rules = fancy_rules;
    }

    let mut init_eg: EGraph = EGraph::new(TrackRewrites::default()).with_explanations_enabled();
    init_eg.add(VecLang::Num(0));
    let mut runner = LoggingRunner::new(Default::default())
        .with_egraph(init_eg)
        .with_expr(&prog)
        .with_node_limit(500_000)
        .with_time_limit(std::time::Duration::from_secs(opts.timeout as u64))
        // .with_hook(|runner| {
        //     eprintln!("Egraph big big? {}", runner.egraph.total_size());
        //     eprintln!("Egraph class big? {}", runner.egraph.number_of_classes());
        //     let (eg, root) = (&runner.egraph, &runner.roots[0]);
        //     let extractor = Extractor::new(&eg, VecCostFn { egraph: &eg });
        //     let (cost, _) = extractor.find_best(*root);
        //     eprintln!("Egraph cost? {}", cost);
        //     Ok(())
        // })
        .with_iter_limit(opts.iter_limit);

    // select the scheduler
    runner = match opts.scheduler {
        opts::SchedulerOpt::Simple => runner.with_scheduler(SimpleScheduler),
        opts::SchedulerOpt::Backoff => runner.with_scheduler(BackoffScheduler::default()),
        opts::SchedulerOpt::Logging => {
            let sched = LoggingScheduler::new(runner.roots[0], prog.clone());
            runner.with_scheduler(sched)
        }
    };

    eprintln!("Starting run with {} rules", rules.len());
    runner = runner.run(&rules);

    eprintln!("Egraph size: {}", runner.egraph.total_size());
    report(&runner);

    // print reason to STDERR.
    eprintln!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );

    // if matches!(opts.scheduler, opts::SchedulerOpt::Logging) {
    //     runner.scheduler.log();
    // }

    // Extract the resulting program
    let (eg, root) = (runner.egraph.clone(), runner.roots[0].clone());
    let extractor = Extractor::new(&eg, VecCostFn {});
    let (cost, out_prog) = extractor.find_best(root);
    (cost, out_prog)

    // eprintln!("==== Starting Second Phase Optimization ====");
    // eprintln!("Using {} rules", snd_phase.len());

    // let mut init_eg: EGraph = EGraph::new(TrackRewrites::default()).with_explanations_enabled();
    // init_eg.add(VecLang::Num(0));
    // let mut runner = LoggingRunner::new(Default::default())
    //     .with_egraph(init_eg)
    //     .with_expr(&out_prog)
    //     .with_node_limit(1_000_000)
    //     .with_time_limit(std::time::Duration::from_secs(300))
    //     .with_iter_limit(iter_limit);
    // runner = runner.run(&snd_phase);
    // report(&runner);

    // let (eg, root) = (runner.egraph.clone(), runner.roots[0].clone());

    // // Always add the literal zero
    // let extractor = Extractor::new(&eg, VecCostFn {});
    // let (cost, out_prog2) = extractor.find_best(root);
    // eprintln!("optimized:\n{}", out_prog2.pretty(80));
    // eprintln!("Egraph cost? {}", cost);

    // (cost, out_prog2)
}
