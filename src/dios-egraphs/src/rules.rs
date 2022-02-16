use std::{collections::HashMap, fs::File, io::BufWriter};

use egg::{rewrite as rw, BackoffScheduler, Extractor, RecExpr, Runner, SimpleScheduler};

use crate::{
    cost::VecCostFn,
    eqsat::{self, do_eqsat},
    external::{external_rules, retain_cost_effective_rules, smart_select_rules},
    handwritten::{handwritten_rules, phases},
    opts::{self, SplitPhase},
    patterns::gen_patterns,
    scheduler::{LoggingData, LoggingScheduler},
    tracking::TrackRewrites,
    veclang::{DiosRwrite, EGraph, VecLang},
};

#[derive(Hash, PartialEq, Eq)]
pub enum Phase {
    PreCompile,
    Compile,
    Opt,
}

#[allow(unused)]
fn retain_rules_by_name(rules: &mut Vec<DiosRwrite>, names: &[&str]) {
    rules.retain(|rewrite| names.contains(&rewrite.name.as_str()))
}

pub type LoggingRunner = Runner<VecLang, TrackRewrites, LoggingData>;

/// Run the rewrite rules over the input program and return the best (cost, program)
pub fn run(prog: &RecExpr<VecLang>, opts: &opts::Opts) -> (f64, RecExpr<VecLang>) {
    // let use_only_ruler = true;

    // let mut rules: Vec<DiosRwrite> = vec![];
    // let mut snd_phase: Vec<DiosRwrite> = vec![];

    // ====================================================================
    // Gather initial rules and optionally filter them by cost differential
    // ====================================================================

    let mut initial_rules: Vec<DiosRwrite> = vec![];

    // add handwritten rules
    if opts.handwritten {
        initial_rules.extend(handwritten_rules(prog, opts.no_ac, opts.no_vec));
    }

    // add external rules
    if let Some(filename) = &opts.rules {
        initial_rules.extend(external_rules(filename));
    }

    if initial_rules.is_empty() {
        eprintln!("Stopping early, no rules were added.");
        return (f64::NAN, prog.clone());
    }

    if let Some(cutoff) = opts.cost_filter {
        initial_rules =
            retain_cost_effective_rules(&initial_rules, opts.no_dup_vars, |x| x > cutoff);
    }

    // ================================
    // Separate rules into three phases
    // ================================

    let mut rules: HashMap<Phase, Vec<DiosRwrite>> = HashMap::from([
        (Phase::PreCompile, vec![]),
        (Phase::Compile, vec![]),
        (Phase::Opt, vec![]),
    ]);

    if let Some(split_phase_opt) = &opts.split_phase {
        match split_phase_opt {
            SplitPhase::Auto => unimplemented!(),
            SplitPhase::Handwritten => {
                for r in initial_rules {
                    rules.entry(phases(&r)).and_modify(|rules| rules.push(r));
                }
            }
        }
    } else {
        rules
            .entry(Phase::Compile)
            .and_modify(|rules| rules.extend(initial_rules));
    }

    // filter out rules that have a cost differential lower than cutoff
    // if let Some(cutoff) = opts.cost_filter {
    //     rules.push(retain_cost_effective_rules(
    //         &initial_rules,
    //         opts.no_dup_vars,
    //         |x| x > cutoff,
    //     ));

    //     // add a second
    //     if opts.split_phase {
    //         rules.push(retain_cost_effective_rules(
    //             &initial_rules,
    //             opts.no_dup_vars,
    //             |x| x <= cutoff,
    //         ));
    //     }
    // }

    // if we are using sub programs, generate patterns from the program
    // and then use smart_select_rules to generate rules.
    // if opts.sub_prog {
    //     let mut fancy_rules: Vec<DiosRwrite> = vec![];
    //     for (i, p) in gen_patterns(prog).into_iter().enumerate() {
    //         if let Some(r) = smart_select_rules(&rules, &p) {
    //             let name = format!("fancy_{}", i);
    //             fancy_rules.push(rw!(name; p => r));
    //         }
    //     }
    //     rules = fancy_rules;
    // }

    // if opts.dump_rules {
    //     if opts.split_phase {
    //         eprintln!("==== First phase rules ====");
    //     }
    //     for r in &rules {
    //         if let (Some(lhs), Some(rhs)) =
    //             (r.searcher.get_pattern_ast(), r.applier.get_pattern_ast())
    //         {
    //             eprintln!("[{}] {} => {}", r.name, lhs, rhs);
    //         } else {
    //             eprintln!("[{}] <opaque>", r.name);
    //         }
    //     }

    //     if opts.split_phase {
    //         eprintln!("==== Second phase rules ====");
    //         for r in &snd_phase {
    //             if let (Some(lhs), Some(rhs)) =
    //                 (r.searcher.get_pattern_ast(), r.applier.get_pattern_ast())
    //             {
    //                 eprintln!("[{}] {} => {}", r.name, lhs, rhs);
    //             } else {
    //                 eprintln!("[{}] <opaque>", r.name);
    //             }
    //         }
    //     }
    // }

    if opts.dry_run {
        eprintln!("Doing dry run. Aborting early.");
        return (f64::NAN, prog.clone());
    }

    // =============
    // Start phase 1
    // =============

    eprintln!("=================================");
    eprintln!("Starting Phase 1: Pre-compilation");
    eprintln!("=================================");

    let eg = eqsat::init_egraph();
    let (cost1, prog1, eg) = do_eqsat(&rules[&Phase::PreCompile], eg, prog, opts);
    eprintln!("Cost: {}", cost1);

    eprintln!("=============================");
    eprintln!("Starting Phase 2: Compilation");
    eprintln!("=============================");

    let (cost2, prog2, eg) = do_eqsat(&rules[&Phase::Compile], eg, &prog1, opts);
    eprintln!("Cost: {} (improved {})", cost2, cost1 - cost2);

    eprintln!("==============================");
    eprintln!("Starting Phase 3: Optimization");
    eprintln!("==============================");

    let (cost3, prog3, _eg) = do_eqsat(&rules[&Phase::Opt], eg, &prog2, opts);
    eprintln!("Cost: {} (improved {})", cost3, cost2 - cost3);

    (cost3, prog3)

    // let mut init_eg: EGraph = EGraph::new(TrackRewrites::default()).with_explanations_enabled();
    // init_eg.add(VecLang::Num(0));
    // let mut runner = LoggingRunner::new(Default::default())
    //     .with_egraph(init_eg)
    //     .with_expr(&prog)
    //     .with_node_limit(10_000_000)
    //     .with_time_limit(std::time::Duration::from_secs(opts.timeout as u64))
    //     .with_iter_limit(opts.iter_limit);

    // // select the scheduler
    // runner = match opts.scheduler {
    //     opts::SchedulerOpt::Simple => runner.with_scheduler(SimpleScheduler),
    //     opts::SchedulerOpt::Backoff => runner.with_scheduler(BackoffScheduler::default()),
    //     opts::SchedulerOpt::Logging => {
    //         if let Some(filename) = &opts.instrument {
    //             let write = Box::new(BufWriter::with_capacity(
    //                 1024,
    //                 File::create(filename).unwrap(),
    //             ));
    //             let mut sched = LoggingScheduler::new_w_write(runner.roots[0], prog.clone(), write);
    //             sched.write_headers();
    //             runner.with_scheduler(sched)
    //         } else {
    //             let sched = LoggingScheduler::new(runner.roots[0], prog.clone());
    //             runner.with_scheduler(sched)
    //         }
    //     }
    // };

    // eprintln!("Starting run with {} rules", rules.len());
    // runner = runner.run(&rules);

    // eprintln!("Egraph size: {}", runner.egraph.total_size());
    // report(&runner);

    // // print reason to STDERR.
    // eprintln!(
    //     "Stopped after {} iterations, reason: {:?}",
    //     runner.iterations.len(),
    //     runner.stop_reason
    // );

    // // if matches!(opts.scheduler, opts::SchedulerOpt::Logging) {
    // //     runner.scheduler.log();
    // // }

    // // Extract the resulting program
    // let (eg, root) = (runner.egraph.clone(), runner.roots[0].clone());
    // let extractor = Extractor::new(&eg, VecCostFn {});
    // let (cost, out_prog) = extractor.find_best(root);

    // // Second phase optimization
    // if opts.split_phase {
    //     eprintln!("==== Starting Second Phase Optimization ====");
    //     eprintln!("Using {} rules", snd_phase.len());

    //     let mut init_eg: EGraph = EGraph::new(TrackRewrites::default()).with_explanations_enabled();
    //     init_eg.add(VecLang::Num(0));
    //     let mut runner = LoggingRunner::new(Default::default())
    //         .with_egraph(init_eg)
    //         .with_expr(&out_prog)
    //         .with_node_limit(1_000_000)
    //         .with_time_limit(std::time::Duration::from_secs(opts.timeout as u64))
    //         .with_iter_limit(opts.iter_limit);
    //     runner = runner.run(&snd_phase);
    //     // report(&runner);

    //     let (eg, root) = (runner.egraph.clone(), runner.roots[0].clone());

    //     // // Always add the literal zero
    //     let extractor = Extractor::new(&eg, VecCostFn {});
    //     let (new_cost, new_prog) = extractor.find_best(root);
    //     eprintln!(
    //         "Improved cost by {} ({} - {})",
    //         cost - new_cost,
    //         cost,
    //         new_cost
    //     );
    //     (new_cost, new_prog)
    // } else {
    //     (cost, out_prog)
    // }
}
