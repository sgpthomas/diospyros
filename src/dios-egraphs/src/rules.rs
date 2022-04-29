use std::{collections::HashMap, fmt::Display};

use egg::{RecExpr, Runner};

use crate::{
    cost::{cost_average, cost_differential},
    eqsat::{self, do_eqsat},
    external::{external_rules, retain_cost_effective_rules},
    handwritten::{self, handwritten_rules},
    opts::{self, SplitPhase},
    scheduler::LoggingData,
    split_by_syntax,
    tracking::TrackRewrites,
    veclang::{DiosRwrite, VecLang},
};

/// Different "phases" of e-graph rule application.
#[derive(Hash, PartialEq, Eq)]
pub enum Phase {
    PreCompile,
    Compile,
    Opt,
}

impl Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let s = match self {
            Phase::PreCompile => "Pre-compilation",
            Phase::Compile => "Compilation",
            Phase::Opt => "Optimization",
        };
        write!(f, "{}", s)
    }
}

fn print_rules(rules: &[DiosRwrite]) {
    for r in rules {
        eprintln!(
            "[{} cd:{:.2} avg:{:.2}] ",
            r.name,
            cost_differential(r),
            cost_average(r)
        );
        if let (Some(lhs), Some(rhs)) =
            (r.searcher.get_pattern_ast(), r.applier.get_pattern_ast())
        {
            eprintln!("  {} => {}", lhs, rhs);
        } else {
            eprintln!("  <opaque>");
        }
    }
}

pub type LoggingRunner = Runner<VecLang, TrackRewrites, LoggingData>;

/// Run the rewrite rules over the input program and return the best (cost, program)
pub fn run(prog: &RecExpr<VecLang>, opts: &opts::Opts) -> (f64, RecExpr<VecLang>) {
    // ====================================================================
    // Gather initial rules and optionally filter them by cost differential
    // ====================================================================

    let mut initial_rules: Vec<DiosRwrite> = vec![];

    // add handwritten rules
    if opts.handwritten {
        initial_rules.extend(handwritten_rules(
            prog,
            opts.vector_width,
            opts.no_ac,
            opts.no_vec,
        ));
    }

    // add external rules
    if let Some(filename) = &opts.rules {
        initial_rules.extend(external_rules(
            opts.vector_width,
            filename,
            opts.pre_desugared,
        ));
    }

    if initial_rules.is_empty() {
        eprintln!("Stopping early, no rules were added.");
        return (f64::NAN, prog.clone());
    }

    if let Some(cutoff) = opts.cost_filter {
        initial_rules = retain_cost_effective_rules(
            &initial_rules,
            opts.no_dup_vars,
            cost_differential,
            |x| x > cutoff,
        );
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
            SplitPhase::Auto => {
                let pre_compile = retain_cost_effective_rules(
                    &initial_rules,
                    false,
                    cost_average,
                    |x| x < 10.0,
                );
                let compile = retain_cost_effective_rules(
                    &initial_rules,
                    false,
                    cost_average,
                    |x| 10.0 <= x && x < 70.0,
                );
                let opt = retain_cost_effective_rules(
                    &initial_rules,
                    false,
                    cost_average,
                    |x| 70.0 <= x,
                );
                rules
                    .entry(Phase::PreCompile)
                    .and_modify(|rules| rules.extend(pre_compile));
                rules
                    .entry(Phase::Compile)
                    .and_modify(|rules| rules.extend(compile));
                rules
                    .entry(Phase::Opt)
                    .and_modify(|rules| rules.extend(opt));
            }
            SplitPhase::Handwritten => {
                for r in initial_rules {
                    rules
                        .entry(handwritten::phases(&r))
                        .and_modify(|rules| rules.push(r));
                }
            }
            SplitPhase::Syntax => {
                for r in initial_rules {
                    rules
                        .entry(split_by_syntax::phases(&r))
                        .and_modify(|rules| rules.push(r));
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

    let order = [Phase::PreCompile, Phase::Compile, Phase::Opt];

    if opts.dry_run {
        eprintln!("Doing dry run. Aborting early.");
    }

    let mut eg = eqsat::init_egraph();
    let mut prog = prog.clone();
    let mut cost = None;
    for (i, phase) in order.iter().enumerate() {
        eprintln!("=================================");
        eprintln!("Starting Phase {}: {}", i + 1, &phase);
        eprintln!("=================================");

        eprintln!("Using {} rules", rules[phase].len());
        if opts.dump_rules {
            print_rules(&rules[phase]);
        }

        if opts.dry_run {
            continue;
        }

        let (new_cost, new_prog, new_eg) = do_eqsat(&rules[phase], eg, &prog, opts);
        if let Some(old_cost) = cost {
            eprintln!("Cost: {} (improved {})", new_cost, old_cost - new_cost);
        } else {
            eprintln!("Cost: {}", new_cost);
        }

        if opts.new_egraph {
            eg = eqsat::init_egraph();
        } else {
            eg = new_eg;
        }
        prog = new_prog;
        cost = Some(new_cost);
    }

    if opts.dry_run {
        (f64::NAN, prog)
    } else {
        (cost.unwrap(), prog)
    }
}
