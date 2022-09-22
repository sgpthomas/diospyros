use std::{
    collections::HashMap,
    fmt::Display,
    fs::{File, OpenOptions},
    io,
    io::Write,
    str::FromStr,
};

use egg::{CostFunction, Extractor, RecExpr, Runner};

use crate::{
    cost::{cost_average, cost_differential, VecCostFn},
    eqsat::{self, do_eqsat},
    external::{external_rules, retain_cost_effective_rules},
    handwritten::{self, handwritten_rules},
    opts::{self, SplitPhase},
    scheduler::LoggingData,
    split_by_syntax,
    top_down_extract::TopDownExtractor,
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

impl FromStr for Phase {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pre-compile" => Ok(Phase::PreCompile),
            "compile" => Ok(Phase::Compile),
            "opt" => Ok(Phase::Opt),
            _ => Err(format!(
                "Unknown phase: {}. Valid options are {}",
                s, "[`pre-compile`, `compile`, `opt`]"
            )),
        }
    }
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

fn print_rules(f: &mut dyn Write, rules: &[DiosRwrite]) -> Result<(), io::Error> {
    for r in rules {
        writeln!(
            f,
            "[{} cd:{:.2} avg:{:.2}] ",
            r.name,
            cost_differential(r),
            cost_average(r)
        )?;
        if let (Some(lhs), Some(rhs)) =
            (r.searcher.get_pattern_ast(), r.applier.get_pattern_ast())
        {
            writeln!(f, "  {} => {}", lhs, rhs)?;
        } else {
            writeln!(f, "  <opaque>")?;
        }
    }
    Ok(())
}

pub type LoggingRunner = Runner<VecLang, TrackRewrites, LoggingData>;

/// Run the rewrite rules over the input program and return the best (cost, program)
pub fn run(orig_prog: &RecExpr<VecLang>, opts: &opts::Opts) -> (f64, RecExpr<VecLang>) {
    // ====================================================================
    // Gather initial rules and optionally filter them by cost differential
    // ====================================================================

    let mut initial_rules: Vec<DiosRwrite> = vec![];

    // add handwritten rules
    if opts.handwritten {
        initial_rules.extend(handwritten_rules(
            orig_prog,
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
        return (f64::NAN, orig_prog.clone());
    }

    if let Some(cutoff) = opts.cost_filter {
        initial_rules = retain_cost_effective_rules(
            &initial_rules,
            opts.no_dup_vars,
            &[(cost_differential, Box::new(move |x| x > cutoff))],
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

    match opts.split_phase {
        Some(SplitPhase::Auto) => {
            let pre_compile = retain_cost_effective_rules(
                &initial_rules,
                false,
                &[
                    // (cost_differential, Box::new(|x: f64| x.abs() < 5.0)),
                    (cost_average, Box::new(|x| x <= 70.0)),
                ],
            );
            let compile = retain_cost_effective_rules(
                &initial_rules,
                false,
                &[(cost_average, Box::new(|x| 10.0 <= x && x < 70.0))],
                // &[(cost_differential, Box::new(|x| x.abs() > 5.0))],
            );
            let opt = retain_cost_effective_rules(
                    &initial_rules,
                    false,
		    &[
			(cost_average, Box::new(|x: f64| x < 10.0))
		    ]
                    // &[
                    //     (cost_average, Box::new(|x| x < 5.0)),
                    //     (cost_differential, Box::new(|x| 2.0 < x && x.abs() < 5.0)),
                    // ],
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
        Some(SplitPhase::Handwritten) => {
            for r in initial_rules {
                rules
                    .entry(handwritten::phases(&r))
                    .and_modify(|rules| rules.push(r));
            }
        }
        Some(SplitPhase::Syntax) => {
            for r in initial_rules {
                rules
                    .entry(split_by_syntax::phases(&r))
                    .and_modify(|rules| rules.push(r));
            }
        }
        Some(SplitPhase::None) | None => {
            rules
                .entry(Phase::Compile)
                .and_modify(|rules| rules.extend(initial_rules));
        }
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

    if opts.dry_run {
        eprintln!("Doing dry run. Aborting early.");
    }

    let mut eg = eqsat::init_egraph();
    let mut prog = orig_prog.clone();
    let mut cost = VecCostFn.cost_rec(&prog);

    let mut file: Option<File> = if let Some(pathbuf) = &opts.dump_rules {
        OpenOptions::new()
            .create(true)
            .write(true)
            .open(pathbuf)
            .ok()
    } else {
        None
    };

    for (i, phase) in opts.phase.iter().enumerate() {
        eprintln!("=================================");
        eprintln!("Starting Phase {}: {}", i + 1, &phase);
        eprintln!("=================================");

        eprintln!("Using {} rules", rules[phase].len());
        if let Some(f) = &mut file {
            writeln!(f, "=================================").unwrap();
            writeln!(f, "Starting Phase {}: {}", i + 1, &phase).unwrap();
            writeln!(f, "=================================").unwrap();

            print_rules(f, &rules[phase]).expect("Failed to write.");
        }

        if opts.dry_run {
            continue;
        }

        // do equality saturation with the rules in this phase
        let (new_cost, new_eg, root, new_prog) = do_eqsat(&rules[phase], eg, &prog, opts);

        eprintln!("Cost: {} (improved {})", new_cost, cost - new_cost);

        if opts.new_egraph && i != opts.phase.len() - 1 {
            if *phase == Phase::PreCompile {
                // let extractor = Extractor::new(
                //     &new_eg,
                //     PhaseCostFn::from_rules(
                //         rules[&Phase::Compile].clone(),
                //         orig_prog.clone(),
                //     ),
                // );
                // let (cost, new_prog) = extractor.find_best(root);
                // let (cost, new_prog) = Extractor::new(&new_eg, VecCostFn).find_best(root);

                let patterns: Vec<
                    std::sync::Arc<dyn egg::Searcher<VecLang, _> + Send + Sync>,
                > = rules[&Phase::Compile]
                    .iter()
                    .map(|x| &x.searcher)
                    .cloned()
                    .collect();

                let new_prog =
                    TopDownExtractor::new(&new_eg, patterns.as_slice(), VecCostFn)
                        .find_best(root);

                eprintln!("new:\n{}", new_prog.pretty(100));
                // eprintln!("Extracted prog cost: {cost}");
                prog = new_prog;
            }
            eg = eqsat::init_egraph();
        } else {
            eg = new_eg;
        }
        cost = new_cost;

        // HACK:
        if i == opts.phase.len() - 1 {
            prog = new_prog;
        }
    }

    eg.rebuild();
    eg.dot().to_pdf("test.pdf").unwrap();
    eprintln!("Wrote test.pdf");

    if opts.dry_run {
        (f64::NAN, prog)
    } else {
        (cost, prog)
    }
}
