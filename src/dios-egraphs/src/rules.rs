use egg::{rewrite as rw, *};

use itertools::Itertools;
// use libc::thread_standard_policy;

use crate::{
    binopsearcher::build_binop_or_zero_rule,
    config::*,
    cost::VecCostFn,
    macsearcher::build_mac_rule,
    patterns::gen_patterns,
    scheduler::{LoggingData, LoggingScheduler},
    searchutils::*,
    tracking::TrackRewrites,
    tree::{get_rewrites_used, print_rewrites_used},
    veclang::{DiosRwrite, EGraph, VecLang},
};

// Check if all the variables, in this case memories, are equivalent
fn is_all_same_memory_or_zero(vars: &Vec<String>) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let vars: Vec<Var> = vars.iter().map(|v| v.parse().unwrap()).collect();
    let zero = VecLang::Num(0);
    move |egraph, _, subst| {
        let non_zero_gets = vars
            .iter()
            .filter(|v| !egraph[subst[**v]].nodes.contains(&zero))
            .unique_by(|v| egraph.find(subst[**v]));
        non_zero_gets.count() < 2
    }
}

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
pub fn run(
    prog: &RecExpr<VecLang>,
    timeout: u64,
    no_ac: bool,
    no_vec: bool,
    iter_limit: usize,
    ruleset: Option<&str>,
) -> (f64, RecExpr<VecLang>) {
    let use_only_ruler = true;
    let (mut rules, snd_phase) = rules(no_ac, no_vec, ruleset, use_only_ruler);
    // filter_applicable_rules(&mut rules, prog);

    rules.extend(vec![
        // build_unop_rule("neg", "VecNeg"),
        // rw!("neg-zero-inv0"; "0" => "(neg 0)"),
        // rw!("neg-zero-inv1"; "(neg 0)" => "0"),
        // rw!("add-0"; "(+ 0 ?a)" => "?a"),
        // rw!("mul-0"; "(* 0 ?a)" => "0"),
        // rw!("mul-1"; "(* 1 ?a)" => "?a"),
        // rw!("add-0-inv"; "?a" => "(+ 0 ?a)"),
        // rw!("mul-1-inv"; "?a" => "(* 1 ?a)"),
        // rw!("div-1"; "(/ ?a 1)" => "?a"),
        // rw!("div-1-inv"; "?a" => "(/ ?a 1)"),
        // rw!("expand-zero-get"; "0" => "(Get 0 0)"),
        // rw!("expand-zero-plus"; "0" => "(+ 0 0)"),
        // rw!("vec-mac-add-mul";
        //         "(VecAdd ?v0 (VecMul ?v1 ?v2))"
        //         => "(VecMAC ?v0 ?v1 ?v2)"),
        // build_litvec_rule(),
    ]);

    // retain_rules_by_name(
    //     &mut rules,
    //     &[
    //         "+_binop_or_zero_vec",
    //         "*_binop_or_zero_vec",
    //         "litvec",
    //         "add-0-inv",
    //         "mul-1-inv",
    //         "vec-mac-add-mul",
    //         "vec-mac",
    //         "neg_unop",
    //         "expand-zero-get",
    //         "neg-zero-inv",
    //     ],
    // );

    let mut init_eg: EGraph = EGraph::new(TrackRewrites::default()).with_explanations_enabled();
    init_eg.add(VecLang::Num(0));
    let mut runner = LoggingRunner::new(Default::default())
        .with_egraph(init_eg)
        .with_expr(&prog)
        .with_node_limit(500_000)
        .with_time_limit(std::time::Duration::from_secs(timeout))
        // .with_hook(|runner| {
        //     eprintln!("Egraph big big? {}", runner.egraph.total_size());
        //     eprintln!("Egraph class big? {}", runner.egraph.number_of_classes());
        //     let (eg, root) = (&runner.egraph, &runner.roots[0]);
        //     let extractor = Extractor::new(&eg, VecCostFn { egraph: &eg });
        //     let (cost, _) = extractor.find_best(*root);
        //     eprintln!("Egraph cost? {}", cost);
        //     Ok(())
        // })
        .with_iter_limit(iter_limit);

    eprintln!("prog: {}", prog.pretty(80));
    let patterns = gen_patterns(prog);
    let mut fancy_rules: Vec<DiosRwrite> = vec![];
    // let cost_effective_rules = &rules;
    for (i, p) in patterns.into_iter().enumerate() {
        if let Some(r) = smart_select_rules(&rules, &p) {
            let name = format!("fancy_{}", i);
            fancy_rules.push(rw!(name; p => r));
        }
    }
    // eprintln!("{:#?}", patterns);
    // panic!("stop here");

    // add scheduler
    // let scheduler = LoggingScheduler::new(runner.roots[0], prog.clone());
    // let scheduler = SimpleScheduler;
    // runner = runner.with_scheduler(scheduler);

    let final_ruleset = fancy_rules.clone();
    // final_ruleset.append(&mut rules);

    eprintln!("Starting run with {} rules", final_ruleset.len());
    runner = runner.run(&final_ruleset);

    eprintln!("Egraph big big? {}", runner.egraph.total_size());
    report(&runner);

    // print reason to STDERR.
    eprintln!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );

    // runner.scheduler.log();

    let (eg, root) = (runner.egraph.clone(), runner.roots[0].clone());

    // Always add the literal zero
    let extractor = Extractor::new(&eg, VecCostFn {});
    let (cost, out_prog) = extractor.find_best(root);
    eprintln!("Egraph cost? {}", cost);
    eprintln!("{}", out_prog.pretty(80));

    eprintln!("==== Starting Second Phase Optimization ====");
    eprintln!("Using {} rules", snd_phase.len());

    let mut init_eg: EGraph = EGraph::new(TrackRewrites::default()).with_explanations_enabled();
    init_eg.add(VecLang::Num(0));
    let mut runner = LoggingRunner::new(Default::default())
        .with_egraph(init_eg)
        .with_expr(&out_prog)
        .with_node_limit(1_000_000)
        .with_time_limit(std::time::Duration::from_secs(300))
        .with_iter_limit(iter_limit);
    runner = runner.run(&snd_phase);
    report(&runner);

    let (eg, root) = (runner.egraph.clone(), runner.roots[0].clone());

    // Always add the literal zero
    let extractor = Extractor::new(&eg, VecCostFn {});
    let (cost, out_prog2) = extractor.find_best(root);
    eprintln!("optimized:\n{}", out_prog2.pretty(80));
    eprintln!("Egraph cost? {}", cost);

    (cost, out_prog2)
}

pub fn build_binop_rule(op_str: &str, vec_str: &str) -> DiosRwrite {
    let searcher: Pattern<VecLang> =
        vec_fold_op(&op_str.to_string(), &"a".to_string(), &"b".to_string())
            .parse()
            .unwrap();

    let applier: Pattern<VecLang> = format!(
        "({} {} {})",
        vec_str,
        vec_with_var(&"a".to_string()),
        vec_with_var(&"b".to_string())
    )
    .parse()
    .unwrap();

    rw!(format!("{}_binop_vec", op_str); { searcher } => { applier })
}

pub fn build_unop_rule(op_str: &str, vec_str: &str) -> DiosRwrite {
    let searcher: Pattern<VecLang> = vec_map_op(&op_str.to_string(), &"a".to_string())
        .parse()
        .unwrap();
    let applier: Pattern<VecLang> = format!("({} {})", vec_str, vec_with_var(&"a".to_string()))
        .parse()
        .unwrap();

    rw!(format!("{}_unop", op_str); { searcher } => { applier })
}

pub fn build_litvec_rule() -> DiosRwrite {
    let mem_vars = ids_with_prefix(&"a".to_string(), vector_width());
    let mut gets: Vec<String> = Vec::with_capacity(vector_width());
    for i in 0..vector_width() {
        gets.push(format!("(Get {} ?{}{})", mem_vars[i], "i", i))
    }
    let all_gets = gets.join(" ");

    let searcher: Pattern<VecLang> = format!("(Vec {})", all_gets).parse().unwrap();

    let applier: Pattern<VecLang> = format!("(LitVec {})", all_gets).parse().unwrap();

    rw!("litvec"; { searcher } => { applier }
        if is_all_same_memory_or_zero(&mem_vars))
}

pub fn rules(
    no_ac: bool,
    no_vec: bool,
    ruleset: Option<&str>,
    only_ruleset: bool,
) -> (Vec<DiosRwrite>, Vec<DiosRwrite>) {
    let mut rules: Vec<DiosRwrite> = vec![];
    let mut snd_rules: Vec<DiosRwrite> = vec![];

    if let Some(filename) = ruleset {
        let ruler = ruler_rules(filename);
        let fst = retain_cost_effective_rules(&ruler, false, |x| x > 5.0);
        let snd = retain_cost_effective_rules(&ruler, true, |x| x > 0.0 && x < 5.0);

        for r in &snd {
            eprintln!(
                "{} => {}",
                r.searcher.get_pattern_ast().unwrap().pretty(80),
                r.applier.get_pattern_ast().unwrap().pretty(80)
            );
        }
        // panic!("asdf");

        rules.extend(fst);
        snd_rules.extend(snd);
        if only_ruleset {
            return (rules, snd_rules);
        }
    }

    rules.extend(vec![
        rw!("add-0"; "(+ 0 ?a)" => "?a"),
        rw!("mul-0"; "(* 0 ?a)" => "0"),
        rw!("mul-1"; "(* 1 ?a)" => "?a"),
        rw!("add-0-inv"; "?a" => "(+ 0 ?a)"),
        rw!("mul-1-inv"; "?a" => "(* 1 ?a)"),
        rw!("div-1"; "(/ ?a 1)" => "?a"),
        rw!("div-1-inv"; "?a" => "(/ ?a 1)"),
        rw!("expand-zero-get"; "0" => "(Get 0 0)"),
        // Literal vectors, that use the same memory or no memory in every lane,
        // are cheaper
        build_litvec_rule(),
    ]);

    // Bidirectional rules
    rules.extend(
        vec![
            // Sign and negate
            rw!("neg-neg"; "(neg (neg ?a))" <=> "?a"),
            rw!("neg-sgn"; "(neg (sgn ?a))" <=> "(sgn (neg ?a))"),
            rw!("neg-zero-inv"; "0" <=> "(neg 0)"),
            rw!("neg-minus"; "(neg ?a)" <=> "(- 0 ?a)"),
            rw!("neg-minus-zero"; "(neg ?a)" <=> "(- 0 ?a)"),
            rw!("sqrt-1-inv"; "1" <=> "(sqrt 1)"),
        ]
        .concat(),
    );

    // Vector rules
    if !no_vec {
        rules.extend(vec![
            // Special MAC fusion rule
            rw!("vec-mac-add-mul";
                "(VecAdd ?v0 (VecMul ?v1 ?v2))"
                => "(VecMAC ?v0 ?v1 ?v2)"),
            // Custom searchers
            build_unop_rule("neg", "VecNeg"),
            build_unop_rule("sqrt", "VecSqrt"),
            build_unop_rule("sgn", "VecSgn"),
            build_binop_rule("/", "VecDiv"),
            build_binop_or_zero_rule("+", "VecAdd"),
            build_binop_or_zero_rule("*", "VecMul"),
            build_binop_or_zero_rule("-", "VecMinus"),
            build_mac_rule(),
        ]);
    } else {
        eprintln!("Skipping vector rules")
    }

    if !no_ac {
        rules.extend(vec![
            //  Basic associativity/commutativity/identities
            rw!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
            rw!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
            rw!("assoc-add"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
            rw!("assoc-mul"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        ]);
    }

    (rules, snd_rules)
}

fn ruler_rules(filename: &str) -> Vec<DiosRwrite> {
    if filename == "" {
        return vec![];
    }
    let contents = std::fs::read_to_string(filename).unwrap();
    let data = json::parse(&contents).unwrap();

    let mut rules = vec![];
    for (idx, eq) in data["eqs"].members().enumerate() {
        let lpat: Pattern<VecLang> = eq["lhs"].as_str().unwrap().parse().unwrap();
        let rpat: Pattern<VecLang> = eq["rhs"].as_str().unwrap().parse().unwrap();

        if eq["bidirectional"].as_bool().unwrap() {
            // we have to clone bc it is a bidirectional rule
            rules.extend(rw!(format!("ruler_{}_lr", idx); { lpat.clone() } <=> { rpat.clone() }))
        } else {
            rules.push(rw!(format!("ruler_{}_lr", idx); { lpat } => { rpat }))
        }
    }

    rules
}

fn walk_recexpr_help<F>(expr: &RecExpr<VecLang>, id: Id, action: F) -> F
where
    F: FnMut(&VecLang),
{
    // let root: Id = (expr.as_ref().len() - 1).into();
    let mut f = action;
    f(&expr[id]);
    for c in expr[id].children() {
        let newf = walk_recexpr_help(expr, *c, f);
        f = newf;
    }
    f
}

fn walk_recexpr<F>(expr: &RecExpr<VecLang>, action: F)
where
    F: FnMut(&VecLang),
{
    walk_recexpr_help(expr, (expr.as_ref().len() - 1).into(), action);
}

fn fold_recexpr<F, T>(expr: &RecExpr<VecLang>, init: T, mut action: F) -> T
where
    F: FnMut(T, &VecLang) -> T,
    T: Clone,
{
    let mut acc = init;
    walk_recexpr(expr, |l| acc = action(acc.clone(), l));
    acc
}

fn filter_vars(expr: &RecExpr<VecLang>) -> Vec<Var> {
    fold_recexpr(expr, vec![], |mut acc, l| {
        if let VecLang::Symbol(s) = l {
            acc.push(format!("{}", s).parse().unwrap());
        }
        acc
    })
}

fn retain_cost_effective_rules<F>(
    rules: &[DiosRwrite],
    all_vars: bool,
    cutoff: F,
) -> Vec<DiosRwrite>
where
    F: Fn(f64) -> bool,
{
    let mut costfn = VecCostFn {};
    let result = rules
        .iter()
        .filter(|r| {
            if let (Some(lhs), Some(rhs)) =
                (r.searcher.get_pattern_ast(), r.applier.get_pattern_ast())
            {
                let lexp: RecExpr<VecLang> = VecLang::from_pattern(lhs);
                let rexp: RecExpr<VecLang> = VecLang::from_pattern(rhs);
                let cost_differential = costfn.cost_rec(&lexp) - costfn.cost_rec(&rexp);

                let lhs_vars = r.searcher.vars();
                let all_vars_p = filter_vars(&lexp).len() == lhs_vars.len();

                if all_vars {
                    cutoff(cost_differential) && all_vars_p
                } else {
                    cutoff(cost_differential)
                }
            } else {
                false
            }
        })
        .cloned()
        .collect_vec();

    eprintln!("Retained {} rules", result.len());

    result
}

fn smart_select_rules(rules: &[DiosRwrite], pat: &Pattern<VecLang>) -> Option<Pattern<VecLang>> {
    // for r in rules.iter() {
    //     match (r.searcher.get_pattern_ast(), r.applier.get_pattern_ast()) {
    //         (Some(lhs), Some(rhs)) => eprintln!("[{}] {} => {}", r.name, lhs, rhs),
    //         _ => eprintln!("custom: {}", r.name),
    //     };
    // }

    // rules.retain(|r| {
    //     matches!(
    //         (r.searcher.get_pattern_ast(), r.applier.get_pattern_ast()),
    //         (Some(_), Some(_))
    //     )
    // });

    // let check: &'static str =
    //     "(Vec (+ ?a (+ ?b (+ (+ (* ?c (* ?d ?e)) (neg (* ?f (* ?d ?g)))) (* ?h (* ?d ?i))))) ?j)";

    let expr: RecExpr<VecLang> = pat.ast.to_string().parse().unwrap();

    let mut egraph: EGraph = EGraph::new(TrackRewrites::default());
    let root = egraph.add_expr(&expr);
    let _zero = egraph.add_expr(&"0".parse().unwrap());

    let mut runner = LoggingRunner::new(Default::default())
        .with_egraph(egraph)
        .with_node_limit(1_000_000)
        .with_time_limit(std::time::Duration::from_secs(30))
        .with_iter_limit(100);

    eprintln!("start:\n{}", pat.pretty(80));

    runner = runner.run(rules);
    // runner.egraph.dot().to_png("graph.png").unwrap();

    // let check_expr: Pattern<VecLang> = check.parse().unwrap();
    let matches: Vec<Id> = pat
        .search(&runner.egraph)
        .iter()
        .map(|m| m.eclass)
        .collect();

    // eprintln!("matches: {:?}", matches);
    let extractor = Extractor::new(
        &runner.egraph,
        VecCostFn {
            // egraph: &runner.egraph,
        },
    );
    let mut best = None;
    for m in matches {
        let (cost, prog) = extractor.find_best(m);
        if let Some((best_cost, _)) = &best {
            if &cost < best_cost {
                best = Some((cost, prog));
            }
        } else {
            best = Some((cost, prog));
        }
    }

    // if let Some((cost, prog)) = best {
    //     eprintln!("best: [{}] {}", cost, prog);
    // } else {
    //     eprintln!("no pattern found");
    // }

    // eprintln!("trying a different technique");
    let (cost, prog) = extractor.find_best(root);
    eprintln!("searching from the root: [{}]\n{}", cost, prog.pretty(80));

    report(&runner);

    // runner = LoggingRunner::new(Default::default())
    //     .with_egraph(runner.egraph)
    //     .with_node_limit(1_000_000)
    //     .with_time_limit(std::time::Duration::from_secs(100))
    //     .with_iter_limit(5);
    // runner.egraph.add_expr(&check_expr);
    // runner = runner.run(&rules);
    // report(&runner);

    // if let Some(class) = &runner.egraph.lookup_expr(&check_expr) {
    //     let extractor = Extractor::new(
    //         &runner.egraph,
    //         VecCostFn {
    //             egraph: &runner.egraph,
    //         },
    //     );
    //     let (cost, prog) = extractor.find_best(*class);
    //     eprintln!("[{}] {}", cost, prog);
    // } else {
    //     eprintln!("nothing happened");
    // }

    // rw!("t"; expr => prog)

    let mut c = VecCostFn {
        // egraph: &runner.egraph,
    };

    let orig_cost = c.cost_rec(&expr);

    // eprintln!("|{} - {}| = {}", cost, orig_cost, (cost - orig_cost).abs());

    if (cost - orig_cost).abs() > 0.0 {
        Some(format!("{}", prog).parse().unwrap())
    } else {
        None
    }
}
