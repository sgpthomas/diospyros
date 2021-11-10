use egg::{rewrite as rw, *};
use rand::prelude::*;

use itertools::Itertools;

use crate::{
    binopsearcher::build_binop_or_zero_rule,
    config::*,
    cost::VecCostFn,
    macsearcher::build_mac_rule,
    scheduler::{LoggingData, LoggingScheduler},
    searchutils::*,
    tracking::{CustomExtractor, TrackRewrites},
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
            dropped = format!("{} {}", dropped, r.name())
        };
        !drop
    });
    if dropped != "" {
        eprintln!("Dropping inapplicable rules:{}", dropped);
    }
}

#[allow(unused)]
fn filter_rules_by_name(rules: &mut Vec<DiosRwrite>, names: &[&str]) {
    rules.retain(|rewrite| names.contains(&rewrite.name()))
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
    let rules = rules(no_ac, no_vec, ruleset);
    // filter_applicable_rules(&mut rules, prog);

    // filter_rules_by_name(
    //     &mut rules,
    //     &[
    //         "sqrt-1-inv",
    //         "neg-neg-rev",
    //         "expand-zero-get",
    //         "add-0-inv",
    //         "mul-1-inv",
    //         "neg-minus",
    //         "div-1-inv",
    //         "+_binop_or_zero",
    //         "neg_unop",
    //         "litvec",
    //         "*_binop_or_zero",
    //         "/_binop",
    //         "-_binop_or_zero",
    //         "vec-mac",
    //         "sqrt_unop",
    //     ],
    // );

    let mut init_eg: EGraph = EGraph::new(TrackRewrites::default()).with_explanations_enabled();
    init_eg.add(VecLang::Num(0));
    let mut runner = Runner::new(Default::default())
        .with_egraph(init_eg)
        .with_expr(&prog)
        .with_node_limit(500_000)
        .with_time_limit(std::time::Duration::from_secs(timeout))
        .with_hook(|runner| {
            eprintln!("Egraph big big? {}", runner.egraph.total_size());
            eprintln!("Egraph class big? {}", runner.egraph.number_of_classes());
            Ok(())
        })
        .with_hook(|runner| {
            let (eg, root) = (&runner.egraph, &runner.roots[0]);
            let extractor = Extractor::new(&eg, VecCostFn { egraph: &eg });
            let (cost, _) = extractor.find_best(*root);
            eprintln!("Egraph cost? {}", cost);
            Ok(())
        })
        .with_iter_limit(iter_limit);

    // add scheduler
    // let scheduler = LoggingScheduler::new(runner.roots[0]);
    let scheduler = SimpleScheduler;
    runner = runner.with_scheduler(scheduler);

    // eprintln!("{:#?}", rules);
    eprintln!("Starting run with {} rules", rules.len());
    runner = runner.run(&rules);

    let start = "(Vec
    (+
      (+
        (* (Get aq 3) (Get bq 0))
        (+ (* (Get aq 0) (Get bq 3)) (* (Get aq 1) (Get bq 2))))
      (neg (* (Get aq 2) (Get bq 1))))
    (+
      (+
        (* (Get aq 3) (Get bq 1))
        (+ (* (Get aq 1) (Get bq 3)) (* (Get aq 2) (Get bq 0))))
      (neg (* (Get aq 0) (Get bq 2)))))"
        .parse()
        .unwrap();

    let end = "(VecAdd
    (VecAdd
      (VecMul (LitVec (Get aq 3) (Get aq 3)) (LitVec (Get bq 0) (Get bq 1)))
      (VecMAC
        (VecMul (Vec (Get aq 0) (Get aq 1)) (Vec (Get bq 3) (Get bq 3)))
        (LitVec (Get aq 1) (Get aq 2))
        (LitVec (Get bq 2) (Get bq 0))))
    (VecNeg (VecMul (LitVec (Get aq 2) (Get aq 0)) (LitVec (Get bq 1) (Get bq 2)))))"
        .parse()
        .unwrap();

    eprintln!("enabled?: {}", runner.egraph.are_explanations_enabled());
    eprintln!(
        "Explanation: {}",
        runner.explain_equivalence(&start, &end).get_flat_string()
    );

    eprintln!("Egraph big big? {}", runner.egraph.total_size());

    report(&runner);

    // print reason to STDERR.
    eprintln!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );

    let (eg, root) = (runner.egraph, runner.roots[0]);

    // Always add the literal zero
    let extractor = Extractor::new(&eg, VecCostFn { egraph: &eg });
    let (cost, prog) = extractor.find_best(root);
    eprintln!("Egraph cost? {}", cost);

    (cost, prog)
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

pub fn rules(no_ac: bool, no_vec: bool, ruleset: Option<&str>) -> Vec<DiosRwrite> {
    let mut rules: Vec<DiosRwrite> = vec![
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
    ];

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

    if let Some(filename) = ruleset {
        rules.extend(ruler_rules(filename));
    }

    let mut rng = rand::thread_rng();
    rules.shuffle(&mut rng);

    rules
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
