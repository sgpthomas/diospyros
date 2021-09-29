use egg::{rewrite as rw, *};

use itertools::Itertools;
use ruler::{dios, Synthesizer};

use crate::{
    binopsearcher::build_binop_or_zero_rule,
    config::*,
    cost::VecCostFn,
    macsearcher::build_mac_rule,
    searchutils::*,
    veclang::{EGraph, VecLang},
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

fn filter_applicable_rules(rules: &mut Vec<Rewrite<VecLang, ()>>, prog: &RecExpr<VecLang>) {
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

fn report(runner: &Runner<VecLang, ()>) {
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

/// Run the rewrite rules over the input program and return the best (cost, program)
pub fn run(
    prog: &RecExpr<VecLang>,
    timeout: u64,
    no_ac: bool,
    no_vec: bool,
) -> (f64, RecExpr<VecLang>) {
    let mut rules = rules(no_ac, no_vec);
    filter_applicable_rules(&mut rules, prog);
    let mut init_eg: EGraph = EGraph::new(());
    init_eg.add(VecLang::Num(0));
    let runner: Runner<VecLang, ()> = Runner::default()
        .with_egraph(init_eg)
        .with_expr(&prog)
        .with_node_limit(10_000_000)
        .with_time_limit(std::time::Duration::from_secs(timeout))
        .with_iter_limit(10_000)
        .run(&rules);

    report(&runner);

    // print reason to STDERR.
    eprintln!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );

    let (eg, root) = (runner.egraph, runner.roots[0]);

    // Always add the literal zero
    let mut extractor = Extractor::new(&eg, VecCostFn { egraph: &eg });
    extractor.find_best(root)
}

pub fn build_binop_rule(op_str: &str, vec_str: &str) -> Rewrite<VecLang, ()> {
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

    rw!(format!("{}_binop", op_str); { searcher } => { applier })
}

pub fn build_unop_rule(op_str: &str, vec_str: &str) -> Rewrite<VecLang, ()> {
    let searcher: Pattern<VecLang> = vec_map_op(&op_str.to_string(), &"a".to_string())
        .parse()
        .unwrap();
    let applier: Pattern<VecLang> = format!("({} {})", vec_str, vec_with_var(&"a".to_string()))
        .parse()
        .unwrap();

    rw!(format!("{}_unop", op_str); { searcher } => { applier })
}

pub fn build_litvec_rule() -> Rewrite<VecLang, ()> {
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

pub fn rules(no_ac: bool, no_vec: bool) -> Vec<Rewrite<VecLang, ()>> {
    let mut rules: Vec<Rewrite<VecLang, ()>> = vec![
        rw!("add-0"; "(+ 0 ?a)" => "?a"),
        rw!("mul-0"; "(* 0 ?a)" => "0"),
        rw!("mul-1"; "(* 1 ?a)" => "?a"),
        rw!("div-1"; "(/ ?a 1)" => "?a"),
        rw!("add-0-inv"; "?a" => "(+ 0 ?a)"),
        rw!("mul-1-inv"; "?a" => "(* 1 ?a)"),
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
            // build_binop_or_zero_rule("+", "VecAdd"),
            build_binop_or_zero_rule("*", "VecMul"),
            // build_binop_or_zero_rule("-", "VecMinus"),
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

    if true {
        rules.extend(ruler_rules());
    }

    rules
}

fn ruler_rules() -> Vec<Rewrite<VecLang, ()>> {
    let p = ruler::SynthParams {
        seed: 0,
        n_samples: 0,
        chunk_size: 100000,
        minimize: false,
        no_constants_above_iter: 999999,
        no_conditionals: false,
        no_run_rewrites: false,
        linear_cvec_matching: false,
        outfile: "out.json".to_string(),
        eqsat_node_limit: 300000,
        eqsat_time_limit: 60,
        important_cvec_offsets: 5,
        str_int_variables: 1,
        complete_cvec: false,
        no_xor: false,
        no_shift: false,
        use_smt: false,
        do_final_run: false,
        // custom
        rules_to_take: 2,
        num_fuzz: 4,
        iters: 2,
        eqsat_iter_limit: 10,
        vector_size: 2,
        variables: 4,
        abs_timeout: 10,
    };

    let mut rules = vec![];

    // start synthesizer
    let syn = Synthesizer::<dios::VecLang>::new(p).run();
    for eq in &syn.eqs {
        eprintln!("{} <=> {}", eq.lhs, eq.rhs);
        let lpat: Pattern<VecLang> = eq.lhs.to_string().parse().unwrap();
        let rpat: Pattern<VecLang> = eq.rhs.to_string().parse().unwrap();

        let left_vars = lpat.vars();
        let right_vars = rpat.vars();

        // if right vars are a subset of left vars, add the rule in this direction
        if rpat.vars().iter().all(|x| left_vars.contains(x)) {
            rules.push(rw!(format!("{}_lr", eq.name); { lpat.clone() } => { rpat.clone() }));
        }

        // if left vars are a subset of right vars, add the rule in the backward direction
        if lpat.vars().iter().all(|x| right_vars.contains(x)) {
            rules.push(rw!(format!("{}_rl", eq.name); { rpat } => { lpat }));
        }
    }

    rules
}
