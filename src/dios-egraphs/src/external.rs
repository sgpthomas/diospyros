use std::path::PathBuf;

use egg::{
    rewrite as rw, CostFunction, Extractor, Id, Language, Pattern, RecExpr, Searcher, Var,
};
use itertools::Itertools;

use crate::{
    cost::VecCostFn,
    handwritten::build_litvec_rule,
    rules::LoggingRunner,
    tracking::TrackRewrites,
    veclang::{DiosRwrite, EGraph, VecLang},
};

/// Return rules read in from a json file.
pub fn external_rules(filename: &PathBuf) -> Vec<DiosRwrite> {
    let contents = std::fs::read_to_string(filename).unwrap();
    let data = json::parse(&contents).unwrap();

    let mut rules = vec![];
    for (idx, eq) in data["eqs"].members().enumerate() {
        let lpat: Pattern<VecLang> = eq["lhs"].as_str().unwrap().parse().unwrap();
        let rpat: Pattern<VecLang> = eq["rhs"].as_str().unwrap().parse().unwrap();

        if eq["bidirectional"].as_bool().unwrap() {
            // we have to clone bc it is a bidirectional rule
            rules.extend(
                rw!(format!("ruler_{}_lr", idx); { lpat.clone() } <=> { rpat.clone() }),
            )
        } else {
            rules.push(rw!(format!("ruler_{}_lr", idx); { lpat } => { rpat }))
        }
    }

    // hack to add some important rules
    rules.extend(vec![
        rw!("vec-neg"; "(Vec (neg ?a) (neg ?b))" => "(VecNeg (Vec ?a ?b))"),
        rw!("vec-neg0-l"; "(Vec 0 (neg ?b))" => "(VecNeg (Vec 0 ?b))"),
        rw!("vec-neg0-r"; "(Vec (neg ?a) 0)" => "(VecNeg (Vec ?a 0))"),
        build_litvec_rule(),
    ]);

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

pub fn retain_cost_effective_rules<F, G>(
    rules: &[DiosRwrite],
    all_vars: bool,
    metric: F,
    cutoff: G,
) -> Vec<DiosRwrite>
where
    F: Fn(&DiosRwrite) -> f64,
    G: Fn(f64) -> bool,
{
    let result = rules
        .iter()
        .filter(|r| {
            let cost_diff = metric(r);
            if let Some(lhs) = r.searcher.get_pattern_ast() {
                if all_vars {
                    let lexp: RecExpr<VecLang> = VecLang::from_pattern(lhs);
                    let lhs_vars = r.searcher.vars();
                    let all_vars_p = filter_vars(&lexp).len() == lhs_vars.len();
                    cutoff(cost_diff) && all_vars_p
                } else {
                    cutoff(cost_diff)
                }
            } else {
                cutoff(cost_diff)
            }
        })
        .cloned()
        .collect_vec();

    eprintln!("Retained {} rules", result.len());

    result
}

#[allow(unused)]
pub fn smart_select_rules(
    rules: &[DiosRwrite],
    pat: &Pattern<VecLang>,
) -> Option<Pattern<VecLang>> {
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

    // report(&runner);

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

    let mut c = VecCostFn {};

    let orig_cost = c.cost_rec(&expr);

    // eprintln!("|{} - {}| = {}", cost, orig_cost, (cost - orig_cost).abs());

    if (cost - orig_cost).abs() > 0.0 {
        Some(format!("{}", prog).parse().unwrap())
    } else {
        None
    }
}
