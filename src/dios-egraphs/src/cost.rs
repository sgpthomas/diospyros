use std::sync::Arc;

use egg::*;

use crate::{
    recexpr_helpers,
    tracking::TrackRewrites,
    veclang::{DiosRwrite, VecLang},
};

pub struct VecCostFn;

impl CostFunction<VecLang> for VecCostFn {
    type Cost = f64;
    // you're passed in an enode whose children are costs instead of eclass ids
    fn cost<C>(&mut self, enode: &VecLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        const LITERAL: f64 = 0.001;
        const STRUCTURE: f64 = 0.1;
        const VEC_OP: f64 = 1.;
        const OP: f64 = 2.;
        const BIG: f64 = 100.0;
        let op_cost = match enode {
            // You get literals for extremely cheap
            VecLang::Num(..) => LITERAL,
            VecLang::Symbol(..) => LITERAL,
            VecLang::Get(..) => LITERAL,

            // And list structures for quite cheap
            VecLang::List(..) => STRUCTURE,
            VecLang::Concat(..) => STRUCTURE,

            // Vectors are cheap if they have literal values
            VecLang::Vec(vals) => {
                // For now, workaround to determine if children are num, symbol,
                // or get
                let non_literals = vals.iter().any(|&x| costs(x) > 3. * LITERAL);
                if non_literals {
                    BIG
                } else {
                    STRUCTURE
                }
            }
            VecLang::LitVec(..) => LITERAL,

            // But scalar and vector ops cost something
            VecLang::Add(vals) => OP * (vals.iter().count() as f64 - 1.),
            VecLang::Mul(vals) => OP * (vals.iter().count() as f64 - 1.),
            VecLang::Minus(vals) => OP * (vals.iter().count() as f64 - 1.),
            VecLang::Div(vals) => OP * (vals.iter().count() as f64 - 1.),

            VecLang::Sgn(..) => OP,
            VecLang::Neg(..) => OP,
            VecLang::Sqrt(..) => OP,

            VecLang::VecAdd(..) => VEC_OP,
            VecLang::VecMinus(..) => VEC_OP,
            VecLang::VecMul(..) => VEC_OP,
            VecLang::VecMAC(..) => VEC_OP,
            VecLang::VecDiv(..) => VEC_OP,
            VecLang::VecNeg(..) => VEC_OP,
            VecLang::VecSqrt(..) => VEC_OP,
            VecLang::VecSgn(..) => VEC_OP,
            VecLang::Or(_) => VEC_OP,
            VecLang::And(_) => VEC_OP,
            VecLang::Ite(_) => VEC_OP,
            VecLang::Lt(_) => VEC_OP,
            // _ => VEC_OP,
        };
        enode.fold(op_cost, |sum, id| sum + costs(id))
    }
}

pub fn cost_average(r: &DiosRwrite) -> f64 {
    if let (Some(lhs), Some(rhs)) =
        (r.searcher.get_pattern_ast(), r.applier.get_pattern_ast())
    {
        let lexp: RecExpr<VecLang> = VecLang::from_pattern(lhs);
        let rexp: RecExpr<VecLang> = VecLang::from_pattern(rhs);
        let mut costfn = VecCostFn {};
        (costfn.cost_rec(&lexp) + costfn.cost_rec(&rexp)) / 2.
    } else {
        match r.name.as_str() {
            "litvec" => 100.,
            "+_binop_or_zero_vec" => 50.,
            "*_binop_or_zero_vec" => 50.,
            "-_binop_or_zero_vec" => 50.,
            "vec-mac" => 100.,
            _ => panic!("rule: {:?}", r),
        }
    }
}

pub fn cost_differential(r: &DiosRwrite) -> f64 {
    if let (Some(lhs), Some(rhs)) =
        (r.searcher.get_pattern_ast(), r.applier.get_pattern_ast())
    {
        let lexp: RecExpr<VecLang> = VecLang::from_pattern(lhs);
        let rexp: RecExpr<VecLang> = VecLang::from_pattern(rhs);
        let mut costfn = VecCostFn {};
        costfn.cost_rec(&lexp) - costfn.cost_rec(&rexp)
    } else {
        match r.name.as_str() {
            "litvec" => 0.099,
            "+_binop_or_zero_vec" => 102.8,
            "*_binop_or_zero_vec" => 102.8,
            "-_binop_or_zero_vec" => 102.8,
            "vec-mac" => 106.7,
            _ => panic!("rule: {:?}", r),
        }
    }
}

/// Checks if the searcher `lhs` matches `expr`.
fn match_recexpr(
    pattern: &egg::RecExpr<ENodeOrVar<VecLang>>,
    pattern_root: &ENodeOrVar<VecLang>,
    expr: &egg::RecExpr<VecLang>,
    expr_root: &VecLang,
) -> bool {
    match (pattern_root, expr_root) {
        // no children
        (ENodeOrVar::ENode(VecLang::Num(n0)), VecLang::Num(n1)) => n0 == n1,
        (ENodeOrVar::ENode(VecLang::Symbol(s0)), VecLang::Symbol(s1)) => s0 == s1,

        // 1 child
        (ENodeOrVar::ENode(VecLang::Sgn(lefts)), VecLang::Sgn(rights))
        | (ENodeOrVar::ENode(VecLang::Sqrt(lefts)), VecLang::Sqrt(rights))
        | (ENodeOrVar::ENode(VecLang::Neg(lefts)), VecLang::Neg(rights))
        | (ENodeOrVar::ENode(VecLang::VecNeg(lefts)), VecLang::VecNeg(rights))
        | (ENodeOrVar::ENode(VecLang::VecSqrt(lefts)), VecLang::VecSqrt(rights))
        | (ENodeOrVar::ENode(VecLang::VecSgn(lefts)), VecLang::VecSgn(rights)) => lefts
            .iter()
            .zip(rights.iter())
            .all(|(l, r)| match_recexpr(&pattern, &pattern[*l], &expr, &expr[*r])),

        // 2 children
        (ENodeOrVar::ENode(VecLang::Add(lefts)), VecLang::Add(rights))
        | (ENodeOrVar::ENode(VecLang::Mul(lefts)), VecLang::Mul(rights))
        | (ENodeOrVar::ENode(VecLang::Minus(lefts)), VecLang::Minus(rights))
        | (ENodeOrVar::ENode(VecLang::Div(lefts)), VecLang::Div(rights))
        | (ENodeOrVar::ENode(VecLang::Or(lefts)), VecLang::Or(rights))
        | (ENodeOrVar::ENode(VecLang::And(lefts)), VecLang::And(rights))
        | (ENodeOrVar::ENode(VecLang::Lt(lefts)), VecLang::Lt(rights))
        | (ENodeOrVar::ENode(VecLang::Get(lefts)), VecLang::Get(rights))
        | (ENodeOrVar::ENode(VecLang::Concat(lefts)), VecLang::Concat(rights))
        | (ENodeOrVar::ENode(VecLang::VecAdd(lefts)), VecLang::VecAdd(rights))
        | (ENodeOrVar::ENode(VecLang::VecMinus(lefts)), VecLang::VecMinus(rights))
        | (ENodeOrVar::ENode(VecLang::VecMul(lefts)), VecLang::VecMul(rights))
        | (ENodeOrVar::ENode(VecLang::VecDiv(lefts)), VecLang::VecDiv(rights)) => lefts
            .iter()
            .zip(rights.iter())
            .all(|(l, r)| match_recexpr(&pattern, &pattern[*l], &expr, &expr[*r])),

        // 3 children
        (ENodeOrVar::ENode(VecLang::Ite(lefts)), VecLang::Ite(rights))
        | (ENodeOrVar::ENode(VecLang::VecMAC(lefts)), VecLang::VecMAC(rights)) => lefts
            .iter()
            .zip(rights.iter())
            .all(|(l, r)| match_recexpr(&pattern, &pattern[*l], &expr, &expr[*r])),

        // n childen
        (ENodeOrVar::ENode(VecLang::List(lefts)), VecLang::List(rights))
        | (ENodeOrVar::ENode(VecLang::Vec(lefts)), VecLang::Vec(rights))
        | (ENodeOrVar::ENode(VecLang::LitVec(lefts)), VecLang::LitVec(rights)) => lefts
            .iter()
            .zip(rights.iter())
            .all(|(l, r)| match_recexpr(&pattern, &pattern[*l], &expr, &expr[*r])),

        // else, we checked everything, return false
        (ENodeOrVar::ENode(_), _) => false,

        // if the pattern is a variable, it matches anything
        (ENodeOrVar::Var(_), _) => true,
    }
}

/// Returns the number of times `lhs` matches `expr` using an e-graph.
fn match_recexpr_egraph(
    lhs: &Arc<dyn egg::Searcher<VecLang, TrackRewrites> + Send + Sync>,
    expr: &egg::RecExpr<VecLang>,
) -> usize {
    let mut egraph: EGraph<VecLang, TrackRewrites> =
        EGraph::new(TrackRewrites::default());
    egraph.add_expr(&expr);
    egraph.rebuild();
    let matches = lhs.search(&egraph);
    let x = matches.len();
    // if x != 0 {
    //     if let Some(ast) = lhs.get_pattern_ast() {
    //         eprintln!("matches({}, {}) = {x}", ast.pretty(80), expr.pretty(80));
    //     } else {
    //         eprintln!("matches({}) = {x}", expr.pretty(80));
    //     }
    //     for m in matches {
    //         eprintln!(
    //             "{m:?} {}",
    //             m.ast
    //                 .as_ref()
    //                 .map_or_else(|| "".to_string(), |x| x.pretty(80))
    //         );
    //     }
    // }
    x
}

/// Defines a cost function based on the rules in a phase.
/// Roughly, this looks at the LHS of each rule in a phase,
/// and gives expressions that match some LHS a low cost.
pub struct PhaseCostFn {
    rules: Vec<DiosRwrite>,
    expr: RecExpr<VecLang>,
}

impl PhaseCostFn {
    pub fn from_rules(rules: Vec<DiosRwrite>, expr: RecExpr<VecLang>) -> Self {
        for r in &rules {
            if let Some(r) = r.searcher.get_pattern_ast() {
                eprintln!("lhs: {}", r.pretty(80));
            }
        }
        PhaseCostFn { rules, expr }
    }
}

impl CostFunction<VecLang> for PhaseCostFn {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &VecLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        const BIG: f64 = 100.0;

        let expr: RecExpr<VecLang> = enode.build_recexpr(|id| self.expr[id].clone());
        let raw_this_cost: f64 = self
            .rules
            .iter()
            .flat_map(|rw| rw.searcher.get_pattern_ast())
            .fold(0, |acc, it| {
                if match_recexpr(
                    it,
                    recexpr_helpers::root(it),
                    &expr,
                    recexpr_helpers::root(&expr),
                ) {
                    acc + 1
                } else {
                    acc
                }
            }) as f64;

        let raw_tot_cost = raw_this_cost
            + match enode {
                VecLang::Num(_) | VecLang::Symbol(_) => 0.0,
                // 1 children
                VecLang::Sgn(ids)
                | VecLang::Sqrt(ids)
                | VecLang::Neg(ids)
                | VecLang::VecNeg(ids)
                | VecLang::VecSqrt(ids)
                | VecLang::VecSgn(ids) => {
                    ids.iter().fold(0.0, |acc, it| acc + costs(*it))
                }

                // 2 children
                VecLang::Add(ids)
                | VecLang::Mul(ids)
                | VecLang::Minus(ids)
                | VecLang::Div(ids)
                | VecLang::Or(ids)
                | VecLang::And(ids)
                | VecLang::Lt(ids)
                | VecLang::Get(ids)
                | VecLang::Concat(ids)
                | VecLang::VecAdd(ids)
                | VecLang::VecMinus(ids)
                | VecLang::VecMul(ids)
                | VecLang::VecDiv(ids) => {
                    ids.iter().fold(0.0, |acc, it| acc + costs(*it))
                }

                // 3 children
                VecLang::VecMAC(ids) | VecLang::Ite(ids) => {
                    ids.iter().fold(0.0, |acc, it| acc + costs(*it))
                }

                // n children
                VecLang::List(ids) | VecLang::Vec(ids) | VecLang::LitVec(ids) => {
                    ids.iter().fold(0.0, |acc, it| acc + costs(*it))
                }
            };

        // add small amount to raw_cost in case it's 0
        1.0 / (raw_tot_cost + 0.01)
    }
}
