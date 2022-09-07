use egg::*;

use crate::veclang::{DiosRwrite, VecLang};

pub struct VecCostFn;

// &'a EGraph
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
