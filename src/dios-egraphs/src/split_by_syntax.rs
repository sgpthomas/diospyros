use egg::{ENodeOrVar, Id, PatternAst};

use crate::{
    recexpr_helpers::fold_recexpr,
    rules::Phase,
    veclang::{DiosRwrite, VecLang},
};

/// If there is a `(Vec ..)` somewhere in `expr`,
/// `has_vec(expr)` will return `true`.
fn has_vec(expr: &PatternAst<VecLang>) -> bool {
    fold_recexpr(expr, false, |acc, l| {
        acc || match l {
            ENodeOrVar::ENode(VecLang::Vec(_)) => true,
            _ => false,
        }
    })
}

/// If the root of `expr` is a `VecOp`, this function
/// returns `true`.
fn is_vec_op(expr: &PatternAst<VecLang>) -> bool {
    let root: Id = (expr.as_ref().len() - 1).into();
    if let ENodeOrVar::ENode(root_node) = &expr[root] {
        match root_node {
            // non vectors
            VecLang::Num(_)
            | VecLang::Add(_)
            | VecLang::Mul(_)
            | VecLang::Minus(_)
            | VecLang::Div(_)
            | VecLang::Or(_)
            | VecLang::And(_)
            | VecLang::Ite(_)
            | VecLang::Lt(_)
            | VecLang::Sgn(_)
            | VecLang::Sqrt(_)
            | VecLang::Neg(_)
            | VecLang::List(_)
            | VecLang::Vec(_)
            | VecLang::LitVec(_)
            | VecLang::Get(_)
            | VecLang::Concat(_)
            | VecLang::Symbol(_) => false,
            // vectors
            VecLang::VecAdd(_)
            | VecLang::VecMinus(_)
            | VecLang::VecMul(_)
            | VecLang::VecDiv(_)
            | VecLang::VecNeg(_)
            | VecLang::VecSqrt(_)
            | VecLang::VecSgn(_)
            | VecLang::VecMAC(_) => true,
        }
    } else {
        false
    }
}

pub fn phases(rule: &DiosRwrite) -> Phase {
    if let (Some(lhs), Some(rhs)) = (
        rule.searcher.get_pattern_ast(),
        rule.applier.get_pattern_ast(),
    ) {
        if !has_vec(lhs) && !has_vec(rhs) && !is_vec_op(lhs) && !is_vec_op(rhs) {
            Phase::PreCompile
        } else if (has_vec(lhs) && is_vec_op(rhs)) || (is_vec_op(lhs) && has_vec(rhs)) {
            Phase::Compile
        } else {
            // default is Phase::Opt
            Phase::Opt
        }
    } else {
        // if we can't get an ast, just throw the rule into the last phase.
        // probably should come up with a better solution at some point.
        Phase::Opt
    }
}
