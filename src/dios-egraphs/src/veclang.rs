use egg::{define_language, ENodeOrVar, Id, Language, Pattern, PatternAst, RecExpr};
use itertools::Itertools;

use crate::tracking::TrackRewrites;

define_language! {
    pub enum VecLang {
        Num(i32),

        // Id is a key to identify EClasses within an EGraph, represents
        // children nodes
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "-" = Minus([Id; 2]),
        "/" = Div([Id; 2]),

        "or" = Or([Id; 2]),
        "&&" = And([Id; 2]),
        "ite" = Ite([Id; 3]),
        "<" = Lt([Id; 2]),

        "sgn" = Sgn([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "neg" = Neg([Id; 1]),

        // Lists have a variable number of elements
        "List" = List(Box<[Id]>),

        // Vectors have width elements
        "Vec" = Vec(Box<[Id]>),

        // Vector with all literals
        "LitVec" = LitVec(Box<[Id]>),

        "Get" = Get([Id; 2]),

        // Used for partitioning and recombining lists
        "Concat" = Concat([Id; 2]),

        // Vector operations that take 2 vectors of inputs
        "VecAdd" = VecAdd([Id; 2]),
        "VecMinus" = VecMinus([Id; 2]),
        "VecMul" = VecMul([Id; 2]),
        "VecDiv" = VecDiv([Id; 2]),
        // "VecMulSgn" = VecMulSgn([Id; 2]),

        // Vector operations that take 1 vector of inputs
        "VecNeg" = VecNeg([Id; 1]),
        "VecSqrt" = VecSqrt([Id; 1]),
        "VecSgn" = VecSgn([Id; 1]),

        // MAC takes 3 lists: acc, v1, v2
        "VecMAC" = VecMAC([Id; 3]),

        // language items are parsed in order, and we want symbol to
        // be a fallback, so we put it last.
        // `Symbol` is an egg-provided interned string type
        Symbol(egg::Symbol),
    }
}

pub type EGraph = egg::EGraph<VecLang, TrackRewrites>;
pub type DiosRwrite = egg::Rewrite<VecLang, TrackRewrites>;

impl VecLang {
    pub fn from_pattern(pat: &PatternAst<Self>) -> RecExpr<Self> {
        pat.as_ref()
            .iter()
            .map(|node| match node {
                ENodeOrVar::Var(v) => VecLang::Symbol(v.to_string().into()),
                ENodeOrVar::ENode(node) => node.clone(),
            })
            .collect_vec()
            .into()
    }
}

// ===================== Vec Ast ===================== //

#[derive(Debug, Clone)]
enum Lang {
    Add(Box<Lang>, Box<Lang>),
    Mul(Box<Lang>, Box<Lang>),
    Minus(Box<Lang>, Box<Lang>),
    Div(Box<Lang>, Box<Lang>),

    Or(Box<Lang>, Box<Lang>),
    And(Box<Lang>, Box<Lang>),
    #[allow(unused)]
    Ite(Box<Lang>, Box<Lang>, Box<Lang>),
    Lt(Box<Lang>, Box<Lang>),

    Neg(Box<Lang>),

    Vec(Vec<Lang>),

    VecAdd(Box<Lang>, Box<Lang>),
    VecMul(Box<Lang>, Box<Lang>),
    VecMinus(Box<Lang>, Box<Lang>),
    VecDiv(Box<Lang>, Box<Lang>),

    VecNeg(Box<Lang>),
    #[allow(unused)]
    VecSqrt(Box<Lang>),
    #[allow(unused)]
    VecSgn(Box<Lang>),

    VecMAC(Box<Lang>, Box<Lang>, Box<Lang>),

    Num(i32),
    Symbol(String),
}

impl Lang {
    fn to_recexpr(&self, expr: &mut egg::RecExpr<VecLang>) -> Id {
        match &self {
            Lang::Add(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::Add([left_id, right_id]))
            }
            Lang::Mul(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::Mul([left_id, right_id]))
            }
            Lang::Minus(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::Minus([left_id, right_id]))
            }
            Lang::Div(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::Div([left_id, right_id]))
            }
            Lang::Or(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::Or([left_id, right_id]))
            }
            Lang::And(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::And([left_id, right_id]))
            }
            Lang::Lt(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::Lt([left_id, right_id]))
            }
            Lang::VecAdd(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::VecAdd([left_id, right_id]))
            }
            Lang::VecMul(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::VecMul([left_id, right_id]))
            }
            Lang::VecMinus(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::VecMinus([left_id, right_id]))
            }
            Lang::VecDiv(left, right) => {
                let left_id = left.to_recexpr(expr);
                let right_id = right.to_recexpr(expr);
                expr.add(VecLang::VecDiv([left_id, right_id]))
            }
            Lang::Ite(_, _, _) => todo!(),
            Lang::Neg(inner) => {
                let id = inner.to_recexpr(expr);
                expr.add(VecLang::Neg([id]))
            }
            Lang::VecNeg(inner) => {
                let id = inner.to_recexpr(expr);
                expr.add(VecLang::VecNeg([id]))
            }
            Lang::VecSqrt(inner) => {
                let id = inner.to_recexpr(expr);
                expr.add(VecLang::VecSqrt([id]))
            }
            Lang::VecSgn(inner) => {
                let id = inner.to_recexpr(expr);
                expr.add(VecLang::VecSgn([id]))
            }
            Lang::VecMAC(a, b, c) => {
                let a_id = a.to_recexpr(expr);
                let b_id = b.to_recexpr(expr);
                let c_id = c.to_recexpr(expr);
                expr.add(VecLang::VecMAC([a_id, b_id, c_id]))
            }
            Lang::Vec(items) => {
                let ids = items.iter().map(|it| it.to_recexpr(expr)).collect_vec();
                expr.add(VecLang::Vec(ids.into_boxed_slice()))
            }
            // Lang::Const(v) => expr.add(VecLang::Const(v.clone())),
            Lang::Num(i) => expr.add(VecLang::Num(*i)),
            Lang::Symbol(s) => expr.add(VecLang::Symbol(s.into())),
        }
    }
}

fn subtree(expr: &RecExpr<VecLang>, new_root: Id) -> RecExpr<VecLang> {
    expr[new_root].build_recexpr(|id| expr[id].clone())
}

impl From<RecExpr<VecLang>> for Lang {
    fn from(expr: RecExpr<VecLang>) -> Self {
        let root: Id = Id::from(expr.as_ref().len() - 1);
        match &expr[root] {
            VecLang::Num(i) => Lang::Num(*i),
            VecLang::Add([left, right]) => Lang::Add(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            VecLang::Mul([left, right]) => Lang::Mul(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            VecLang::Minus([left, right]) => Lang::Minus(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            VecLang::Div([left, right]) => Lang::Div(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            VecLang::Or([left, right]) => Lang::Or(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            VecLang::And([left, right]) => Lang::And(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            VecLang::Ite(_) => todo!(),
            VecLang::Lt([left, right]) => Lang::Lt(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            VecLang::Sgn(_) => todo!(),
            VecLang::Sqrt(_) => todo!(),
            VecLang::Neg([inner]) => Lang::Neg(Box::new(subtree(&expr, *inner).into())),
            VecLang::List(_) => todo!(),
            VecLang::Vec(items) => Lang::Vec(
                items
                    .iter()
                    .map(|id| subtree(&expr, *id).into())
                    .collect_vec(),
            ),
            VecLang::LitVec(_) => todo!(),
            VecLang::Get(_) => todo!(),
            VecLang::Concat(_) => todo!(),
            VecLang::VecAdd([left, right]) => Lang::VecAdd(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            VecLang::VecMinus([left, right]) => Lang::VecMinus(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            VecLang::VecMul([left, right]) => Lang::VecMul(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            VecLang::VecDiv([left, right]) => Lang::VecDiv(
                Box::new(subtree(&expr, *left).into()),
                Box::new(subtree(&expr, *right).into()),
            ),
            // VecLang::VecMulSgn(_) => todo!(),
            VecLang::VecNeg([inner]) => {
                Lang::VecNeg(Box::new(subtree(&expr, *inner).into()))
            }
            VecLang::VecSqrt(_) => todo!(),
            VecLang::VecSgn(_) => todo!(),
            VecLang::VecMAC([a, b, c]) => Lang::VecMAC(
                Box::new(subtree(&expr, *a).into()),
                Box::new(subtree(&expr, *b).into()),
                Box::new(subtree(&expr, *c).into()),
            ),
            // VecLang::Const(v) => Lang::Const(v.clone()),
            VecLang::Symbol(sym) => Lang::Symbol(sym.to_string()),
        }
    }
}

impl Into<egg::RecExpr<VecLang>> for Lang {
    fn into(self) -> egg::RecExpr<VecLang> {
        let mut expr = egg::RecExpr::default();
        self.to_recexpr(&mut expr);
        expr
    }
}

trait Desugar {
    fn desugar(self, n_lanes: usize) -> Self;
}

impl Desugar for Pattern<Lang> {
    fn desugar(self, n_lanes: usize) -> Self {
        VecLang::from_pattern(&self.ast).desugar()
    }
}

impl Desugar for Lang {
    /// Expand single-lane vector instructions to some number of lanes.
    fn desugar(self, n_lanes: usize) -> Self {
        match self {
            Lang::Vec(items) if items.len() == 1 => {
                let inner = match &items[0] {
                    Lang::Num(_) => todo!(),
                    Lang::Symbol(ref v) => (0..n_lanes)
                        .into_iter()
                        .map(|n| Lang::Symbol(format!("{v}{n}")))
                        .collect_vec(),
                    Lang::Add(l, r) => match ((**l).clone(), (**r).clone()) {
                        (Lang::Symbol(l), Lang::Symbol(r)) => {
                            let mut inner: Vec<Lang> = vec![];
                            for n in 0..n_lanes {
                                let l_n = Lang::Symbol(format!("{l}{n}").into());
                                let r_n = Lang::Symbol(format!("{r}{n}").into());
                                inner.push(Lang::Add(Box::new(l_n), Box::new(r_n)));
                            }
                            inner
                        }
                        _ => todo!(),
                    },
                    Lang::Mul(l, r) => match ((**l).clone(), (**r).clone()) {
                        (Lang::Symbol(l), Lang::Symbol(r)) => {
                            let mut inner: Vec<Lang> = vec![];
                            for n in 0..n_lanes {
                                let l_n = Lang::Symbol(format!("{l}{n}").into());
                                let r_n = Lang::Symbol(format!("{r}{n}").into());
                                inner.push(Lang::Mul(Box::new(l_n), Box::new(r_n)));
                            }
                            inner
                        }
                        _ => todo!(),
                    },
                    Lang::Minus(l, r) => match ((**l).clone(), (**r).clone()) {
                        (Lang::Symbol(l), Lang::Symbol(r)) => {
                            let mut inner: Vec<Lang> = vec![];
                            for n in 0..n_lanes {
                                let l_n = Lang::Symbol(format!("{l}{n}").into());
                                let r_n = Lang::Symbol(format!("{r}{n}").into());
                                inner.push(Lang::Minus(Box::new(l_n), Box::new(r_n)));
                            }
                            inner
                        }
                        _ => todo!(),
                    },
                    x => vec![x.clone()],
                };
                Lang::Vec(inner)
            }
            Lang::Vec(_items) => todo!(), // Lang::Vec(
            //     items
            //         .into_iter()
            //         .map(|item| item.desugar(n_lanes))
            //         .collect_vec(),
            // )
            Lang::Add(left, right) => Lang::Add(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::Mul(left, right) => Lang::Mul(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::Minus(left, right) => Lang::Minus(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::Div(left, right) => Lang::Div(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::Or(left, right) => Lang::Or(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::And(left, right) => Lang::And(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::Ite(_, _, _) => todo!(),
            Lang::Lt(left, right) => Lang::Lt(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::Neg(inner) => Lang::Neg(Box::new(inner.desugar(n_lanes))),
            Lang::VecAdd(left, right) => Lang::VecAdd(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::VecMul(left, right) => Lang::VecMul(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::VecMinus(left, right) => Lang::VecMinus(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::VecDiv(left, right) => Lang::VecDiv(
                Box::new(left.desugar(n_lanes)),
                Box::new(right.desugar(n_lanes)),
            ),
            Lang::VecNeg(inner) => Lang::VecNeg(Box::new(inner.desugar(n_lanes))),
            Lang::VecSqrt(inner) => Lang::VecSqrt(Box::new(inner.desugar(n_lanes))),
            Lang::VecSgn(inner) => Lang::VecSgn(Box::new(inner.desugar(n_lanes))),
            Lang::VecMAC(a, b, c) => Lang::VecMAC(
                Box::new(a.desugar(n_lanes)),
                Box::new(b.desugar(n_lanes)),
                Box::new(c.desugar(n_lanes)),
            ),

            x @ Lang::Num(_) => x,
            x @ Lang::Symbol(_) => x,
        }
    }
}
