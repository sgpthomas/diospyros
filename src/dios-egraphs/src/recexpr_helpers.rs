use egg::{Id, Language, RecExpr};

pub fn walk_recexpr_help<F, L: Language>(expr: &RecExpr<L>, id: Id, action: F) -> F
where
    F: FnMut(&L),
{
    let mut f = action;
    f(&expr[id]);
    for c in expr[id].children() {
        let newf = walk_recexpr_help(expr, *c, f);
        f = newf;
    }
    f
}

pub fn walk_recexpr<F, L: Language>(expr: &RecExpr<L>, action: F)
where
    F: FnMut(&L),
{
    walk_recexpr_help(expr, (expr.as_ref().len() - 1).into(), action);
}

pub fn fold_recexpr<F, L: Language, T>(expr: &RecExpr<L>, init: T, mut action: F) -> T
where
    F: FnMut(T, &L) -> T,
    T: Clone,
{
    let mut acc = init;
    walk_recexpr(expr, |l| acc = action(acc.clone(), l));
    acc
}

pub fn root<L: Language>(expr: &RecExpr<L>) -> &L {
    &expr.as_ref()[expr.as_ref().len() - 1]
}
