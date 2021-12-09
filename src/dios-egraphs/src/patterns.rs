use egg::{ENodeOrVar, Id, Pattern, RecExpr};

use crate::veclang::VecLang;

pub fn gen_patterns(root: &Id, expr: &RecExpr<VecLang>) -> Vec<Pattern<VecLang>> {
    // let pat: Pattern<_> = expr.as_ref().into();

    eprintln!("pat: [{}] {:?}", root, expr[*root]);

    todo!();
}
