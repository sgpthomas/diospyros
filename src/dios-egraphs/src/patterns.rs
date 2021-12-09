use std::collections::HashMap;

use egg::{ENodeOrVar, Id, Language, Pattern, RecExpr};

use crate::veclang::VecLang;

fn find_root(expr: &RecExpr<VecLang>) -> Id {
    let root = 0;
    // stores a mapping from ENodes to things that have that enode as a child
    // nodes
    let mut mappings: HashMap<Id, Vec<usize>> = HashMap::default();
    for (idx, e) in expr.as_ref().iter().enumerate() {
        e.for_each(|child_id| {
            mappings
                .entry(child_id)
                .and_modify(|v| v.push(idx))
                .or_insert_with(|| vec![idx]);
        })
    }

    for (k, v) in mappings.iter() {
        if v.len() > 1 {
            eprintln!("{} -> {:?}", k, v);
        }
    }

    root.into()
}

pub fn gen_patterns(root: &Id, expr: &RecExpr<VecLang>) -> Vec<Pattern<VecLang>> {
    // let pat: Pattern<_> = expr.as_ref().into();
    let _x = expr[*root].clone().map_children(|id| {
        eprintln!("{}", id);
        id
    });

    find_root(expr);

    // eprintln!("pat: [{}] {:?}", root, );

    todo!();
}
