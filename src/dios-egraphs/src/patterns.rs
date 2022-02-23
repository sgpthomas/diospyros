use std::{collections::HashMap, str::FromStr};

use egg::{ENodeOrVar, Id, Language, Pattern, RecExpr, Var};
use itertools::Itertools;

use crate::veclang::VecLang;

fn fresh_symbol(n: u32) -> Var {
    Var::from_str(&format!("?s{}", n)).unwrap()
}

fn depth_help<L: Language>(expr: &RecExpr<L>, id: &[Id]) -> u32 {
    id.iter()
        .map(|x| {
            if expr[*x].is_leaf() {
                0
            } else {
                depth_help(expr, expr[*x].children()) + 1
            }
        })
        .max()
        .unwrap()
}

fn depth<L: Language>(expr: &RecExpr<L>) -> u32 {
    let root: Id = (expr.as_ref().len() - 1).into();
    depth_help(expr, &[root])
}

fn height<L: Language>(expr: &RecExpr<L>, id: &Id) -> u32 {
    let mut ref_id: Option<Id> = None;

    for (i, node) in expr.as_ref().iter().enumerate() {
        if node.any(|x| &x == id) {
            // eprintln!("{:?}", node);
            ref_id = Some(i.into());
            break;
        }
    }
    if let Some(x) = ref_id {
        1 + height(expr, &x)
    } else {
        0
    }
}

fn subtree<L: Language + std::fmt::Display>(expr: &RecExpr<L>, id: Id) -> RecExpr<L> {
    expr[id].build_recexpr(|x| expr[x].clone())
}

// fn replace_gets(expr: &RecExpr<VecLang>, id: Id) ->
fn replace_get(pat: &Pattern<VecLang>) -> Pattern<VecLang> {
    let root: Id = (pat.ast.as_ref().len() - 1).into();
    let mut i = 0;
    pat.ast[root]
        .build_recexpr(|x| {
            if matches!(pat.ast[x], ENodeOrVar::ENode(VecLang::Get(_))) {
                let sym = fresh_symbol(i);
                i += 1;
                ENodeOrVar::Var(sym)
            } else {
                pat.ast[x].clone()
            }
        })
        .alpha_rename()
        .into()
}

fn should_replace(pat: &Pattern<VecLang>, id: &Id) -> bool {
    // pat.ast[*id].is_leaf()
    let c0 = pat.ast[*id].all(|i| pat.ast[i].is_leaf());
    let h = height(&pat.ast, id);
    let c1 = h + 3 >= depth(&pat.ast);
    // eprintln!("{}, {}", h, depth(&pat.ast));
    c0 && c1 && !pat.ast[*id].is_leaf()
}

fn gen_pattern_replace_leaves(pat: &Pattern<VecLang>) -> Option<Pattern<VecLang>> {
    let root: Id = (pat.ast.as_ref().len() - 1).into();

    if pat.ast[root].is_leaf() {
        None
    } else {
        let mut i = 0;
        Some(
            pat.ast[root]
                .build_recexpr(|x| {
                    if should_replace(&pat, &x) {
                        eprintln!("repl {}", pat.ast[x]);
                        let sym = fresh_symbol(i);
                        i += 1;
                        ENodeOrVar::Var(sym)
                    } else {
                        pat.ast[x].clone()
                    }
                })
                .alpha_rename()
                .into(),
        )
    }
}

fn gen_patterns_of_depth(pat: &Pattern<VecLang>, d: u32) -> Vec<Pattern<VecLang>> {
    let root: Id = (pat.ast.as_ref().len() - 1).into();

    let mut pats: Vec<Pattern<VecLang>> = vec![];
    let mut children = vec![root];
    while let Some(c) = children.pop() {
        if depth_help(&pat.ast, &[c]) == d {
            pats.push(subtree(&pat.ast, c).alpha_rename().into());
        } else {
            children.extend(pat.ast[c].children());
        }
    }

    pats
}

#[allow(unused)]
fn enumerate_patterns_by_depth(expr: &RecExpr<VecLang>) -> Vec<Pattern<VecLang>> {
    // let mut patterns = vec![];
    let mut pattern_map: HashMap<RecExpr<ENodeOrVar<VecLang>>, u32> = HashMap::new();

    let mut next: Pattern<_> = replace_get(&expr.as_ref().into());

    let mut limit = 100;
    let depth_limit = (2, 100);
    let count_limit = 0;

    while let Some(new_pat) = gen_pattern_replace_leaves(&next) {
        // panic!("asdf");
        for d in depth_limit.0..u32::min(depth_limit.1, depth(&new_pat.ast)) {
            let new_pats = gen_patterns_of_depth(&new_pat, d);
            // eprintln!(
            //     "new_pat [{}/{}]: {}",
            //     d,
            //     depth(&new_pat.ast),
            //     new_pat.pretty(80)
            // );
            // patterns.extend(new_pats.clone());
            for pat in new_pats {
                // eprintln!("  - found: {}", pat.pretty(80));
                pattern_map
                    .entry(pat.ast.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }

        // patterns.push(new_pat.clone());
        next = new_pat;

        if limit == 0 {
            break;
        } else {
            limit -= 1;
        }
    }

    // eprintln!(
    //     "{}",
    //     pattern_map
    //         .iter()
    //         .sorted_by_key(|&(_pat, count)| *count)
    //         // .filter(|&(pat, count)| *count >= count_limit && depth(pat) > depth_limit)
    //         .filter(|(pat, &count)| {
    //             count >= count_limit && depth(&pat) > depth_limit.0 && depth(&pat) < depth_limit.1
    //         })
    //         .map(|(pat, count)| format!("pat: [{}] {}", count, pat))
    //         .join("\n")
    // );
    // panic!("asdf");

    // patterns
    //     .into_iter()
    //     .unique_by(|p| p.ast.clone())
    //     .collect_vec()
    pattern_map
        .into_iter()
        .sorted_by_key(|(_pat, count)| *count)
        .filter_map(|(pat, count)| {
            if count >= count_limit
                && depth(&pat) > depth_limit.0
                && depth(&pat) < depth_limit.1
            {
                Some(pat.into())
            } else {
                None
            }
        })
        .collect()
}

#[allow(unused)]
fn enumerate_patterns_by_children(expr: &RecExpr<VecLang>) -> Vec<Pattern<VecLang>> {
    let root: Id = (expr.as_ref().len() - 1).into();

    let mut patterns: Vec<Pattern<VecLang>> = vec![];

    expr[root].build_recexpr(|x| {
        if matches!(expr[x], VecLang::Vec(_)) {
            eprintln!("{:?}", expr[x]);
            patterns.push(replace_get(&subtree(expr, x).as_ref().into()));
        }
        expr[x].clone()
    });

    for p in &patterns {
        eprintln!("p: {}", p.pretty(80));
    }

    patterns
}

#[allow(unused)]
pub fn gen_patterns(expr: &RecExpr<VecLang>) -> Vec<Pattern<VecLang>> {
    enumerate_patterns_by_children(expr)
}
