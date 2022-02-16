use std::io::Write;

use crate::veclang::VecLang;
use egg::TreeExplanation;

/// Get a flat list of all the rewrites used in a tree explanation
#[allow(unused)]
pub fn get_rewrites_used(tree: &TreeExplanation<VecLang>) -> Vec<String> {
    let mut rules: Vec<String> = vec![];
    for term in tree {
        if let Some(r) = &term.backward_rule {
            rules.push(format!("<={}", r.as_str()));
        }

        if let Some(r) = &term.forward_rule {
            rules.push(format!("=>{}", r.as_str()));
        }

        for child in &term.child_proofs {
            rules.append(&mut get_rewrites_used(&child));
        }
    }
    rules
}

#[allow(unused)]
pub fn print_rewrites_used(write: &mut Box<dyn Write>, pre: &str, rules: &[String]) {
    for r in rules {
        writeln!(write, "{}{}", pre, r);
    }
}

/// Checks whether a rewrite rule is used at the toplevel.
#[allow(unused)]
pub fn is_rewrite_used<S>(rule_name: S, tree: &TreeExplanation<VecLang>) -> bool
where
    S: AsRef<str>,
{
    for term in tree {
        if let Some(r) = &term.backward_rule {
            return r.as_str() == rule_name.as_ref();
        }

        if let Some(r) = &term.forward_rule {
            return r.as_str() == rule_name.as_ref();
        }
    }
    false
}
