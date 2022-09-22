use std::{collections::HashMap, fmt::Display};

use egg::{Language, Searcher};
use itertools::Itertools;

use crate::recexpr_helpers::LanguageHelpers;

struct HeadlessPattern<L: egg::Language + Display> {
    legs: Vec<egg::Pattern<L>>,
}

impl<L: egg::Language + Display> From<&egg::PatternAst<L>> for HeadlessPattern<L> {
    fn from(pattern: &egg::PatternAst<L>) -> Self {
        let root: egg::Id = (pattern.as_ref().len() - 1).into();
        let legs = match &pattern[root] {
            egg::ENodeOrVar::ENode(node) => node
                .children()
                .iter()
                .map(|child| {
                    egg::Pattern::new(
                        pattern[*child].build_recexpr(|x| pattern[x].clone()),
                    )
                })
                .collect(),
            egg::ENodeOrVar::Var(_) => vec![],
        };
        HeadlessPattern { legs }
    }
}

impl<L: egg::Language + Display> HeadlessPattern<L> {
    fn search_eclass<N: egg::Analysis<L>>(
        &self,
        egraph: &egg::EGraph<L, N>,
        eclass: egg::Id,
    ) -> Vec<L> {
        let class = &egraph[eclass];
        class
            .nodes
            .iter()
            // we only want to keep nodes that have at least one child different
            // than itself
            .filter(|node| node.all(|id| id != eclass))
            // we only want to consider nodes with the same number as legs as self
            .filter(|node| node.children().len() == self.legs.len())
            .map(|node| {
                node.children()
                    .iter()
                    .zip(self.legs.iter())
                    .filter_map(|(child_id, pattern)| {
                        pattern
                            .search_eclass(&egraph, *child_id)
                            .map(|res| res.reify())
                    })
                    .collect::<Vec<_>>()
            })
            // only keep matches that match every leg pattern
            .filter(|l| l.len() == self.legs.len())
            .flatten()
            .flatten()
            // we only want to count each child once here.
            .unique()
            .collect()
    }
}

trait Reify<L: egg::Language + Display> {
    fn reify(&self) -> Vec<L>;
}

impl<'a, L: egg::Language + Display> Reify<L> for egg::SearchMatches<'a, L> {
    fn reify(&self) -> Vec<L> {
        self.substs
            .iter()
            .filter_map(|subst| {
                if let Some(ast) = &self.ast {
                    Some((subst, ast))
                } else {
                    None
                }
            })
            .map(|(subst, ast)| {
                let mut owned = ast.as_ref().as_ref().to_vec();
                owned.iter_mut().rev().for_each(|l| {
                    l.update_children(|id| match &ast[id] {
                        egg::ENodeOrVar::ENode(_) => id,
                        egg::ENodeOrVar::Var(var) => *(subst.get(*var).unwrap()),
                    })
                });
                let blah: Vec<L> = owned
                    .into_iter()
                    .filter_map(|l| match l {
                        egg::ENodeOrVar::ENode(inner) => Some(inner),
                        egg::ENodeOrVar::Var(_) => None,
                    })
                    .collect();
                blah
            })
            .flatten()
            .collect()
    }
}

pub struct TopDownExtractor<
    'a,
    L: egg::Language + Display,
    N: egg::Analysis<L>,
    CF: egg::CostFunction<L>,
> {
    node_score_map: HashMap<(egg::Id, L), i64>,
    egraph: &'a egg::EGraph<L, N>,
    patterns: &'a [std::sync::Arc<dyn egg::Searcher<L, N> + Send + Sync>],
    extractor: egg::Extractor<'a, CF, L, N>,
}

impl<'a, L: egg::Language + Display, N: egg::Analysis<L>, CF: egg::CostFunction<L>>
    TopDownExtractor<'a, L, N, CF>
{
    /// Create a new `TopDownExtractor`.
    pub fn new(
        egraph: &'a egg::EGraph<L, N>,
        patterns: &'a [std::sync::Arc<dyn egg::Searcher<L, N> + Send + Sync>],
        cost_fn: CF,
    ) -> Self {
        let extractor = egg::Extractor::new(&egraph, cost_fn);
        let mut td_extractor = Self {
            node_score_map: HashMap::default(),
            egraph,
            patterns,
            extractor,
        };

        td_extractor.find_scores();
        td_extractor
    }

    pub fn find_best(self, root: egg::Id) -> egg::RecExpr<L> {
        let root_node = self.select_best_node(None, root).clone();
        // eprintln!("root: {:?}", root_node);
        root_node.build_recexpr_w_parent(root, &|parent_id, id| {
            self.select_best_node(Some(parent_id), id).clone()
        })
    }

    fn select_best_node(&self, parent_id: Option<egg::Id>, eclass_id: egg::Id) -> &L {
        if let Some(parent_id) = parent_id {
            let node = self.egraph[eclass_id]
                .nodes
                .iter()
                .map(|node| {
                    (
                        self.node_score_map
                            .get(&(parent_id, node.clone()))
                            .map(|x| *x),
                        node,
                    )
                })
                .filter_map(|(score, node)| score.map(|s| (s, node)))
                .max_by_key(|(score, _node)| *score);
            // eprintln!("  score: ({parent_id}, {eclass_id}) -> {node:?}");
            match node {
                Some((_score, l)) => l,
                None => self.extractor.find_best_node(eclass_id),
            }
        } else {
            &self.egraph[eclass_id].nodes[0]
        }
        // eprintln!("finding best for {eclass_id}");
    }

    fn find_scores(&mut self) {
        // go through each class, and assign all children nodes a score
        for (_i, class) in self.egraph.classes().enumerate() {
            // eprintln!("id: {i} =====");
            // eprintln!("{class:?}");
            for pat in self.patterns {
                if let Some(p_ast) = pat.get_pattern_ast() {
                    let headless: HeadlessPattern<_> = p_ast.into();
                    let n_matches = headless.search_eclass(&self.egraph, class.id);

                    // if !n_matches.is_empty() {
                    //     eprintln!("pattern: {}", p_ast.pretty(80));
                    //     eprintln!("{:?} -> {n_matches:?}", class.nodes);
                    // }

                    for m in n_matches {
                        self.node_score_map
                            .entry((class.id, m))
                            .and_modify(|x| *x += 1)
                            .or_insert(1);
                    }
                }
            }
        }

        // now assign each eclass a score. we need to do this in a separate
        // pass because the other loop is updating the children of each eclass

        // eprintln!(
        //     "{{\n{}\n}}",
        //     self.node_score_map
        //         .iter()
        //         .map(|((id, l), v)| format!("  {id},{l:?} -> {v}"))
        //         .join("\n")
        // );
    }
}
