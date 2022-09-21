use std::{collections::HashMap, fmt::Display};

use egg::{Language, Searcher};
use itertools::Itertools;

pub struct TopDownExtractor<'a, L: egg::Language + Display, N: egg::Analysis<L>> {
    score_map: HashMap<L, f64>,
    egraph: &'a egg::EGraph<L, N>,
    patterns: &'a [std::sync::Arc<dyn egg::Searcher<L, N> + Send + Sync>],
}

struct RootlessPattern<L: egg::Language + Display> {
    legs: Vec<egg::Pattern<L>>,
}

impl<L: egg::Language + Display> From<&egg::PatternAst<L>> for RootlessPattern<L> {
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
        RootlessPattern { legs }
    }
}

impl<L: egg::Language + Display> RootlessPattern<L> {
    fn search_eclass<N: egg::Analysis<L>>(
        &self,
        egraph: &egg::EGraph<L, N>,
        eclass: egg::Id,
    ) -> Vec<L> {
        let class = &egraph[eclass];
        class
            .nodes
            .iter()
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
                    .flatten()
                    .collect::<Vec<_>>()
            })
            .flatten()
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

impl<'a, L: egg::Language + Display, N: egg::Analysis<L>> TopDownExtractor<'a, L, N> {
    /// Create a new `TopDownExtractor`.
    pub fn new(
        egraph: &'a egg::EGraph<L, N>,
        patterns: &'a [std::sync::Arc<dyn egg::Searcher<L, N> + Send + Sync>],
    ) -> Self {
        let mut extractor = Self {
            score_map: HashMap::default(),
            egraph,
            patterns,
        };

        for p in extractor.patterns {
            if let Some(a) = p.get_pattern_ast() {
                eprintln!("{}", a.pretty(80));
            }
        }

        extractor.find_scores();
        extractor
    }

    pub fn find_best(&self, root: egg::Id) -> (f64, egg::RecExpr<L>) {
        todo!()
    }

    fn find_scores(&mut self) {
        // go through each class, and assign all children nodes a score
        for (i, class) in self.egraph.classes().enumerate() {
            eprintln!("id: {i} =====");
            eprintln!("{class:?}");
            for pat in self.patterns {
                eprintln!("pattern: {}", pat.get_pattern_ast().unwrap().pretty(80),);

                let rootless: RootlessPattern<_> = pat.get_pattern_ast().unwrap().into();
                let n_matches = rootless.search_eclass(&self.egraph, class.id);

                eprintln!("{:?} -> {n_matches:?}", class.nodes);

                for m in n_matches {
                    self.score_map
                        .entry(m)
                        .and_modify(|x| *x += 1.0)
                        .or_insert(1.0);
                }
            }
        }

        eprintln!("{:#?}", self.score_map);
    }
}
