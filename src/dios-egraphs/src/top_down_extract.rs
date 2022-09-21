use std::{collections::HashMap, fmt::Display};

use egg::{Language, Searcher};

pub struct TopDownExtractor<'a, L: egg::Language, N: egg::Analysis<L>> {
    score_map: HashMap<L, f64>,
    egraph: &'a egg::EGraph<L, N>,
    patterns: &'a [std::sync::Arc<dyn egg::Searcher<L, N> + Send + Sync>],
}

struct RootlessPattern<L: egg::Language> {
    legs: Vec<egg::Pattern<L>>,
}

impl<L: egg::Language> From<&egg::PatternAst<L>> for RootlessPattern<L> {
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

impl<L: egg::Language> RootlessPattern<L> {
    fn search_eclass<N: egg::Analysis<L>>(
        &self,
        egraph: &egg::EGraph<L, N>,
        eclass: egg::Id,
    ) -> usize {
        let class = &egraph[eclass];
        let mut count = 0;
        for node in &class.nodes {
            let children = node.children();
            if children.len() == self.legs.len() {
                for (child_id, pattern) in children.iter().zip(self.legs.iter()) {
                    if let Some(res) = pattern.search_eclass(&egraph, *child_id) {
                        eprintln!("{:?}", res.reify());
                        count += 1;
                    }
                }
            }
        }
        count
    }
}

trait Reify<L: egg::Language> {
    fn reify(&self) -> Vec<egg::RecExpr<L>>;
}

impl<'a, L: egg::Language> Reify<L> for egg::SearchMatches<'a, L> {
    fn reify(&self) -> Vec<egg::RecExpr<L>> {
        if let Some(ast) = &self.ast {
            let hack: Vec<L> = ast
                .as_ref()
                .as_ref()
                .iter()
                .flat_map(|n| match n {
                    egg::ENodeOrVar::ENode(x) => Some(x.clone()),
                    egg::ENodeOrVar::Var(_) => None,
                })
                .collect();
            let r: egg::RecExpr<L> = hack.into();
            vec![]
        } else {
            panic!()
        }
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

    /// Use a fixed point algorithm to calculate the score
    /// for each e-class.
    fn find_scores(&mut self) {
        let mut did_something = true;
        while did_something {
            // if this stays false for the body of the loop, then
            // we're done.
            did_something = false;

            for (i, class) in self.egraph.classes().enumerate() {
                eprintln!("id: {i} =====");
                eprintln!("{class:?}");
                if i == 7 {
                    let x = 4;
                    eprintln!(
                        "pattern: {}\n{:?}",
                        self.patterns[x].get_pattern_ast().unwrap().pretty(80),
                        self.patterns[x].get_pattern_ast().unwrap()
                    );

                    let rootless: RootlessPattern<_> =
                        self.patterns[x].get_pattern_ast().unwrap().into();
                    let n_matches = rootless.search_eclass(&self.egraph, class.id);

                    eprintln!("{:?} -> {n_matches}", class.nodes);
                }
            }
        }
    }
}
