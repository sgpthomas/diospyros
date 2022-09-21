use std::{collections::HashMap, fmt::Display};

use itertools::Itertools;

#[derive(Debug)]
pub struct TopDownExtractor<'a, L: egg::Language, N: egg::Analysis<L>> {
    score_map: HashMap<egg::Id, (f64, L)>,
    egraph: &'a egg::EGraph<L, N>,
    rules: &'a [egg::RecExpr<egg::ENodeOrVar<L>>],
}

impl<'a, L: egg::Language + Display, N: egg::Analysis<L>> TopDownExtractor<'a, L, N> {
    /// Create a new `TopDownExtractor`.
    pub fn new(
        egraph: &'a egg::EGraph<L, N>,
        rules: &'a [egg::RecExpr<egg::ENodeOrVar<L>>],
    ) -> Self {
        let mut extractor = Self {
            score_map: HashMap::default(),
            egraph,
            rules,
        };

        eprintln!("{:?}", rules[4]);

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
                if i == 7 {
                    eprintln!("{:?}", class.nodes);
                }
                for node in &class.nodes {
                    let children = node.children();
                    if children.len() == 0 {
                        // pass
                    } else if children.len() == 1 {
                        // pass
                    } else {
                        let first = &children[0];
                        let sibling_ids = &children[1..];
                        let sibling_nodes: Vec<_> = sibling_ids
                            .iter()
                            .map(|id| &self.egraph[*id].nodes)
                            .collect();
                        for node in &self.egraph[*first].nodes {
                            if i == 7 {
                                eprintln!("* {node:?} {sibling_nodes:?}");
                            }
                        }
                    }
                    // let children = node.children().iter().map(|id| &self.egraph[*id]);
                    // for child in children {
                    //     for node in &child.nodes {
                    //         if i == 7 {
                    //             eprintln!("* {node:?}");
                    //         }
                    //     }
                    // }
                }
            }
        }
    }
}
