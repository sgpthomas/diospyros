use std::collections::{HashMap, HashSet};

use crate::veclang::VecLang;
use egg::{Analysis, CostFunction, DidMerge, EClass, Id, Language, RecExpr};

use std::cmp::Ordering;

#[derive(Default, Clone)]
pub struct TrackRewrites;
impl Analysis<VecLang> for TrackRewrites {
    // type Data = HashSet<String>;
    // type Data = Vec<String>;
    type Data = ();

    fn make(_egraph: &egg::EGraph<VecLang, Self>, _enode: &VecLang) -> Self::Data {
        // HashSet::new()
        // Vec::new()
        ()
    }

    fn merge(&self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        // *to = &*to | &from;
        // to.extend(from);
        DidMerge(false, false)
    }

    fn modify(_egraph: &mut egg::EGraph<VecLang, Self>, _id: egg::Id) {}
}

impl TrackRewrites {
    pub fn trace(root: &VecLang, expr: &[VecLang]) {
        println!("{:?}", root);
        root.for_each(|child| {
            let idx: usize = child.into();
            Self::trace(&expr[idx], expr);
        })
    }
}

fn cmp<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    // None is high
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => a.partial_cmp(&b).unwrap(),
    }
}

pub struct CustomExtractor<'a, CF: CostFunction<L>, L: Language, N: Analysis<L>> {
    cost_function: CF,
    costs: HashMap<Id, (CF::Cost, L)>,
    egraph: &'a egg::EGraph<L, N>,
}
impl<'a, CF, L, N> CustomExtractor<'a, CF, L, N>
where
    CF: CostFunction<L>,
    L: Language,
    N: Analysis<L>,
{
    /// Create a new `Extractor` given an `EGraph` and a
    /// `CostFunction`.
    ///
    /// The extraction does all the work on creation, so this function
    /// performs the greedy search for cheapest representative of each
    /// eclass.
    pub fn new(egraph: &'a egg::EGraph<L, N>, cost_function: CF) -> Self {
        let costs = HashMap::default();
        let mut extractor = Self {
            costs,
            egraph,
            cost_function,
        };
        extractor.find_costs();

        extractor
    }

    /// Find the cheapest (lowest cost) represented `RecExpr` in the
    /// given eclass.
    pub fn find_best(&mut self, eclass: Id) -> (CF::Cost, RecExpr<L>) {
        let mut expr = RecExpr::default();
        let (_, cost) = self.find_best_rec(&mut expr, eclass);
        (cost, expr)
    }

    fn find_best_rec(&mut self, expr: &mut RecExpr<L>, eclass: Id) -> (Id, CF::Cost) {
        let id = self.egraph.find(eclass);
        let (best_cost, best_node) = match self.costs.get(&id) {
            Some(result) => result.clone(),
            None => panic!("Failed to extract from eclass {}", id),
        };

        let node = best_node.map_children(|child| self.find_best_rec(expr, child).0);
        println!("expr: {:?} {:?}", node, self.egraph[eclass].data);

        (expr.add(node), best_cost)
    }

    fn node_total_cost(&mut self, node: &L) -> Option<CF::Cost> {
        let eg = &self.egraph;
        let has_cost = |&id| self.costs.contains_key(&eg.find(id));
        if node.children().iter().all(has_cost) {
            let costs = &self.costs;
            let cost_f = |id| costs[&eg.find(id)].0.clone();
            Some(self.cost_function.cost(&node, cost_f))
        } else {
            None
        }
    }

    fn find_costs(&mut self) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                let pass = self.make_pass(class);
                match (self.costs.get(&class.id), pass) {
                    (None, Some(new)) => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    (Some(old), Some(new)) if new.0 < old.0 => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    _ => (),
                }
            }
        }

        for class in self.egraph.classes() {
            if !self.costs.contains_key(&class.id) {
                log::warn!(
                    "Failed to compute cost for eclass {}: {:?}",
                    class.id,
                    class.nodes
                )
            }
        }
    }

    fn make_pass(&mut self, eclass: &EClass<L, N::Data>) -> Option<(CF::Cost, L)> {
        let (cost, node) = eclass
            .iter()
            .map(|n| (self.node_total_cost(n), n))
            .min_by(|a, b| cmp(&a.0, &b.0))
            .unwrap_or_else(|| panic!("Can't extract, eclass is empty: {:#?}", eclass));
        cost.map(|c| (c, node.clone()))
    }
}
