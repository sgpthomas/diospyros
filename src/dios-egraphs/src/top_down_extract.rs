use std::collections::HashMap;

#[derive(Debug)]
pub struct TopDownExtractor<L: egg::Language> {
    score_map: HashMap<egg::Id, (f64, L)>,
}

impl<L: egg::Language> TopDownExtractor<L> {
    /// Create a new `TopDownExtractor`.
    pub fn new() -> Self {
        let extractor = Self {
            score_map: HashMap::default(),
        };
        extractor
    }
}
