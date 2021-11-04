use egg::{Rewrite, RewriteScheduler};

/// Stores a map from rule name (string) and iteration number (usize)
/// to the number of times that rule was applied in that iteration.
#[derive(Default)]
pub struct LoggingScheduler;

impl<L: egg::Language, N: egg::Analysis<L>> RewriteScheduler<L, N> for LoggingScheduler {
    fn apply_rewrite(
        &mut self,
        _iteration: usize,
        egraph: &mut egg::EGraph<L, N>,
        rewrite: &Rewrite<L, N>,
        matches: Vec<egg::SearchMatches>,
    ) -> usize {
        // apply the rule
        let applications = rewrite.apply(egraph, &matches);

        // logging stuff

        // update the number of times this rule was used for this iteration.
        // self.n_used
        //     .entry((rewrite.name().to_string(), iteration))
        //     .and_modify(|v| *v += 1)
        //     .or_insert(0);

        // eprintln!("Rewrite {}", rewrite.name());

        // return the number of applications
        applications.len()
    }
}

#[derive(Default)]
pub struct LoggingData;

impl<L: egg::Language, N: egg::Analysis<L>> egg::IterationData<L, N> for LoggingData {
    fn make(_runner: &egg::Runner<L, N, Self>) -> Self {
        eprintln!("iter data was called");
        LoggingData
    }
}
