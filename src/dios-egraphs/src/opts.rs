use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

use argh::FromArgs;
/// The Diospyros Equality Saturation Compiler
#[derive(FromArgs)]
pub struct Opts {
    /// the input file.
    #[argh(positional, from_str_fn(read_path))]
    pub input: PathBuf,

    /// disable associativity and commutativity rules. Only valid with --handwritten.
    #[argh(switch)]
    pub no_ac: bool,

    /// disable vector rules. Only valid with --handwritten.
    #[argh(switch)]
    pub no_vec: bool,

    /// enable handwritten rules.
    #[argh(switch)]
    pub handwritten: bool,

    /// path to external rules json.
    #[argh(option, from_str_fn(read_path))]
    pub rules: Option<PathBuf>,

    /// only use rules that have a cost differential.
    #[argh(option)]
    pub cost_filter: Option<f64>,

    /// filters out rules that have duplicate variables.
    #[argh(switch)]
    pub no_dup_vars: bool,

    /// perform pre-compilation, compilation, and optimization
    /// phases separately.
    #[argh(option)]
    pub split_phase: Option<SplitPhase>,

    /// start from a new egraph for each phase.
    #[argh(switch)]
    pub new_egraph: bool,

    /// run eqsat on extracted sub programs instead of on the whole program.
    #[argh(switch)]
    pub sub_prog: bool,

    /// iteration limit for equality saturation.
    #[argh(option, default = "20")]
    pub iter_limit: usize,

    /// timeout for equality saturation.
    #[argh(option, default = "180")]
    pub timeout: usize,

    /// scheduler to use for equality saturation
    #[argh(option, default = "SchedulerOpt::default()")]
    pub scheduler: SchedulerOpt,

    /// don't run eq saturation at all.
    #[argh(switch)]
    pub dry_run: bool,

    /// dump rules.
    #[argh(switch)]
    pub dump_rules: bool,

    /// instrument the eq sat process
    #[argh(option, from_str_fn(read_path))]
    pub instrument: Option<PathBuf>,

    /// vector width
    #[argh(option, default = "2")]
    pub vector_width: usize,

    /// pre-desugared
    #[argh(switch)]
    pub pre_desugared: bool,
}

fn read_path(path: &str) -> Result<PathBuf, String> {
    Ok(Path::new(path).into())
}

pub enum SchedulerOpt {
    /// use egg::SimpleScheduler
    Simple,

    /// use egg::BackoffScheduler
    Backoff,

    /// use the custom LoggingScheduler
    Logging,
}

impl Default for SchedulerOpt {
    fn default() -> Self {
        SchedulerOpt::Simple
    }
}

impl FromStr for SchedulerOpt {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "simple" => Ok(SchedulerOpt::Simple),
            "backoff" => Ok(SchedulerOpt::Backoff),
            "logging" => Ok(SchedulerOpt::Logging),
            s => Err(format!(
                "Unknown compilation mode: {}. Valid options are {}",
                s, "[`simple`, `backoff`, `logging`]"
            )),
        }
    }
}

pub enum SplitPhase {
    /// Automatically split phases by performing a rule analysis.
    Auto,

    /// Split rules according to a hand specified split.
    Handwritten,

    /// Split rules by looking at the syntax of the rules.
    Syntax,
}

impl FromStr for SplitPhase {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(SplitPhase::Auto),
            "hand" | "handwritten" => Ok(SplitPhase::Handwritten),
            "syntax" => Ok(SplitPhase::Syntax),
            s => Err(format!(
                "Unknown split phase method: {}. Valid options are {}",
                s, "[`auto`, `handwritten`, `syntax`]"
            )),
        }
    }
}
