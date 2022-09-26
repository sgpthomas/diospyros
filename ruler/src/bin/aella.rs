use egg::{define_language, Id};
use itertools::Itertools;
use rand::Rng;
use ruler::{letter, map, self_product, SynthLanguage};
use std::collections::BTreeMap;

define_language! {
    pub enum Aella {
    "+" = Plus([Id; 2]),
    "-" = Sub([Id; 2]),
    "*" = Times([Id; 2]),
    "/" = Div([Id; 2]),
    "==" = Eq([Id; 2]),
    "<=" = Lte([Id; 2]),
    "!" = Not([Id; 1]),
    "&&" = And([Id; 2]),
    "seq" = Seq([Id; 2]),
    ":=" = Assign([Id; 2]),
    "while" = While([Id; 2]),
    "mov" = AsmMov([Id; 2]),
    "add" = AsmAdd([Id; 3]),
    "sub" = AsmSub([Id; 3]),
    "smull" = AsmSmull([Id; 3]),
    "sdiv" = AsmSdiv([Id; 3]),
    Num(i64),
    FreeVar(egg::Symbol),
    BoundVar(egg::Symbol),
    }
}

type Env = BTreeMap<egg::Symbol, i64>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Int(i64),
    VarName(egg::Symbol),
    Env(Env),
}

impl Value {
    fn val2<F>(lhs: &Self, rhs: &Self, f: F) -> Option<Value>
    where
        F: Fn(i64, i64) -> Value,
    {
        if let (Value::Int(lv), Value::Int(rv)) = (lhs, rhs) {
            Some(f(*lv, *rv))
        } else {
            None
        }
    }
}

fn union(mut e1: Env, e2: &Env) -> Env {
    e1.extend(e2.into_iter());
    e1
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl SynthLanguage for Aella {
    type Constant = Value;
    type Config = ();

    fn eval<'a, F>(&'a self, cvec_len: usize, mut get: F) -> ruler::CVec<Self>
    where
        F: FnMut(&'a Id) -> &'a ruler::CVec<Self>,
    {
        match self {
            Aella::Plus([l, r]) => {
                map!(get, l, r => Value::val2(l, r, |l, r| Value::Int(l + r)))
            }
            Aella::Sub([l, r]) => {
                map!(get, l, r => Value::val2(l, r, |l, r| Value::Int(l - r)))
            }
            Aella::Times([_l, _r]) => todo!(),
            Aella::Div([_l, _r]) => todo!(),
            Aella::Eq([_l, _r]) => todo!(),
            Aella::Lte([_l, _r]) => todo!(),
            Aella::Not([_inner]) => todo!(),
            Aella::And([_l, _r]) => todo!(),
            Aella::Seq([l, r]) => get(l)
                .iter()
                .zip(get(r).iter())
                .map(|tup| match tup {
                    (Some(Value::Env(e1)), Some(Value::Env(e2))) => {
                        Some(Value::Env(union(e1.clone(), e2)))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            Aella::Assign([l, r]) => get(l)
                .iter()
                .zip(get(r).iter())
                .map(|tup| match tup {
                    (Some(Value::VarName(name)), Some(Value::Int(val))) => {
                        Some(Value::Env(BTreeMap::from([(*name, *val)])))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            Aella::While(_) => todo!(),
            Aella::AsmMov([rd, src]) => get(rd)
                .iter()
                .zip(get(src).iter())
                .map(|tup| match tup {
                    (Some(Value::VarName(name)), Some(Value::Int(val))) => {
                        Some(Value::Env(BTreeMap::from([(*name, *val)])))
                    }
                    _ => None,
                })
                .collect_vec(),
            Aella::AsmAdd([rd, rn, rm]) => get(rd)
                .iter()
                .zip(get(rn).iter())
                .zip(get(rm).iter())
                .map(|tup| match tup {
                    (
                        (Some(Value::VarName(name)), Some(Value::Int(l))),
                        Some(Value::Int(r)),
                    ) => Some(Value::Env(BTreeMap::from([(*name, l + r)]))),
                    _ => None,
                })
                .collect_vec(),
            Aella::AsmSub([rd, rn, rm]) => get(rd)
                .iter()
                .zip(get(rn).iter())
                .zip(get(rm).iter())
                .map(|tup| match tup {
                    (
                        (Some(Value::VarName(name)), Some(Value::Int(l))),
                        Some(Value::Int(r)),
                    ) => Some(Value::Env(BTreeMap::from([(*name, l - r)]))),
                    _ => None,
                })
                .collect_vec(),
            Aella::AsmSmull(_) => todo!(),
            Aella::AsmSdiv(_) => todo!(),
            Aella::Num(i) => vec![Some(Value::Int(*i)); cvec_len],
            Aella::FreeVar(_) => {
                // I'm not sure why this is just empty tbh.
                // my guess is that it's because we never
                // actually evaluate this. it's cvec comes
                // from initialization. still strange.
                // unreachable!("I guess this is evaluated?")
                vec![]
            }
            Aella::BoundVar(s) => vec![Some(Value::VarName(*s)); cvec_len],
        }
    }

    fn to_var(&self) -> Option<egg::Symbol> {
        if let Aella::FreeVar(sym) = self {
            Some(*sym)
        } else {
            None
        }
    }

    fn mk_var(sym: egg::Symbol) -> Self {
        Aella::FreeVar(sym)
    }

    fn to_constant(&self) -> Option<&Self::Constant> {
        // The only place that this function is used is for
        // `is_constant`. I'm just overriding that function
        // directly because I can't return a reference.
        unreachable!("I thought this wasn't used.")
    }

    fn mk_constant(c: Self::Constant) -> Option<Self> {
        match c {
            Value::Int(val) => Some(Aella::Num(val)),
            Value::VarName(name) => Some(Aella::BoundVar(name)),
            Value::Env(_) => None,
        }
    }

    fn is_constant(&self) -> bool {
        matches!(self, Aella::Num(..))
    }

    fn init_synth(synth: &mut ruler::Synthesizer<Self, ruler::Uninit>) {
        // initial constants that will be in the graph
        let constants = [-1, 0, 1];

        let cvec_len = self_product(
            &constants
                .iter()
                .map(|x| Some(Value::Int(*x)))
                .collect::<Vec<_>>(),
            synth.params.variables,
        )
        .len();

        // make a new egraph
        let mut egraph = egg::EGraph::new(ruler::SynthAnalysis { cvec_len });

        // add constants to the egraph
        for v in constants {
            egraph.add(Aella::Num(v));
        }

        // add some initial variables to the egraph as well
        // for now, I'll add the same number of freevars as boundvars
        // to the graph. This is a basically random decision.
        for i in 0..synth.params.variables {
            let free_var = egg::Symbol::from(letter(i));
            let free_id = egraph.add(Aella::FreeVar(free_var));

            eprintln!("{i} {}", synth.params.variables);

            let bound_var =
                egg::Symbol::from(letter(synth.params.variables + i));
            let bound_id = egraph.add(Aella::BoundVar(bound_var));

            // make the cvec use real data
            egraph[free_id].data.cvec = (0..cvec_len)
                .map(|_| Value::Int(synth.rng.gen_range(-100, 100)))
                .map(Some)
                .collect::<Vec<_>>();

            // the cvec of a bound variable is just it's name
            egraph[bound_id].data.cvec =
                vec![Some(Value::VarName(bound_var)); cvec_len];
        }

        eprintln!("egraph: {egraph:#?}");

        synth.egraph = egraph;
    }

    fn make_layer<'a>(
        ids: Vec<Id>,
        synth: &'a ruler::Synthesizer<Self, ruler::Init>,
        _iter: usize,
    ) -> Box<dyn Iterator<Item = Self> + 'a> {
        let binops = (0..2)
            .map(|_| ids.clone())
            .multi_cartesian_product()
            .filter(move |ids| !ids.iter().all(|x| synth.egraph[*x].data.exact))
            .map(|ids| [ids[0], ids[1]])
            .map(move |x| {
                vec![
                    Aella::Plus(x),
                    Aella::Sub(x),
                    Aella::Assign(x),
                    Aella::Seq(x),
                    Aella::AsmMov(x),
                ]
            })
            .flatten();

        let terops = (0..3)
            .map(|_| ids.clone())
            .multi_cartesian_product()
            .filter(move |ids| !ids.iter().all(|x| synth.egraph[*x].data.exact))
            .map(|ids| [ids[0], ids[1], ids[2]])
            .map(move |x| vec![Aella::AsmAdd(x), Aella::AsmSub(x)])
            .flatten();

        Box::new(binops.chain(terops))
    }

    fn is_valid(
        _synth: &mut ruler::Synthesizer<Self, ruler::Init>,
        _lhs: &egg::Pattern<Self>,
        _rhs: &egg::Pattern<Self>,
    ) -> bool {
        // why is this separate from the cvec thing???
        // maybe this allows equality to be stronger than
        // cvec fuzzing??

        // construct an environment for the variables
        // let mut env = HashMap::default();

        // for var in lhs.vars().into_iter().chain(rhs.vars().into_iter()) {
        //     env.insert(var, vec![]);
        // }

        // let lvec = Self::eval_pattern(lhs, &env, 10);
        // let rvec = Self::eval_pattern(lhs, &env, 10);

        // eprintln!("lhs: {lhs:?} => {lvec:?}");
        // eprintln!("rhs: {rhs:?} => {rvec:?}");
        // panic!("env: {:?}", env);

        true
    }
}

fn main() {
    <Aella as ruler::Main>::main()
}
