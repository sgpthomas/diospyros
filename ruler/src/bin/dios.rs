use egg::{define_language, Id};
use itertools::Itertools;
use num::integer::Roots;
use rand::Rng;
use rand_pcg::Pcg64;
use ruler::{letter, map, self_product, CVec, Equality, SynthAnalysis, SynthLanguage, Synthesizer};
use rustc_hash::FxHasher;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::hash::BuildHasherDefault;
use std::str::FromStr;
use z3::ast::{Ast, Bool, Datatype, Int};
use z3::{DatatypeAccessor, DatatypeBuilder, Sort};

#[derive(Debug, PartialOrd, Ord, PartialEq, Eq, Hash, Clone)]
pub enum Value {
    // starts with i
    Int(i32),
    // starts with [
    List(Vec<Value>),
    // starts with <
    Vec(Vec<Value>),
    // starts with b
    Bool(bool),
}

impl FromStr for Value {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, <Self as FromStr>::Err> {
        if s.starts_with("i") {
            let v: i32 = s[1..].parse().map_err(|_| "Bad integer.".to_string())?;
            Ok(Value::Int(v))
        } else {
            Err(format!("{} didn't match anything.", s))
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Value::Int(i) => write!(f, "{}", i),
            Value::Bool(b) => write!(f, "{}", b),
            Value::List(l) => write!(f, "{:?}", l),
            Value::Vec(v) => write!(
                f,
                "<{}>",
                v.into_iter()
                    .map(|x| format!("{}", x))
                    .collect_vec()
                    .join(",")
            ),
        }
    }
}

fn split_into_halves(n: usize) -> (usize, usize) {
    if n % 2 == 0 {
        (n / 2, n / 2)
    } else {
        (n / 2, n / 2 + 1)
    }
}

impl Value {
    fn int1<F>(arg: &Self, f: F) -> Option<Value>
    where
        F: Fn(i32) -> Value,
    {
        if let Value::Int(val) = arg {
            Some(f(*val))
        } else {
            None
        }
    }

    fn int2<F>(lhs: &Self, rhs: &Self, f: F) -> Option<Value>
    where
        F: Fn(i32, i32) -> Value,
    {
        if let (Value::Int(lv), Value::Int(rv)) = (lhs, rhs) {
            Some(f(*lv, *rv))
        } else {
            None
        }
    }

    fn bool2<F>(lhs: &Self, rhs: &Self, f: F) -> Option<Value>
    where
        F: Fn(bool, bool) -> Value,
    {
        if let (Value::Bool(lv), Value::Bool(rv)) = (lhs, rhs) {
            Some(f(*lv, *rv))
        } else {
            None
        }
    }

    fn vec1<F>(val: &Self, f: F) -> Option<Value>
    where
        F: Fn(&[Value]) -> Value,
    {
        if let Value::Vec(v) = val {
            Some(f(v))
        } else {
            None
        }
    }

    fn vec2<F>(lhs: &Self, rhs: &Self, f: F) -> Option<Value>
    where
        F: Fn(&[Value], &[Value]) -> Option<Value>,
    {
        if let (Value::Vec(v1), Value::Vec(v2)) = (lhs, rhs) {
            if v1.len() == v2.len() {
                f(v1, v2)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn vec2_op<F>(lhs: &Self, rhs: &Self, f: F) -> Option<Value>
    where
        F: Fn(&Value, &Value) -> Option<Value>,
    {
        Self::vec2(lhs, rhs, |lhs, rhs| {
            lhs.iter()
                .zip(rhs)
                .map(|(l, r)| f(l, r))
                .collect::<Option<Vec<Value>>>()
                .map(|v| Value::Vec(v))
        })
    }

    #[allow(unused)]
    fn int_range(min: i32, max: i32, num_samples: usize) -> Vec<Value> {
        (min..=max)
            .step_by(((max - min) as usize) / num_samples)
            .map(|x| Value::Int(x))
            .collect::<Vec<_>>()
    }

    fn sample_int(rng: &mut Pcg64, min: i32, max: i32, num_samples: usize) -> Vec<Value> {
        (0..num_samples)
            .map(|_| Value::Int(rng.gen_range(min, max)))
            .collect::<Vec<_>>()
    }

    fn sample_vec(
        rng: &mut Pcg64,
        min: i32,
        max: i32,
        list_size: usize,
        num_samples: usize,
    ) -> Vec<Value> {
        (0..num_samples)
            .map(|_| Value::Vec(Value::sample_int(rng, min, max, list_size)))
            .collect::<Vec<_>>()
    }
}

fn sgn(x: i32) -> i32 {
    if x > 0 {
        1
    } else if x == 0 {
        0
    } else {
        -1
    }
}

define_language! {
    pub enum VecLang {
        // Id is a key to identify EClasses within an EGraph, represents
        // children nodes
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "-" = Minus([Id; 2]),
        "/" = Div([Id; 2]),

        "or" = Or([Id; 2]),
        "&&" = And([Id; 2]),
        "ite" = Ite([Id; 3]),
        "<" = Lt([Id; 2]),

        "sgn" = Sgn([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "neg" = Neg([Id; 1]),

        // Lists have a variable number of elements
        "List" = List(Box<[Id]>),

        // Vectors have width elements
        "Vec" = Vec(Box<[Id]>),

        // Vector with all literals
        // "LitVec" = LitVec(Box<[Id]>),

        "Get" = Get([Id; 2]),

        // Used for partitioning and recombining lists
        "Concat" = Concat([Id; 2]),

        // Vector operations that take 2 vectors of inputs
        "VecAdd" = VecAdd([Id; 2]),
        "VecMinus" = VecMinus([Id; 2]),
        "VecMul" = VecMul([Id; 2]),
        "VecDiv" = VecDiv([Id; 2]),
        // "VecMulSgn" = VecMulSgn([Id; 2]),

        // Vector operations that take 1 vector of inputs
        "VecNeg" = VecNeg([Id; 1]),
        "VecSqrt" = VecSqrt([Id; 1]),
        "VecSgn" = VecSgn([Id; 1]),

        // MAC takes 3 lists: acc, v1, v2
        // "VecMAC" = VecMAC([Id; 3]),

        Const(Value),

        // language items are parsed in order, and we want symbol to
        // be a fallback, so we put it last.
        // `Symbol` is an egg-provided interned string type
        Symbol(egg::Symbol),
    }
}

impl SynthLanguage for VecLang {
    type Constant = Value;

    fn to_var(&self) -> Option<egg::Symbol> {
        if let VecLang::Symbol(sym) = self {
            Some(*sym)
        } else {
            None
        }
    }

    fn mk_var(sym: egg::Symbol) -> Self {
        VecLang::Symbol(sym)
    }

    fn to_constant(&self) -> Option<&Self::Constant> {
        if let VecLang::Const(n) = self {
            Some(n)
        } else {
            None
        }
    }

    fn mk_constant(c: <Self as SynthLanguage>::Constant) -> Self {
        VecLang::Const(c)
    }

    fn eval<'a, F>(&'a self, cvec_len: usize, mut get: F) -> CVec<Self>
    where
        F: FnMut(&'a Id) -> &'a CVec<Self>,
    {
        match self {
            VecLang::Const(i) => vec![Some(i.clone()); cvec_len],
            VecLang::Add([l, r]) => map!(get, l, r => Value::int2(l, r, |l, r| Value::Int(l + r))),
            VecLang::Mul([l, r]) => map!(get, l, r => Value::int2(l, r, |l, r| Value::Int(l * r))),
            VecLang::Minus([l, r]) => {
                map!(get, l, r => Value::int2(l, r, |l, r| Value::Int(l - r)))
            }
            VecLang::Div([l, r]) => get(l)
                .iter()
                .zip(get(r).iter())
                .map(|tup| match tup {
                    (Some(Value::Int(a)), Some(Value::Int(b))) => {
                        if *b != 0 {
                            if *a == 0 {
                                Some(Value::Int(0))
                            } else if a >= b {
                                Some(Value::Int(a / b))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            VecLang::Or([l, r]) => {
                map!(get, l, r => Value::bool2(l, r, |l, r| Value::Bool(l || r)))
            }
            VecLang::And([l, r]) => {
                map!(get, l, r => Value::bool2(l, r, |l, r| Value::Bool(l && r)))
            }
            VecLang::Ite([_b, _t, _f]) => todo!(),
            VecLang::Lt([l, r]) => map!(get, l, r => Value::int2(l, r, |l, r| Value::Bool(l < r))),
            VecLang::Sgn([x]) => {
                map!(get, x => Value::int1(x, |x| Value::Int(sgn(x))))
            }
            VecLang::Sqrt([x]) => get(x)
                .iter()
                .map(|a| match a {
                    Some(Value::Int(a)) => {
                        if *a >= 0 {
                            Some(Value::Int(a.sqrt()))
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            VecLang::Neg([x]) => map!(get, x => Value::int1(x, |x| Value::Int(-x))),
            VecLang::List(l) => l
                .iter()
                .fold(vec![Some(vec![]); cvec_len], |mut acc, item| {
                    acc.iter_mut().zip(get(item)).for_each(|(mut v, i)| {
                        if let (Some(v), Some(i)) = (&mut v, i) {
                            v.push(i.clone());
                        } else {
                            *v = None;
                        }
                    });
                    acc
                })
                .into_iter()
                .map(|acc| {
                    if let Some(x) = acc {
                        Some(Value::List(x))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            VecLang::Vec(l) => l
                .iter()
                .fold(vec![Some(vec![]); cvec_len], |mut acc, item| {
                    acc.iter_mut().zip(get(item)).for_each(|(mut v, i)| {
                        if let (Some(v), Some(i)) = (&mut v, i) {
                            v.push(i.clone());
                        } else {
                            *v = None;
                        }
                    });
                    acc
                })
                .into_iter()
                .map(|acc| {
                    if let Some(x) = acc {
                        Some(Value::Vec(x))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            // VecLang::LitVec(x) => todo!(),
            VecLang::Get([l, i]) => map!(get, l, i => {
                if let (Value::Vec(v), Value::Int(idx)) = (l, i) {
            // get index and clone the inner Value if there is one
                    v.get(*idx as usize).map(|inner| inner.clone())
                } else {
            None
            }
                }),
            VecLang::Concat([l, r]) => {
                map!(get, l, r => Value::vec2(l, r, |l, r| {
                Some(Value::List(l.iter().chain(r).cloned().collect::<Vec<_>>()))
                    }))
            }
            VecLang::VecAdd([l, r]) => {
                map!(get, l, r => Value::vec2_op(l, r, |l, r| Value::int2(l, r, |l, r| Value::Int(l + r))))
            }
            VecLang::VecMinus([l, r]) => {
                map!(get, l, r => Value::vec2_op(l, r, |l, r| Value::int2(l, r, |l, r| Value::Int(l - r))))
            }
            VecLang::VecMul([l, r]) => {
                map!(get, l, r => Value::vec2_op(l, r, |l, r| Value::int2(l, r, |l, r| Value::Int(l * r))))
            }
            VecLang::VecDiv([l, r]) => {
                map!(get, l, r => Value::vec2_op(l, r, |l, r| {
                match (l, r) {
                    (Value::Int(a), Value::Int(b)) => {
                        if *b != 0 {
                            Some(Value::Int(a / b))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
                }))
            }
            VecLang::VecNeg([l]) => {
                map!(get, l => Value::vec1(l, |l| {
                    Value::Vec(l.iter().map(|tup| match tup {
                    Value::Int(a) => Value::Int(-a),
                        _ => panic!("Ill-formed")
                    }).collect::<Vec<_>>())
                }))
            }
            VecLang::VecSqrt([l]) => {
                map!(get, l => Value::vec1(l, |l| {
                    Value::Vec(l.iter().map(|tup| match tup {
                    Value::Int(a) => Value::Int(a.sqrt()),
                        _ => panic!("Ill-formed")
                    }).collect::<Vec<_>>())
                }))
            }
            VecLang::VecSgn([l]) => {
                map!(get, l => Value::vec1(l, |l| {
                    Value::Vec(l.iter().map(|tup| match tup {
                    Value::Int(a) => Value::Int(sgn(*a)),
                        _ => panic!("Ill-formed")
                    }).collect::<Vec<_>>())
                }))
            }
            // VecLang::VecMAC([a, b, c]) => todo!(),
            VecLang::Symbol(_) => vec![],
        }
    }

    fn init_synth(synth: &mut Synthesizer<Self>) {
        let consts: [Value; 2] = [
            // Value::List(vec![]),
            // Value::Vec(vec![Value::Int(0); synth.params.vector_size]),
            // Value::Vec(vec![Value::Int(1); synth.params.vector_size]),
            // Value::Vec(vec![]),
            // Value::Int(-1),
            Value::Int(0),
            Value::Int(1),
            // Value::Bool(true),
            // Value::Bool(false),
        ];

        // add_eq(
        //     synth,
        //     "1",
        //     "(+ (+ ?a ?b) (+ ?c ?d))",
        //     "(+ (+ ?c ?b) (+ ?a ?d))",
        // );
        // add_eq(synth, "2", "(+ ?a ?b)", "(+ ?b ?a)");
        // add_eq(synth, "3", "?a", "(+ ?a i0)");

        // initial set of rewrite rules
        // let assoc_add: ruler::Equality<VecLang> = ruler::Equality::new(
        //     &"(+ (+ ?a ?b) ?c)".parse().unwrap(),
        //     &"(+ ?a (+ ?b ?c))".parse().unwrap(),
        // )
        // .unwrap();
        // synth.equalities.insert("assoc_add".into(), assoc_add);

        // let iso_add: Equality<VecLang> = Equality::new(
        //     &"(VecAdd (Vec ?a ?b) (Vec ?c ?d))".parse().unwrap(),
        //     &"(Vec (+ ?a ?c) (+ ?b ?d))".parse().unwrap(),
        // )
        // .unwrap();
        // synth.equalities.insert("iso_add".into(), iso_add);

        // add_eq(
        //     synth,
        //     "iso_add_n3",
        //     "(VecAdd ?a ?b)",
        //     "(VecAdd (Vec (Get ?a i0) (Get ?a i1))
        //              (Vec (Get ?b i0) (Get ?b i1)))",
        // );

        // add_eq(
        //     synth,
        //     "iso_add_n3",
        //     "(VecAdd ?a (VecAdd ?b ?c))",
        //     "(VecAdd (Vec (Get ?a i0) (Get ?a i1) (Get ?a i2))
        //              (VecAdd (Vec (Get ?b i0) (Get ?b i1) (Get ?b i2))
        //                      (Vec (Get ?c i0) (Get ?c i1) (Get ?c i2))))",
        // );
        // add_eq(
        //     synth,
        //     "assoc_add",
        //     "(VecAdd (VecAdd ?a ?b) ?c)",
        //     "(VecAdd ?a (VecAdd ?b ?c))",
        // );
        // add_eq(
        //     synth,
        //     "iso_add",
        //     "(VecAdd (Vec ?a ?b) (Vec ?c ?d))",
        //     "(Vec (+ ?a ?c) (+ ?b ?d))",
        // );

        // add_eq(
        //     synth,
        //     "iso_mul",
        //     "(VecMul (Vec ?a ?b) (Vec ?c ?d))",
        //     "(Vec (* ?a ?c) (* ?b ?d))",
        // );

        // add_eq(synth, "get0", "(Get (Vec ?a ?b ?c) i0)", "?a");
        // add_eq(synth, "get1", "(Get (Vec ?a ?b ?c) i1)", "?b");
        // add_eq(synth, "get1", "(Get (Vec ?a ?b ?c) i2)", "?c");

        // let iso_add: ruler::Equality<VecLang> = ruler::Equality::new(
        //     &"(VecAdd ?a ?b)".parse().unwrap(),
        //     &"(VecAdd (Vec (Get ?a i0) (Get ?a i1)) (Vec (Get ?b i0) (Get ?b i1)))"
        //         .parse()
        //         .unwrap(),
        // )
        // .unwrap();
        // synth.equalities.insert("iso_add".into(), iso_add);

        // let comm_add: ruler::Equality<VecLang> =
        //     ruler::Equality::new(&"(+ ?a ?b)".parse().unwrap(), &"(+ ?b ?a)".parse().unwrap())
        //         .unwrap();
        // synth.equalities.insert("comm_add".into(), comm_add);
        // let assoc_add: ruler::Equality<VecLang> = ruler::Equality::new(
        //     &"(+ ?a (+ ?b ?c))".parse().unwrap(),
        //     &"(+ (+ ?a ?b) ?c)".parse().unwrap(),
        // )
        // .unwrap();
        // synth.equalities.insert("assoc_add".into(), assoc_add);

        // let get_vec: ruler::Equality<VecLang> = ruler::Equality::new(
        //     &"(VecAdd ?a (VecAdd ?b ?c))".parse().unwrap(),
        //     &"(Vec (+ (Get ?a i0) (+ (Get ?b i0) (Get ?c i0)))
        //            (+ (Get ?a i1) (+ (Get ?b i1) (Get ?c i1)))
        //            (+ (Get ?a i2) (+ (Get ?b i2) (Get ?c i2))))"
        //         .parse()
        //         .unwrap(),
        // )
        // .unwrap();
        // synth.equalities.insert("get_vec".into(), get_vec);

        // let assoc_vec_add: ruler::Equality<VecLang> = ruler::Equality::new(
        //     &"(VecAdd ?a (VecAdd ?b ?c))".parse().unwrap(),
        //     &"(VecAdd (VecAdd ?a ?b) ?c)".parse().unwrap(),
        // )
        // .unwrap();
        // synth
        //     .equalities
        //     .insert("assoc_vec_add".into(), assoc_vec_add);

        let consts_cross = self_product(
            &consts.iter().map(|x| Some(x.clone())).collect::<Vec<_>>(),
            synth.params.variables,
        );

        let size = consts_cross[0].len();

        // new egraph

        // let rewrites: Vec<egg::Rewrite<VecLang, SynthAnalysis>> = vec![
        //     rewrite!("add-comm"; "(+ ?a ?b)" <=> "(+ ?b ?a)"),
        //     rewrite!("add-vec-iso";
        // 	     "(VecAdd (Vec ?a ?b) (Vec ?c ?d))" <=> "(Vec (+ ?a ?c) (+ ?b ?d))"),
        // ]
        // .concat();

        // let mut runner = egg::Runner::default()
        //     .with_iter_limit(10)
        //     .with_node_limit(10_000)
        //     .with_scheduler(SimpleScheduler);

        // runner = runner.with_expr(&"(Vec (+ ?a ?c) (+ ?b ?d))".parse().unwrap());
        // runner = runner.with_expr(&"(Vec (+ ?a ?c) (+ ?d ?b))".parse().unwrap());
        // runner = runner.with_expr(&"(+ ?a ?b)".parse().unwrap());
        // runner = runner.with_expr(&"(+ ?b ?a)".parse().unwrap());

        // log::info!(
        //     "before {:?} ?= {:?}",
        //     runner.egraph.find(runner.roots[0]),
        //     runner.egraph.find(runner.roots[1])
        // );
        // runner = runner.run(&rewrites);
        // log::info!(
        //     "after {:?} ?= {:?}",
        //     runner.egraph.find(runner.roots[0]),
        //     runner.egraph.find(runner.roots[1])
        // );

        // runner.egraph.dot().to_png("play.png").unwrap();

        // log::info!("runner roots: {:?}", runner.roots);

        // let mut extract = egg::Extractor::new(&runner.egraph, egg::AstSize);
        // for ids in runner.roots.iter().combinations(2) {
        //     let l = *ids[0];
        //     let r = *ids[1];
        //     log::info!("{} <=?=> {}", l, r);
        //     if runner.egraph.find(l) != runner.egraph.find(r) {
        //         let (lcost, left) = extract.find_best(l);
        //         let (rcost, right) = extract.find_best(r);
        //         log::info!("lcost: {:?}, rcost: {:?}", lcost, rcost);
        //         log::info!("{} <=> {}", left, right);
        //     }
        // }

        // panic!();

        let mut egraph = egg::EGraph::new(SynthAnalysis { cvec_len: size });

        // add constants
        for v in consts.iter() {
            egraph.add(VecLang::Const(v.clone()));
        }

        // add variables
        for i in 0..synth.params.variables {
            let var = egg::Symbol::from(letter(i));
            let id = egraph.add(VecLang::Symbol(var));

            // make the cvec use real data
            let mut cvec = vec![];

            let (n_ints, n_vecs) = split_into_halves(size);

            cvec.extend(
                Value::sample_int(&mut synth.rng, -100, 100, n_ints)
                    .into_iter()
                    .map(Some),
            );

            cvec.extend(
                Value::sample_vec(&mut synth.rng, -100, 100, synth.params.vector_size, n_vecs)
                    .into_iter()
                    .map(Some),
            );

            egraph[id].data.cvec = cvec;
        }

        // set egraph to the one we just constructed
        synth.egraph = egraph;
    }

    /// Plan for `make_layer`
    /// even iter
    ///   normal binary ops
    /// odd iter
    ///   depth 1 and depth 2 vector operations
    fn make_layer<'a>(
        ids: Vec<Id>,
        synth: &'a Synthesizer<Self>,
        mut iter: usize,
    ) -> Box<dyn Iterator<Item = Self> + 'a> {
        // if iter % 2 == 0 {
        iter = iter - 1; // make iter start at 0
        let binops = if iter < 2 {
            Some(
                (0..2)
                    .map(|_| ids.clone())
                    .multi_cartesian_product()
                    .filter(move |ids| !ids.iter().all(|x| synth.egraph[*x].data.exact))
                    .map(|ids| [ids[0], ids[1]])
                    .map(|x| {
                        vec![
                            VecLang::Add(x),
                            // VecLang::Minus(x),
                            VecLang::Mul(x),
                            // VecLang::Div(x),
                            // VecLang::Or(x),
                            // VecLang::And(x),
                            // VecLang::Ite(x),
                            VecLang::Lt(x),
                        ]
                    })
                    .flatten(),
            )
        } else {
            None
        };

        let vec_stuff = if false {
            let vec_binops = (0..2)
                .map(|_| ids.clone())
                .multi_cartesian_product()
                .filter(move |ids| !ids.iter().all(|x| synth.egraph[*x].data.exact))
                .map(|ids| [ids[0], ids[1]])
                .map(|x| {
                    vec![
                        // don't fold
                        // VecLang::Concat(x),
                        VecLang::VecAdd(x),
                        VecLang::VecMul(x),
                        // VecLang::VecMinus(x),
                    ]
                })
                .flatten();

            let vec = (0..synth.params.vector_size)
                .map(|_| ids.clone())
                .multi_cartesian_product()
                .filter(move |ids| !ids.iter().all(|x| synth.egraph[*x].data.exact))
                .map(|x| vec![VecLang::Vec(x.into_boxed_slice())])
                .flatten();

            Some(vec_binops.chain(vec))
        } else {
            None
        };

        match (binops, vec_stuff) {
            (Some(b), Some(v)) => Box::new(b.chain(v)),
            (Some(i), _) => Box::new(i),
            (_, Some(i)) => Box::new(i),
            (_, _) => panic!(),
        }
    }

    fn is_valid(
        synth: &mut Synthesizer<Self>,
        lhs: &egg::Pattern<Self>,
        rhs: &egg::Pattern<Self>,
    ) -> bool {
        let mut cfg = z3::Config::new();
        cfg.set_timeout_msec(1000);
        let ctx = z3::Context::new(&cfg);
        let solver = z3::Solver::new(&ctx);

        let left = egg_to_z3(&ctx, Self::instantiate(lhs).as_ref());
        let right = egg_to_z3(&ctx, Self::instantiate(rhs).as_ref());

        let ret = if let (Some((lexpr, lasses)), Some((rexpr, rasses))) = (left, right) {
            // for ass in lasses.iter().chain(rasses.iter()) {
            //     solver.reset();
            //     solver.assert(ass);
            //     match solver.check() {
            //         z3::SatResult::Unsat => {
            //             log::info!("z3 ass was unsat {}", ass);
            //             return false;
            //         }
            //         z3::SatResult::Unknown => {
            //             return false;
            //         }
            //         z3::SatResult::Sat => {
            //             continue;
            //         }
            //     }
            // }
            // solver.reset();
            let all: Vec<_> = lasses.into_iter().chain(rasses.into_iter()).collect();

            solver.assert(&lexpr._eq(&rexpr).not());
            log::info!("z3 eq {} <=> {}", lhs, rhs);

            match solver.check_assumptions(&all) {
                z3::SatResult::Sat => {
                    log::info!("counter-example: {:?}", solver.get_model());
                    false
                }
                z3::SatResult::Unsat => {
                    log::info!("z3 validation: success for {} => {}", lhs, rhs);
                    log::info!("core: {:?}", solver.get_unsat_core());
                    true
                }
                z3::SatResult::Unknown => {
                    synth.smt_unknown += 1;
                    log::debug!("z3 validation: unknown for {} => {}", lhs, rhs);
                    false
                    // vecs_eq(&lvec, &rvec)
                }
            }
        } else {
            // use fuzzing to determine equality

            // let n = synth.params.num_fuzz;
            // let n = 10;
            let mut env = HashMap::default();

            if lhs.vars() != rhs.vars() {
                return false;
            }

            for var in lhs.vars() {
                env.insert(var, vec![]);
            }

            for var in rhs.vars() {
                env.insert(var, vec![]);
            }

            let (_n_ints, n_vecs) = split_into_halves(10);
            // let (n_neg_ints, n_pos_ints) = split_into_halves(n_ints);

            let possibilities = vec![-5, -2, -1, 0, 1, 2, 5];
            // let possibilities = vec![0, 1, 2];
            for perms in possibilities.iter().permutations(env.keys().len()) {
                for (i, cvec) in perms.iter().zip(env.values_mut()) {
                    cvec.push(Some(Value::Int(**i)));
                }
            }

            // add some random vectors
            let mut length = 0;
            for cvec in env.values_mut() {
                cvec.extend(
                    Value::sample_vec(&mut synth.rng, -100, 100, synth.params.vector_size, n_vecs)
                        .into_iter()
                        .map(Some),
                );
                length = cvec.len();
            }

            // debug(
            //     "(< (+ ?a ?b) (+ ?c ?a))",
            //     "(< ?b (+ i1 ?c))",
            //     length,
            //     &[
            //         ("?a", Value::Int(66)),
            //         ("?b", Value::Int(89)),
            //         ("?c", Value::Int(-6)),
            //         ("?d", Value::Int(83)),
            //     ],
            // );
            // panic!();

            let lvec = Self::eval_pattern(lhs, &env, length);
            let rvec = Self::eval_pattern(rhs, &env, length);

            if lvec != rvec {
                log::debug!("  env: {:?}", env);
                log::debug!("  lhs: {}, rhs: {}", lhs, rhs);
                log::debug!("  lvec: {:?}, rvec: {:?}", lvec, rvec);
            }

            vecs_eq(&lvec, &rvec)
        };
        // only compare values where both sides are defined
        ret
    }
}

fn vecs_eq(lvec: &CVec<VecLang>, rvec: &CVec<VecLang>) -> bool {
    if lvec.iter().all(|x| x.is_none()) && rvec.iter().all(|x| x.is_none()) {
        false
    } else {
        lvec.iter().zip(rvec.iter()).all(|tup| match tup {
            (Some(l), Some(r)) => l == r,
            _ => true,
        })
    }
}

#[allow(unused)]
fn add_eq(synth: &mut Synthesizer<VecLang>, name: &str, left: &str, right: &str) {
    let rule: Equality<VecLang> =
        Equality::new(&left.parse().unwrap(), &right.parse().unwrap()).unwrap();
    synth.equalities.insert(name.into(), rule);
}

#[allow(unused)]
fn debug(left: &str, right: &str, n: usize, env_pairs: &[(&str, Value)]) {
    let mut env: HashMap<egg::Var, Vec<Option<Value>>, BuildHasherDefault<FxHasher>> =
        HashMap::default();

    env_pairs.iter().for_each(|(var, value)| {
        env.insert(egg::Var::from_str(var).unwrap(), vec![Some(value.clone())]);
    });

    let pleft: egg::Pattern<VecLang> = left.parse().unwrap();
    let pright: egg::Pattern<VecLang> = right.parse().unwrap();
    log::info!("left");
    let lres = VecLang::eval_pattern(&pleft, &env, n);
    log::info!("right");
    let rres = VecLang::eval_pattern(&pright, &env, n);
    log::info!(
        "TEST:\n  {:?}\n    ?= ({})\n  {:?}",
        lres,
        vecs_eq(&lres, &rres),
        rres
    );
    log::info!("{} => {}", left, right);
}

fn main() {
    VecLang::main()
}

fn egg_to_z3<'a>(ctx: &'a z3::Context, expr: &[VecLang]) -> Option<(Datatype<'a>, Vec<Bool<'a>>)> {
    let mut buf: Vec<(Datatype, bool)> = vec![];
    let mut assumes: Vec<Bool> = vec![];

    // data type representing either ints or bools
    let int_bool = DatatypeBuilder::new(&ctx, "Int+Bool")
        .variant(
            "Int",
            vec![("Int.v", DatatypeAccessor::Sort(Sort::int(&ctx)))],
        )
        .variant(
            "Bool",
            vec![("Bool.v", DatatypeAccessor::Sort(Sort::bool(&ctx)))],
        )
        .finish();

    let int_cons = &int_bool.variants[0].constructor;
    let int_get = &int_bool.variants[0].accessors[0];
    let bool_cons = &int_bool.variants[1].constructor;
    let bool_get = &int_bool.variants[1].accessors[0];

    for node in expr.as_ref().iter() {
        match node {
            VecLang::Const(Value::Int(i)) => buf.push((
                int_cons
                    .apply(&[&Int::from_i64(&ctx, *i as i64)])
                    .as_datatype()
                    .unwrap(),
                false,
            )),
            VecLang::Symbol(v) => buf.push((
                Datatype::new_const(&ctx, v.to_string(), &int_bool.sort),
                false,
            )),
            VecLang::Add([x, y]) => {
                let x_int = int_get.apply(&[&buf[usize::from(*x)].0]).as_int().unwrap();
                let y_int = int_get.apply(&[&buf[usize::from(*y)].0]).as_int().unwrap();
                let res = int_cons.apply(&[&(x_int + y_int)]).as_datatype().unwrap();
                buf.push((res, false))
            }
            VecLang::Mul([x, y]) => {
                let x_int = int_get.apply(&[&buf[usize::from(*x)].0]).as_int().unwrap();
                let y_int = int_get.apply(&[&buf[usize::from(*y)].0]).as_int().unwrap();
                let res = int_cons.apply(&[&(x_int * y_int)]).as_datatype().unwrap();
                buf.push((res, false))
            }
            // VecLang::Div([x, y]) => {
            // 	let denom = int_get.apply(&[buf[usize::from(*x)].0]).as_int().unwrap();
            // }
            VecLang::Lt([a, b]) => {
                let a_int = int_get.apply(&[&buf[usize::from(*a)].0]).as_int().unwrap();
                let b_int = int_get.apply(&[&buf[usize::from(*b)].0]).as_int().unwrap();
                let res = bool_cons
                    .apply(&[&Int::lt(&a_int, &b_int)])
                    .as_datatype()
                    .unwrap();
                buf.push((res, true))
            }
            _ => return None,
        }
    }

    let (head, b) = buf.pop().unwrap();
    if b {
        let ass = bool_get.apply(&[&head]).as_bool().unwrap();
        assumes.push(ass);
    }
    Some((head, assumes))
}
