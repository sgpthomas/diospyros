#!/usr/bin/env python3

from veclang import *


EXPR = """
(Concat
  (Vec
    (+
      (+
        (* (Get aq 3) (Get bq 0))
        (+ (* (Get aq 0) (Get bq 3)) (* (Get aq 1) (Get bq 2))))
      (neg (* (Get aq 2) (Get bq 1))))
    (+
      (+
        (* (Get aq 3) (Get bq 1))
        (+ (* (Get aq 1) (Get bq 3)) (* (Get aq 2) (Get bq 0))))
      (neg (* (Get aq 0) (Get bq 2)))))
  (Concat
    (Vec
      (+
        (+
          (* (Get aq 3) (Get bq 2))
          (+ (* (Get aq 2) (Get bq 3)) (* (Get aq 0) (Get bq 1))))
        (neg (* (Get aq 1) (Get bq 0))))
      (+
        (* (Get aq 3) (Get bq 3))
        (+
          (neg (* (Get aq 0) (Get bq 0)))
          (+ (neg (* (Get aq 1) (Get bq 1))) (neg (* (Get aq 2) (Get bq 2)))))))
    (Concat
      (Vec
        (+
          (Get at 0)
          (+
            (Get bt 0)
            (+
              (+
                (*
                  (Get aq 1)
                  (* 2 (+ (* (Get aq 0) (Get bt 1)) (neg (* (Get aq 1) (Get bt 0))))))
                (neg
                  (*
                    (Get aq 2)
                    (* 2 (+ (* (Get aq 2) (Get bt 0)) (neg (* (Get aq 0) (Get bt 2))))))))
              (*
                (Get aq 3)
                (* 2 (+ (* (Get aq 1) (Get bt 2)) (neg (* (Get aq 2) (Get bt 1)))))))))
        (+
          (Get at 1)
          (+
            (Get bt 1)
            (+
              (+
                (*
                  (Get aq 2)
                  (* 2 (+ (* (Get aq 1) (Get bt 2)) (neg (* (Get aq 2) (Get bt 1))))))
                (neg
                  (*
                    (Get aq 0)
                    (* 2 (+ (* (Get aq 0) (Get bt 1)) (neg (* (Get aq 1) (Get bt 0))))))))
              (*
                (Get aq 3)
                (* 2 (+ (* (Get aq 2) (Get bt 0)) (neg (* (Get aq 0) (Get bt 2))))))))))
      (Vec
        (+
          (Get at 2)
          (+
            (Get bt 2)
            (+
              (+
                (*
                  (Get aq 0)
                  (* 2 (+ (* (Get aq 2) (Get bt 0)) (neg (* (Get aq 0) (Get bt 2))))))
                (neg
                  (*
                    (Get aq 1)
                    (* 2 (+ (* (Get aq 1) (Get bt 2)) (neg (* (Get aq 2) (Get bt 1))))))))
              (*
                (Get aq 3)
                (* 2 (+ (* (Get aq 0) (Get bt 1)) (neg (* (Get aq 1) (Get bt 0)))))))))
        0))))
"""


ALPH = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def get_alph(i):
    if i < len(ALPH):
        return f"?{ALPH[i]}"
    else:
        c = ALPH[i % len(ALPH)]
        n = i // len(ALPH)
        return f"?{c}{n}"


def children_to_vars(children):
    ret = []
    for i, _c in enumerate(children):
        ret.append(Lit(f"?{ALPH[i]}"))
    return ret


def bot_up_pat(prog):
    patterns = set()
    if type(prog) == Op:
        p = Op(prog.op, children=children_to_vars(prog.children))
        patterns.add(pprint(p))
        for c in prog.children:
            patterns.update(bot_up_pat(c))
    elif type(prog) == VecOp:
        p = VecOp(op=prog.op, children=children_to_vars(prog.children))
        patterns.add(pprint(p))
        for c in prog.children:
            patterns.update(bot_up_pat(c))
    elif type(prog) == Vec:
        p = Vec(children=children_to_vars(prog.children))
        patterns.add(pprint(p))
        for c in prog.children:
            patterns.update(bot_up_pat(c))
    return patterns


def progs_of_depth(prog, d):
    if type(prog) == VecOp or type(prog) == Op or type(prog) == Vec:
        if any([depth(c) == d for c in prog.children]):
            return [prog]
        else:
            return sum([progs_of_depth(c, d) for c in prog.children], [])
    elif type(prog) == Lit:
        return []
    else:
        raise Exception("Tree was ill-formed")


def replace_leaves(prog, assgns=None):
    if assgns is None:
        assgns = {}

    if type(prog) == Lit or \
       all([type(c) == Lit and type(c.item) == sexp.Symbol for c in prog.children]):
        key = pprint(prog)
        if not key in assgns:
            assgns[key] = Lit(sexp.Symbol(get_alph(len(assgns))))
        return assgns[key]
    elif type(prog) == VecOp:
        return VecOp(op=prog.op, children=[replace_leaves(p, assgns) for p in prog.children])
    elif type(prog) == Op:
        return Op(op=prog.op, children=[replace_leaves(p, assgns) for p in prog.children])
    elif type(prog) == Vec:
        return Vec(children=[replace_leaves(p, assgns) for p in prog.children])
    else:
        raise Exception("Unknown type.")

def rename(prog, assgns=None):
    if assgns is None:
        assgns = {}

    if type(prog) == Lit:
        if type(prog.item) == sexp.Symbol:
            val = prog.item._val
            if val not in assgns:
                assgns[val] = get_alph(len(assgns))
            return Lit(sexp.Symbol(assgns[val]))
        else:
            return prog
    elif type(prog) == VecOp:
        return VecOp(op=prog.op, children=[rename(c, assgns) for c in prog.children])
    elif type(prog) == Op:
        return Op(op=prog.op, children=[rename(c, assgns) for c in prog.children])
    elif type(prog) == Vec:
        return Vec(children=[rename(c, assgns) for c in prog.children])
    else:
        raise Exception(f"Unknown prog: {prog}")


def main():
    parsed = parse(sexp.loads(EXPR))
    # for p in top_down_pat(parsed):
    #     print(p)
    pats = []
    inp = parsed

    all_pats = set()

    a = replace_leaves(inp)
    for d in range(depth(a)):
        print(f"==== {d} ====")

        for p in set([pprint(rename(p)) for p in progs_of_depth(a, d)]):
            print(p)

    # while depth(inp) > 0:
    #     nxt = top_down_pat(inp)
    #     all_pats.update(bot_up_pat(inp))
    #     pats.append(nxt)
    #     inp = nxt

    for p in pats:
        print(depth(p), pprint(p))

    print("=====")

    for p in all_pats:
        print(p)


if __name__ == "__main__":
    main()
