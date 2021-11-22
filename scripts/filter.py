#!/usr/bin/env python3

import sys
import json
import sexpdata as sexp
from dataclasses import dataclass
from typing import List, Any


@dataclass
class Node:
    """A type representing a node in an AST."""


@dataclass
class VecOp(Node):
    """Vector Operation."""

    op: str
    children: List[Node]


@dataclass
class Op(Node):
    """Non vector operation."""

    op: str
    children: List[Node]


@dataclass
class Vec(Node):
    """A Vector."""

    children: List[Node]


@dataclass
class Lit(Node):
    """A literal."""

    item: Any

def parse(raw):
    """Do f."""
    if type(raw) == list:
        head = raw[0]._val
        rest = raw[1:]
        if head in ["Get", "LitVec"]:
            return Lit(raw)
        elif head in ["+", "*", "/", "-", "or", "&&", "ite", "<", "sgn", "sqrt", "neg"]:
            return Op(head, [parse(c) for c in rest])
        elif head in ["VecAdd", "VecMinus", "VecMul", "VecDiv", "VecNeg", "VecSqrt", "VecSgn", "VecMAC"]:
            return VecOp(head, [parse(c) for c in rest])
        elif head in ["Vec"]:
            return Vec([parse(c) for c in rest])
        else:
            raise Exception(f"Unknown class: {head}, ({raw})")
    else:
        return Lit(raw)

    
COSTS = {
    "Struct": 0.1,
    "Lit": 0.001,
    "VecOp": 1,
    "Op": 1,
    "Big": 100
}


def cost(prog):
    """Find the cost of a program tree."""
    if type(prog) == VecOp:
        return COSTS["VecOp"] + sum([cost(p) for p in prog.children])
    elif type(prog) == Op:
        return COSTS["Op"] + sum([cost(p) for p in prog.children])
    elif type(prog) == Lit:
        return COSTS["Lit"]
    elif type(prog) == Vec:
        vec_of_lits = all([type(c) == Lit for c in prog.children])
        if vec_of_lits:
            return COSTS["Struct"] + sum([cost(p) for p in prog.children])
        else:
            return COSTS["Big"] + sum([cost(p) for p in prog.children])
    else:
        raise Exception("something went wrong")


def leaves(prog):
    if type(prog) == VecOp:
        return sum([leaves(c) for c in prog.children], [])
    elif type(prog) == Op:
        return sum([leaves(c) for c in prog.children], [])
    elif type(prog) == Lit:
        if type(prog.item) == sexp.Symbol:
            return [prog.item._val]
        else:
            return []
    elif type(prog) == Vec:
        return sum([leaves(c) for c in prog.children], [])
    else:
        raise Exception(f"unknown: {prog}")


def uses_vec_mac(prog):
    if type(prog) == VecOp:
        return prog.op == "VecMAC"
    else:
        return False


def tree_depth(prog):
    if type(prog) == VecOp or type(prog) == Op or type(prog) == Vec:
        return 1 + max([tree_depth(c) for c in prog.children])
    elif type(prog) == Lit:
        return 0
    else:
        raise Exception(f"unknown: {prog}")


def main():
    """Become the main function."""
    filename = sys.argv[1]
    with open(filename, "r") as f:
        j = json.load(f)
        # limit = 0
        n_kept = 0
        for eq in j["eqs"]:
            # print(f"{eq['lhs']} <=> {eq['rhs']}")
            lhs = parse(sexp.loads(eq["lhs"]))
            rhs = parse(sexp.loads(eq["rhs"]))
            # if getheads(lhs) != getheads(rhs):
            cost_diff = abs(cost(lhs) - cost(rhs)) > 2
            all_vars = len(set(leaves(lhs))) == 4 and len(set(leaves(rhs))) == 4
            vec_mac = (uses_vec_mac(lhs) and tree_depth(lhs) == 1) or (tree_depth(rhs) == 1 and uses_vec_mac(rhs))
            if vec_mac:
                n_kept += 1
                print(f"{eq['lhs']} <=> {eq['rhs']}")
                print("cost: {}".format(abs(cost(lhs) - cost(rhs))))

            # if limit > 3:
            #     break
            # limit += 1
        print(f"# kept/total: {n_kept} / {len(j['eqs'])}")


if __name__ == "__main__":
    main()
