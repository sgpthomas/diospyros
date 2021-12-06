import sexpdata as sexp
from dataclasses import dataclass
from typing import List, Any

@dataclass(eq=True, frozen=True)
class Node:
    """A type representing a node in an AST."""


@dataclass(eq=True, frozen=True)
class VecOp(Node):
    """Vector Operation."""

    op: str
    children: List[Node]


@dataclass(eq=True, frozen=True)
class Op(Node):
    """Non vector operation."""

    op: str
    children: List[Node]


@dataclass(eq=True, frozen=True)
class Vec(Node):
    """A Vector."""

    children: List[Node]


@dataclass(eq=True, frozen=True)
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
        elif head in ["+", "*", "/", "-", "or", "&&", "ite", "<", "sgn", "sqrt", "neg", "Concat"]:
            return Op(head, [parse(c) for c in rest])
        elif head in ["VecAdd", "VecMinus", "VecMul", "VecDiv", "VecNeg", "VecSqrt", "VecSgn", "VecMAC"]:
            return VecOp(head, [parse(c) for c in rest])
        elif head in ["Vec"]:
            return Vec([parse(c) for c in rest])
        else:
            raise Exception(f"Unknown class: {head}, ({raw})")
    else:
        return Lit(raw)


def pprint(prog):
    if type(prog) == VecOp or type(prog) == Op:
        return "({} {})".format(prog.op, " ".join([pprint(c) for c in prog.children]))
    elif type(prog) == Lit:
        if type(prog.item) == sexp.Symbol:
            return "{}".format(prog.item._val)
        else:
            return "{}".format(prog.item)
    elif type(prog) == Vec:
        return "(Vec {})".format(" ".join([pprint(c) for c in prog.children]))
    else:
        raise Exception(f"Unknown type: {type(prog)} ({prog})")

    
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


def depth(prog):
    if type(prog) == VecOp or type(prog) == Op or type(prog) == Vec:
        return max([depth(c) + 1 for c in prog.children])
    elif type(prog) == Lit:
        return 0
    else:
        raise Exception(f"tooo deep to peer into...: {prog}")
