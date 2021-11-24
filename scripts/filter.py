#!/usr/bin/env python3

import sys
import json
import sexpdata as sexp
from dataclasses import dataclass
from typing import List, Any
import networkx as nx
import itertools


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
        raise Exception(f"Unknown type: {prog}")

    
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


def operators(prog):
    """Get the operators of the prog tree."""
    if type(prog) == VecOp or type(prog) == Op:
        return sum([operators(c) for c in prog.children], [prog.op])
    elif type(prog) == Vec:
        return sum([operators(c) for c in prog.children], [])
    elif type(prog) == Lit:
        return []
    else:
        raise Exception(f"Unknown: {prog}")


def dependency_graph(ruleset):
    """Build a graph where the nodes in the graph are all the operators covered
    by the rewrite rules and there is an edge between two nodes if there is a rewrite
    rule that constructs the second node from the first."""

    G = nx.Graph()

    for (lhs, rhs) in ruleset:
        lops = operators(lhs)
        rops = operators(rhs)

        G.add_nodes_from(lops)
        G.add_nodes_from(rops)

        G.add_edges_from(itertools.product(lops, rops))

    return G
        

def main():
    """Become the main function."""
    filename = sys.argv[1]
    with open(filename, "r") as f:
        j = json.load(f)
        n_kept = 0
        ruleset = []
        for eq in j["eqs"]:
            # print(f"{eq['lhs']} <=> {eq['rhs']}")
            lhs = parse(sexp.loads(eq["lhs"]))
            rhs = parse(sexp.loads(eq["rhs"]))
            ruleset.append((lhs, rhs))

        limit = 0
        for (lhs, rhs) in ruleset:
            # if getheads(lhs) != getheads(rhs):
            c = abs(cost(lhs) - cost(rhs))
            cost_diff = c > 2
            all_vars = len(set(leaves(lhs))) == 4 and len(set(leaves(rhs))) == 4
            vec_mac_lhs = (uses_vec_mac(lhs) and tree_depth(lhs) == 1)
            vec_mac_rhs = (tree_depth(rhs) == 1 and uses_vec_mac(rhs))
            vec_mac = vec_mac_lhs or vec_mac_rhs
            if cost_diff:
                n_kept += 1
                print(f"[{c}] {pprint(lhs)} <=> {pprint(rhs)}")

            # if limit > 3:
            #     break
            # limit += 1
        print(f"# kept/total: {n_kept} / {len(j['eqs'])}")

        # G = dependency_graph(ruleset)
        # nx.nx_agraph.write_dot(G, "a.dot")


if __name__ == "__main__":
    main()
