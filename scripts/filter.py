#!/usr/bin/env python3

import sys
import json
import sexpdata as sexp
from veclang import *
import networkx as nx
import itertools

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


def leaves_f(prog, typ):
    return list(filter(lambda it: type(it), leaves(prog)))


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

        # limit = 0
        for (lhs, rhs) in ruleset:
            # if getheads(lhs) != getheads(rhs):
            c = cost(lhs) - cost(rhs)
            cost_diff = c > 2
            all_vars = len(set(leaves(lhs))) == 4 and len(set(leaves(rhs))) == 4
            no_repeated = sorted(set(leaves_f(lhs, str))) == sorted(leaves(lhs))
            vec_mac_lhs = (uses_vec_mac(lhs) and tree_depth(lhs) == 1)
            vec_mac_rhs = (tree_depth(rhs) == 1 and uses_vec_mac(rhs))
            vec_mac = vec_mac_lhs or vec_mac_rhs
            if no_repeated and cost_diff:
                n_kept += 1
                if c > 0:
                    print(f"[{c}] {pprint(lhs)} <=> {pprint(rhs)}")
                else:
                    print(f"[{-c}] {pprint(rhs)} <=> {pprint(lhs)}")
            # if limit > 3:
            #     break
            # limit += 1

        # check = "(Vec ?a 0)"
        # check_tree = parse(sexp.loads(check))

        # for (lhs, rhs) in ruleset:
        #     if lhs == check_tree or rhs == check_tree:
        #         print(f"{pprint(lhs)} <=> {pprint(rhs)}")
                

        print(f"# kept/total: {n_kept} / {len(j['eqs'])}")

        # G = dependency_graph(ruleset)
        # nx.nx_agraph.write_dot(G, "a.dot")


if __name__ == "__main__":
    main()
