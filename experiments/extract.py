#!/usr/bin/env python3
"""Extract data from experiment results."""

import sys
import json
import csv
from pathlib import Path
import re


def get_cost(res_dir, exp):
    """Extract program cost from an experiment."""
    err_file = res_dir / Path(exp["stderr"])
    with err_file.open("r") as f:
        lines = f.readlines()
        if "Cost:" in lines[-1]:
            return float(lines[-1].split(" ")[1].strip())
        else:
            return -1.0


def get_n_rules(res_dir, exp, n):
    """Extract number of rules from an experiment."""
    err_file = res_dir / Path(exp["stderr"])
    phase_n_rules = []
    with err_file.open("r") as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith("Starting run with"):
                phase_n_rules.append(int(l.split(" ")[3]))
    if n < len(phase_n_rules):
        return phase_n_rules[n]
    else:
        return None


def get_improved_cost(res_dir, exp):
    """Extract number of rules from an experiment."""
    err_file = res_dir / Path(exp["stderr"])
    res = None
    with err_file.open("r") as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith("Improved cost by"):
                res = float(l.split(" ")[3])
    return res


def generate_metadata(parameters, exp):
    """Generate experiment metadata."""
    res = []
    for p in parameters:
        if len(p["args"]) == 0:
            res += [p["flag"] in exp["cmd"]]
        else:
            if p["flag"] not in exp["cmd"]:
                res += [None]
            else:
                for val in p["args"]:
                    regex = re.compile(f"{p['flag']} {val}\\b")
                    if re.search(regex, exp["cmd"]) is not None:
                        res += [val]
                        break
    return res


def main():
    """Perform the main function."""
    config_file = Path(sys.argv[1])
    result_dir = Path(sys.argv[2])
    output_fn = Path(sys.argv[3])

    config = json.load(config_file.open("r"))

    parameters = config["parameters"]

    keyf = json.load((result_dir / "key.json").open("r"))
    data = keyf["experiments"]

    data_fields = [
        ("time", lambda e: e["time"]),
        ("cost", lambda e: get_cost(result_dir, e)),
        ("phase_1_n", lambda e: get_n_rules(result_dir, e, 0)),
        ("phase_2_n", lambda e: get_n_rules(result_dir, e, 1)),
        ("phase_3_n", lambda e: get_n_rules(result_dir, e, 2)),
        ("impr_cost", lambda e: get_improved_cost(result_dir, e))
    ]

    headers = [p["name"] for p in parameters] + ["bench", "id", "variable", "value"]

    with output_fn.open("w") as f:
        c = csv.writer(f, delimiter=",")
        c.writerow(headers)
        for exp in data:
            meta = generate_metadata(parameters, exp)
            eid = exp["stdout"].split("-")[0]
            for field, fn in data_fields:
                row = meta + [exp["bench"], eid, field, fn(exp)]
                c.writerow(row)


if __name__ == "__main__":
    main()
