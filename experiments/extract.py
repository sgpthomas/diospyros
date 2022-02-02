#!/usr/bin/env python3
"""Extract data from experiment results."""

import sys
import json
import csv
from pathlib import Path


def get_cost(exp):
    """Extract program cost from an experiment."""
    err_file = Path(exp["stderr"])
    with err_file.open("r") as f:
        lines = f.readlines()
        if "Cost:" in lines[-1]:
            return float(lines[-1].split(" ")[1].strip())
        else:
            return -1.0


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
                    if f"{p['flag']} {val}" in exp["cmd"]:
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
        ("cost", get_cost)
    ]

    headers = [p["name"] for p in parameters] + ["bench", "variable", "value"]

    with output_fn.open("w") as f:
        c = csv.writer(f, delimiter=",")
        c.writerow(headers)
        for exp in data:
            meta = generate_metadata(parameters, exp)
            for field, fn in data_fields:
                row = meta + [exp["bench"], field, fn(exp)]
                c.writerow(row)


if __name__ == "__main__":
    main()
