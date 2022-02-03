#!/usr/bin/env python3
"""A simple testbed for running experiments on AutoDios."""

import json
import sys
from functools import reduce
import datetime
import subprocess as sp
from pathlib import Path
import time
from itertools import product
from multiprocessing import Pool


def reform(arg_list):
    """Reform arg list so that we can run it as a command."""
    filt = filter(lambda x: x != "", arg_list)
    return list(sum(map(lambda x: x.split(" "), filt), []))


class Parameter:
    """Represents a parameter in configuration space."""

    def __init__(self, name, flag, args=[]):
        """Construct a parameter."""
        self.name = name
        self.flag = flag
        self.args = args

    def to_list(self):
        """Convert parameter to a list of strings."""
        res = []
        if len(self.args) == 0:
            res += [self.flag]
        else:
            for arg_a in self.args:
                res.append(f"{self.flag} {arg_a}")
        res += [""]
        return res

    def __mul__(self, other):
        """Return a list representing all combinations of parameters."""
        res = []
        for x in self.to_list():
            for y in other.to_list():
                res.append([x, y])

        print(self.to_list(), other.to_list(), res)
        return res

    def __rmul__(self, other):
        """Multiply a list of parameters by a parameter."""
        assert(isinstance(other, list))
        res = []
        for x in other:
            for y in self.to_list():
                res.append(x + [y])
        return res

    def __repr__(self):
        """Define the representation for a Parameter."""
        return f"<'{self.name}' {self.flag} {self.args}>"


def run_experiment(exp_id, config, benchmark, exp):
    """Run experiment on input `inp` using args from `exp`."""
    print(f"[{exp_id}] Started {benchmark} {' '.join(exp)}")

    inp = config["input_template"].format(benchmark)
    base = [config["base_command"], inp, config["extra"]]
    full = reform(base + exp)
    stdout, stderr = "", ""
    exec_time = None

    try:
        start = time.time()
        res = sp.run(full, timeout=config["timeout"], capture_output=True)
        end = time.time()
        exec_time = end - start
        stdout = res.stdout.decode("utf-8")
        stderr = res.stderr.decode("utf-8")
    except sp.TimeoutExpired as e:
        exec_time = "Timed out."
        if e.stdout is not None:
            stdout = e.stdout.decode("utf-8")

        if e.stderr is not None:
            stderr = e.stderr.decode("utf-8")

    return (exec_time, stdout, stderr)


def main():
    """Start testbed."""
    assert(len(sys.argv) > 1)
    config_file = sys.argv[1]
    config = None

    with open(config_file, "r") as f:
        config = json.load(f)

    benchmarks = config["benchmarks"]
    timeout = config["timeout"]

    params = []
    for param in config["parameters"]:
        params.append(Parameter(param["name"], param["flag"], param["args"]))

    exps = reduce(lambda x, y: x * y, params)
    exps = map(lambda x: list(filter(lambda y: y != "", x)), exps)
    exps = list(exps)
    for e in exps:
        print(list(e))

    n_experiments = len(exps) * len(benchmarks)
    est_time = datetime.timedelta(seconds=n_experiments * timeout)
    print(f"Found {n_experiments} experiments.")
    print(f"Max time: {est_time}.")

    print("Compiling binary.")
    sp.run(config["compile_command"].split(" "))

    results = {"experiments": []}
    assert(len(sys.argv) > 2)
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(exist_ok=False)
    keyf = output_dir / "key.json"
    keyf.touch()
    keyf_fp = keyf.open("w")

    with Pool(processes=2) as pool:
        jobs = {}
        for eid, (exp, bench) in enumerate(product(exps, benchmarks)):
            args = (eid, config, bench, exp)
            jobs[eid] = {
                "result": pool.apply_async(run_experiment, args),
                "bench": bench,
                "exp": exp
            }

        # We have queued all the jobs.
        pool.close()

        # collect jobs and commit results
        for eid, j in jobs.items():
            (time, out, err) = j["result"].get()
            bench = j["bench"]
            exp = j["exp"]
            print(f"[{eid}/{n_experiments}] {bench} took {time}")

            result_fn_stdout = output_dir / f"{eid}-{bench}.out"
            result_fn_stderr = output_dir / f"{eid}-{bench}.err"
            if out != "":
                result_fn_stdout.touch()
                result_fn_stdout.write_text(out)
            if err != "":
                result_fn_stderr.touch()
                result_fn_stderr.write_text(err)

            results["experiments"] += [{
                "bench": bench,
                "cmd": " ".join(exp),
                "time": time,
                "stdout": result_fn_stdout.parts[-1],
                "stderr": result_fn_stderr.parts[-1],
            }]
            keyf_fp.seek(0)
            json.dump(results, keyf_fp, indent=2)
            keyf_fp.flush()

        pool.join()


if __name__ == "__main__":
    main()
