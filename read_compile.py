#!/usr/bin/env python3

import glob
import sys

def main():
    root = sys.argv[1]
    for log in glob.glob(f"{root}/*/*/compile-log.txt"):
        # print("Log:", log)
        cost = [log]
        with open(log, "r") as f:
            for line in f.readlines():
                if "Cost" in line:
                    cost.append(line.split(": ")[1].strip())
                    # print(line)
        print(", ".join(cost))

if __name__ == "__main__":
    main()
