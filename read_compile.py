#!/usr/bin/env python3

import glob
import sys

def main():
    root = sys.argv[1]
    for log in glob.glob(f"{root}/*/*/compile-log.txt"):
        print("Log:", log)
        with open(log, "r") as f:
            for line in f.readlines():
                if "Cost" in line:
                    print(line)

if __name__ == "__main__":
    main()
