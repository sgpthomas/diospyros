#!/usr/bin/env python3

import glob
import sys
import re
import json

def main():
    root = sys.argv[1]
    for log in glob.glob(f"{root}/*/*/stats.json"):
        with open(log, "r") as f:
            j = json.load(f)
            cost = j["cost"]
            time = j["time"]
            print(f"{log}, {cost}, {time}")

if __name__ == "__main__":
    main()
