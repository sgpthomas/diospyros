#!/usr/bin/env bash

# enable command echoing
set -x

DIR="rulesets/$1"
PREFIX="n"
RULER_FLAGS="--num-fuzz 4 --iters 2 --variables 4 --eqsat-iter-limit 2 --vector-size 2"

function run() {
    export RUST_LOG=info
    cargo run --manifest-path ruler/Cargo.toml --release --bin \
	  dios -- synth $RULER_FLAGS --rules-to-take "$1" \
	  --outfile "$DIR/$PREFIX-$1-rules.json" \
	  >> "$DIR/stdout.log" \
	  2>> "$DIR/stderr.log"
}

mkdir -p "$DIR"
run 1
run 2
run 3
run 4
run 5
run 6
run 7
run 8
run 9
run 10 
run 15 
run 20
run 25
run 30
run 35
run 40
run 45
run 50
run 55
run 60
run 65
run 70
run 75
run 80
run 85
run 90
run 95
run 100

