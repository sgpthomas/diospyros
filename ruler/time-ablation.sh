#!/usr/bin/env bash

function run() {
    RUST_LOG=info cargo run --manifest-path ruler/Cargo.toml \
		  --release --bin \
		  dios -- synth \
		  --dios-config "configs/vec_assoc.json" \
		  --variables 4 \
		  --iters 3 \
		  --vector-size 2 \
		  --eqsat-iter-limit 3 \
		  --eqsat-time-limit 120 \
		  --num-fuzz 4 \
		  --abs-timeout "$1" \
		  --outfile "time-ablation-var4/rules-t-$1.json"
}

mkdir -p "time-ablation-var4"
run 60 
run 600
run 6000
run 60000
run 100000
run 200000
