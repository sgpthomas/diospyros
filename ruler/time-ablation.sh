#!/usr/bin/env bash

function run() {
    RUST_LOG=info cargo run --manifest-path ruler/Cargo.toml \
		  --release --bin \
		  dios -- synth \
		  --dios-config "configs/vec_assoc.json" \
		  --iters 3 \
		  --vector-size 2 \
		  --eqsat-iter-limit 3 \
		  --eqsat-time-limit 60 \
		  --num-fuzz 4 \
		  --abs-timeout "$1" \
		  --outfile "time-ablation/rules-t-$1.json"
}

mkdir -p "time-ablation"
run 60 
run 600
run 6000
run 60000
