#!/usr/bin/env bash

function run() {
    RUST_LOG=info cargo run --manifest-path ruler/Cargo.toml \
	    --release --bin \
	    dios -- synth \
	    --dios-config "variable_dup_abl_n_ops/$1.json" \
	    --variables 4 \
	    --iters 2 \
	    --vector-size 2 \
	    --eqsat-iter-limit 3 \
	    --eqsat-time-limit 120 \
	    --num-fuzz 6 \
	    --abs-timeout 6000 \
	    --outfile "variable_dup_abl_n_ops/rules-$1.json" \
	    >"variable_dup_abl_n_ops/$1.out" \
	    2>"variable_dup_abl_n_ops/$1.err" || true
    echo "Finished $1"
}

# run "1-vd-False"
# run "1-vd-True"
# run "2-vd-False"
# run "2-vd-True"
# run "3-vd-False"
# run "3-vd-True"
run "4-vd-False"
run "4-vd-True"
