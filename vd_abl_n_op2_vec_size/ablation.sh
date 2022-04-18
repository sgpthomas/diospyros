#!/usr/bin/env bash

function run() {
    echo "Starting vec-$2-vd-$1"
    RUST_LOG=info cargo run --manifest-path ruler/Cargo.toml \
	    --release --bin \
	    dios -- synth \
	    --dios-config "vd_abl_n_op2_vec_size/vd-$1.json" \
	    --variables $((2*$2)) \
	    --iters 2 \
	    --vector-size $2 \
	    --eqsat-iter-limit 3 \
	    --eqsat-time-limit 120 \
	    --num-fuzz 10 \
	    --abs-timeout 60000 \
	    --outfile "vd_abl_n_op2_vec_size/rules-vec-$2-$1.json" \
	    >"vd_abl_n_op2_vec_size/vec-$2-vd-$1.out" \
	    2>"vd_abl_n_op2_vec_size/vec-$2-vd-$1.err" || true
    echo "Finished vec-$2-vd-$1"
    sleep 5
}

rm vd_abl_n_op2_vec_size/rules-*.json

# run False 2 || true
# run True 2 || true
# run False 3 || true
run True 3 || true
run False 4 || true
run True 4 || true
run False 8 || true
run True 8 || true
