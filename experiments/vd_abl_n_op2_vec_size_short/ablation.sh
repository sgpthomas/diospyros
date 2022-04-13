#!/usr/bin/env bash
function run() {
    echo "Starting vec-$2-vd-$1"
    RUST_LOG=info cargo run --manifest-path ruler/Cargo.toml \
	    --release --bin \
	    dios -- synth \
	    --dios-config "experiments/vd_abl_n_op2_vec_size_short/vd-$1.json" \
	    --variables $((2*$2)) \
	    --iters 2 \
	    --vector-size $2 \
	    --eqsat-iter-limit 2 \
	    --eqsat-time-limit 30 \
	    --num-fuzz 10 \
	    --abs-timeout 60000 \
	    --outfile "experiments/vd_abl_n_op2_vec_size_short/rules-vec-$2-$1.json" \
	    >"experiments/vd_abl_n_op2_vec_size_short/vec-$2-vd-$1.out" \
	    2>"experiments/vd_abl_n_op2_vec_size_short/vec-$2-vd-$1.err" || true
    echo "Finished vec-$2-vd-$1"
    sleep 1
}

rm experiments/vd_abl_n_op2_vec_size_short/rules-*.json

run False 2 || true
run False 3 || true
run False 4 || true
run False 8 || true
run True 2 || true
run True 3 || true
run True 4 || true
run True 8 || true
