# Diospyros for LLVM

This directory contains an experimental [LLVM][] pass that optimizes programs using Diospyros.

## Build It

To get started, you will need **LLVM 11.x.x**.
Using [Homebrew][] on macOS, for example, try `brew install llvm@11` to get the right version.

Because our Rust library relies on [the `llvm-sys` crate][llvm-sys], you will need an existing installation of `llvm-config` on your `$PATH`.
To use a Homebrew-installed LLVM, for example, you may need something like this:

    $ export PATH=`brew --prefix llvm@11`/bin:$PATH

If you're on macOS, you will also need an annoying hack to make the library build (in these [Rust](https://github.com/rust-lang/rust/issues/62874) [bugs](https://github.com/rust-lang/cargo/issues/8628)).
Add a file `.cargo/config` here, in this directory, with these [contents](https://pyo3.rs/v0.5.2/):

    [target.x86_64-apple-darwin]
    rustflags = [
      "-C", "link-arg=-undefined",
      "-C", "link-arg=dynamic_lookup",
    ]

Then, build the pass library with:

    $ cargo build

## Run the Pass

Use this [Clang][] invocation to run the optimization on a C source file:

    $ clang -Xclang -load -Xclang target/debug/libllvmlib.so a.c

To see the generated LLVM IR, you'll need the `-emit-llvm` flag, among a few others:

    $ clang -Xclang -load -Xclang target/debug/libllvmlib.so -emit-llvm -S -o - a.c

You can also pass the `-O2` flag to see the optimized LLVM IR:

    $ clang -O2 -Xclang -load -Xclang target/debug/libllvmlib.so -emit-llvm -S -o - a.c

Note that you may need to change `.so` to `.dylib` on macOS. (Check which file extension actually exists.)

As usual, make sure that the `clang` you invoke here is from the same LLVM installation against which you built the pass above.

## Testing

Test files provided in the llvm-tests/ folder can be run with [Runt][]:

    $ runt

If Runt was not installed during the Diospyros installation process, it can be installed with 

    $ cargo install runt

[llvm]: https://llvm.org
[clang]: https://clang.llvm.org
[llvm-sys]: https://crates.io/crates/llvm-sys
[homebrew]: https://brew.sh
[runt]: https://github.com/rachitnigam/runt