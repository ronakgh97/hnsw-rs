#!/usr/bin/env just --justfile

all:
  cargo clippy --fix --all-targets --release
  cargo fmt --all
  cargo check --all-targets --release
  cargo nextest --all-targets --release --no-capture
  cargo build --all-targets --release

lint:
  cargo clippy --all-targets --release