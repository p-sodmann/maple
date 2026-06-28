//! Compiles the Slint UI markup (`ui/app.slint` and its imports) into Rust
//! code, surfaced in the crate via `slint::include_modules!()`.

fn main() {
    slint_build::compile("ui/app.slint").expect("failed to compile ui/app.slint");
}
