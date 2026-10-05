#[path = "src/search_index.rs"]
mod search_index;

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("../src/search/content-index.json");
    println!("cargo:rerun-if-changed={}", json_path.display());

    if json_path.exists() {
        let json = std::fs::read_to_string(&json_path).expect("read packaged search index JSON");
        let sqlite_path = manifest_dir.join("resources/content-index.sqlite");
        search_index::write_index_file(&json, &sqlite_path)
            .unwrap_or_else(|error| panic!("failed to write packaged search index: {error}"));
    }

    tauri_build::build()
}
