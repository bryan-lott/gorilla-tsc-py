//! Runs the Python test suite against the extension module this `cargo test`
//! just built, so that `cargo mutants` sees binding mutants through pytest.
//!
//! Python is `$GTSC_PYTHON` or the workspace `.venv`; it needs pytest and
//! hypothesis (`uv sync`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Build the cdylib from the current sources. `cargo test` only builds the
/// rlib, and the outer cargo holds the main target dir's lock, so this uses a
/// target dir of its own (incremental across runs).
fn built_extension() -> PathBuf {
    let ws = workspace();
    let target = ws.join("target/pytest-ext");
    let status = Command::new(env!("CARGO"))
        .args(["build", "--quiet", "-p", "gorilla-tsc-py", "--lib"])
        .env("CARGO_TARGET_DIR", &target)
        .current_dir(&ws)
        .status()
        .unwrap();
    assert!(status.success(), "building the extension failed");
    let name = if cfg!(windows) {
        "_native.dll"
    } else if cfg!(target_os = "macos") {
        "lib_native.dylib"
    } else {
        "lib_native.so"
    };
    target.join("debug").join(name)
}

#[test]
fn python_suite_passes_against_this_build() {
    let ws = workspace();
    let venv_python = if cfg!(windows) { ".venv/Scripts/python.exe" } else { ".venv/bin/python" };
    let python = std::env::var_os("GTSC_PYTHON").map_or_else(|| ws.join(venv_python), PathBuf::from);
    assert!(python.exists(), "no Python at {}; run `uv sync` or set GTSC_PYTHON", python.display());

    // Stage the package: pure-Python sources plus the fresh extension.
    let stage = tempdir();
    let pkg = stage.join("gorilla_tsc_py");
    std::fs::create_dir_all(&pkg).unwrap();
    for entry in std::fs::read_dir(ws.join("python/gorilla_tsc_py")).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().is_some_and(|e| e == "py" || e == "pyi") {
            std::fs::copy(&p, pkg.join(p.file_name().unwrap())).unwrap();
        }
    }
    // The name Python imports an extension module under on this platform.
    let module = if cfg!(windows) { "_native.pyd" } else { "_native.abi3.so" };
    std::fs::copy(built_extension(), pkg.join(module)).unwrap();

    let status = Command::new(python)
        .args(["-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--import-mode=importlib"])
        .arg(ws.join("tests"))
        .env("PYTHONPATH", &stage)
        .env("PYTHONSAFEPATH", "1")
        .env("HYPOTHESIS_PROFILE", "mutants")
        .current_dir(&stage)
        .status()
        .unwrap();
    std::fs::remove_dir_all(&stage).ok();
    assert!(status.success(), "pytest failed");
}

fn tempdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("gtsc-py-{}-{:?}", std::process::id(), std::thread::current().id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}
