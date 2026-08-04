use std::fs;
use std::path::Path;

#[path = "src/cli.rs"]
mod cli;
#[path = "src/run_all_cli.rs"]
mod run_all_cli;

use clap::CommandFactory;
use clap_complete::{generate_to, shells::{Bash, Zsh, Fish, Elvish, PowerShell}};

fn generate_completions_for(mut cmd: clap::Command, out_dir: &Path) {
    let name = cmd.get_name().to_string();
    generate_to(Bash, &mut cmd, &name, out_dir).unwrap();
    generate_to(Zsh, &mut cmd, &name, out_dir).unwrap();
    generate_to(Fish, &mut cmd, &name, out_dir).unwrap();
    generate_to(Elvish, &mut cmd, &name, out_dir).unwrap();
    generate_to(PowerShell, &mut cmd, &name, out_dir).unwrap();
}

fn main() {
    let out_dir = Path::new("completions");
    fs::create_dir_all(out_dir).unwrap();

    generate_completions_for(<cli::Args as CommandFactory>::command(), out_dir);
    generate_completions_for(<run_all_cli::Args as CommandFactory>::command(), out_dir);

    println!("cargo:warning=Generated shell completions in {}", out_dir.display());
}
