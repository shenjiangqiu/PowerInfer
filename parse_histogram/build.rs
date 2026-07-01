use std::fs;
use std::path::Path;

#[path = "src/cli.rs"]
mod cli;

use clap::CommandFactory;
use clap_complete::{generate_to, shells::{Bash, Zsh, Fish, Elvish, PowerShell}};

fn main() {
    let out_dir = Path::new("completions");
    fs::create_dir_all(out_dir).unwrap();

    let mut cmd = <cli::Args as CommandFactory>::command();
    let name = cmd.get_name().to_string();

    generate_to(Bash, &mut cmd, &name, out_dir).unwrap();
    generate_to(Zsh, &mut cmd, &name, out_dir).unwrap();
    generate_to(Fish, &mut cmd, &name, out_dir).unwrap();
    generate_to(Elvish, &mut cmd, &name, out_dir).unwrap();
    generate_to(PowerShell, &mut cmd, &name, out_dir).unwrap();

    println!("cargo:warning=Generated shell completions in {}", out_dir.display());
}
