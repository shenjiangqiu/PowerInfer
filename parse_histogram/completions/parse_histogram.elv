
use builtin;
use str;

set edit:completion:arg-completer[parse_histogram] = {|@words|
    fn spaces {|n|
        builtin:repeat $n ' ' | str:join ''
    }
    fn cand {|text desc|
        edit:complex-candidate $text &display=$text' '(spaces (- 14 (wcswidth $text)))$desc
    }
    var command = 'parse_histogram'
    for word $words[1..-1] {
        if (str:has-prefix $word '-') {
            break
        }
        set command = $command';'$word
    }
    var completions = [
        &'parse_histogram'= {
            cand -f 'Path to the binary file'
            cand --file 'Path to the binary file'
            cand -l 'Filter by layer (only process records with this layer id)'
            cand --layer 'Filter by layer (only process records with this layer id)'
            cand -b 'Filter by batch (only process records with this batch id)'
            cand --batch 'Filter by batch (only process records with this batch id)'
            cand -h 'Print help'
            cand --help 'Print help'
            cand histogram 'Compute per-layer histograms of positive-score positions'
            cand print 'Print info of the first N records'
            cand sparsity 'Compute sparsity statistics (overall and per-layer)'
            cand simulate 'Run PIM simulation with given activation threshold'
            cand to-cycle 'Convert simulation stats to cycle counts (auto-runs simulation if needed)'
            cand help 'Print this message or the help of the given subcommand(s)'
        }
        &'parse_histogram;histogram'= {
            cand -o 'Save histogram as JSON to this file'
            cand --output 'Save histogram as JSON to this file'
            cand -h 'Print help'
            cand --help 'Print help'
        }
        &'parse_histogram;print'= {
            cand -n 'Number of records to print'
            cand --count 'Number of records to print'
            cand -h 'Print help'
            cand --help 'Print help'
        }
        &'parse_histogram;sparsity'= {
            cand -h 'Print help'
            cand --help 'Print help'
        }
        &'parse_histogram;simulate'= {
            cand -t 'Activation threshold (default: 0.0)'
            cand --threshold 'Activation threshold (default: 0.0)'
            cand -o 'Save result as JSON to this file (auto-derived if omitted)'
            cand --output 'Save result as JSON to this file (auto-derived if omitted)'
            cand -r 'Path to remap JSON for balanced bank placement'
            cand --remap 'Path to remap JSON for balanced bank placement'
            cand -h 'Print help'
            cand --help 'Print help'
        }
        &'parse_histogram;to-cycle'= {
            cand -t 'Activation threshold for simulation (default: 0.0)'
            cand --threshold 'Activation threshold for simulation (default: 0.0)'
            cand -o 'Save cycle result as JSON (default: stdout)'
            cand --output 'Save cycle result as JSON (default: stdout)'
            cand -r 'Path to remap JSON for balanced bank placement'
            cand --remap 'Path to remap JSON for balanced bank placement'
            cand -h 'Print help'
            cand --help 'Print help'
        }
        &'parse_histogram;help'= {
            cand histogram 'Compute per-layer histograms of positive-score positions'
            cand print 'Print info of the first N records'
            cand sparsity 'Compute sparsity statistics (overall and per-layer)'
            cand simulate 'Run PIM simulation with given activation threshold'
            cand to-cycle 'Convert simulation stats to cycle counts (auto-runs simulation if needed)'
            cand help 'Print this message or the help of the given subcommand(s)'
        }
        &'parse_histogram;help;histogram'= {
        }
        &'parse_histogram;help;print'= {
        }
        &'parse_histogram;help;sparsity'= {
        }
        &'parse_histogram;help;simulate'= {
        }
        &'parse_histogram;help;to-cycle'= {
        }
        &'parse_histogram;help;help'= {
        }
    ]
    $completions[$command]
}
