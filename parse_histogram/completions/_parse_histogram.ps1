
using namespace System.Management.Automation
using namespace System.Management.Automation.Language

Register-ArgumentCompleter -Native -CommandName 'parse_histogram' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    $commandElements = $commandAst.CommandElements
    $command = @(
        'parse_histogram'
        for ($i = 1; $i -lt $commandElements.Count; $i++) {
            $element = $commandElements[$i]
            if ($element -isnot [StringConstantExpressionAst] -or
                $element.StringConstantType -ne [StringConstantType]::BareWord -or
                $element.Value.StartsWith('-') -or
                $element.Value -eq $wordToComplete) {
                break
        }
        $element.Value
    }) -join ';'

    $completions = @(switch ($command) {
        'parse_histogram' {
            [CompletionResult]::new('-f', '-f', [CompletionResultType]::ParameterName, 'Path to the binary file')
            [CompletionResult]::new('--file', '--file', [CompletionResultType]::ParameterName, 'Path to the binary file')
            [CompletionResult]::new('-l', '-l', [CompletionResultType]::ParameterName, 'Filter by layer (only process records with this layer id)')
            [CompletionResult]::new('--layer', '--layer', [CompletionResultType]::ParameterName, 'Filter by layer (only process records with this layer id)')
            [CompletionResult]::new('-b', '-b', [CompletionResultType]::ParameterName, 'Filter by batch (only process records with this batch id)')
            [CompletionResult]::new('--batch', '--batch', [CompletionResultType]::ParameterName, 'Filter by batch (only process records with this batch id)')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('histogram', 'histogram', [CompletionResultType]::ParameterValue, 'Compute per-layer histograms of positive-score positions')
            [CompletionResult]::new('print', 'print', [CompletionResultType]::ParameterValue, 'Print info of the first N records')
            [CompletionResult]::new('sparsity', 'sparsity', [CompletionResultType]::ParameterValue, 'Compute sparsity statistics (overall and per-layer)')
            [CompletionResult]::new('simulate', 'simulate', [CompletionResultType]::ParameterValue, 'Run PIM simulation with given activation threshold')
            [CompletionResult]::new('to-cycle', 'to-cycle', [CompletionResultType]::ParameterValue, 'Convert simulation stats to cycle counts (auto-runs simulation if needed)')
            [CompletionResult]::new('help', 'help', [CompletionResultType]::ParameterValue, 'Print this message or the help of the given subcommand(s)')
            break
        }
        'parse_histogram;histogram' {
            [CompletionResult]::new('-o', '-o', [CompletionResultType]::ParameterName, 'Save histogram as JSON to this file')
            [CompletionResult]::new('--output', '--output', [CompletionResultType]::ParameterName, 'Save histogram as JSON to this file')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'parse_histogram;print' {
            [CompletionResult]::new('-n', '-n', [CompletionResultType]::ParameterName, 'Number of records to print')
            [CompletionResult]::new('--count', '--count', [CompletionResultType]::ParameterName, 'Number of records to print')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'parse_histogram;sparsity' {
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'parse_histogram;simulate' {
            [CompletionResult]::new('-t', '-t', [CompletionResultType]::ParameterName, 'Activation threshold (default: 0.0)')
            [CompletionResult]::new('--threshold', '--threshold', [CompletionResultType]::ParameterName, 'Activation threshold (default: 0.0)')
            [CompletionResult]::new('-o', '-o', [CompletionResultType]::ParameterName, 'Save result as JSON to this file (auto-derived if omitted)')
            [CompletionResult]::new('--output', '--output', [CompletionResultType]::ParameterName, 'Save result as JSON to this file (auto-derived if omitted)')
            [CompletionResult]::new('-r', '-r', [CompletionResultType]::ParameterName, 'Path to remap JSON for balanced bank placement')
            [CompletionResult]::new('--remap', '--remap', [CompletionResultType]::ParameterName, 'Path to remap JSON for balanced bank placement')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'parse_histogram;to-cycle' {
            [CompletionResult]::new('-t', '-t', [CompletionResultType]::ParameterName, 'Activation threshold for simulation (default: 0.0)')
            [CompletionResult]::new('--threshold', '--threshold', [CompletionResultType]::ParameterName, 'Activation threshold for simulation (default: 0.0)')
            [CompletionResult]::new('-o', '-o', [CompletionResultType]::ParameterName, 'Save cycle result as JSON (default: stdout)')
            [CompletionResult]::new('--output', '--output', [CompletionResultType]::ParameterName, 'Save cycle result as JSON (default: stdout)')
            [CompletionResult]::new('-r', '-r', [CompletionResultType]::ParameterName, 'Path to remap JSON for balanced bank placement')
            [CompletionResult]::new('--remap', '--remap', [CompletionResultType]::ParameterName, 'Path to remap JSON for balanced bank placement')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'parse_histogram;help' {
            [CompletionResult]::new('histogram', 'histogram', [CompletionResultType]::ParameterValue, 'Compute per-layer histograms of positive-score positions')
            [CompletionResult]::new('print', 'print', [CompletionResultType]::ParameterValue, 'Print info of the first N records')
            [CompletionResult]::new('sparsity', 'sparsity', [CompletionResultType]::ParameterValue, 'Compute sparsity statistics (overall and per-layer)')
            [CompletionResult]::new('simulate', 'simulate', [CompletionResultType]::ParameterValue, 'Run PIM simulation with given activation threshold')
            [CompletionResult]::new('to-cycle', 'to-cycle', [CompletionResultType]::ParameterValue, 'Convert simulation stats to cycle counts (auto-runs simulation if needed)')
            [CompletionResult]::new('help', 'help', [CompletionResultType]::ParameterValue, 'Print this message or the help of the given subcommand(s)')
            break
        }
        'parse_histogram;help;histogram' {
            break
        }
        'parse_histogram;help;print' {
            break
        }
        'parse_histogram;help;sparsity' {
            break
        }
        'parse_histogram;help;simulate' {
            break
        }
        'parse_histogram;help;to-cycle' {
            break
        }
        'parse_histogram;help;help' {
            break
        }
    })

    $completions.Where{ $_.CompletionText -like "$wordToComplete*" } |
        Sort-Object -Property ListItemText
}
