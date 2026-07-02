# Print an optspec for argparse to handle cmd's options that are independent of any subcommand.
function __fish_parse_histogram_global_optspecs
	string join \n f/file= l/layer= b/batch= h/help
end

function __fish_parse_histogram_needs_command
	# Figure out if the current invocation already has a command.
	set -l cmd (commandline -opc)
	set -e cmd[1]
	argparse -s (__fish_parse_histogram_global_optspecs) -- $cmd 2>/dev/null
	or return
	if set -q argv[1]
		# Also print the command, so this can be used to figure out what it is.
		echo $argv[1]
		return 1
	end
	return 0
end

function __fish_parse_histogram_using_subcommand
	set -l cmd (__fish_parse_histogram_needs_command)
	test -z "$cmd"
	and return 1
	contains -- $cmd[1] $argv
end

complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -s f -l file -d 'Path to the binary file' -r -F
complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -s l -l layer -d 'Filter by layer (only process records with this layer id)' -r
complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -s b -l batch -d 'Filter by batch (only process records with this batch id)' -r
complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -s h -l help -d 'Print help'
complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -f -a "histogram" -d 'Compute per-layer histograms of positive-score positions'
complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -f -a "print" -d 'Print info of the first N records'
complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -f -a "sparsity" -d 'Compute sparsity statistics (overall and per-layer)'
complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -f -a "simulate" -d 'Run PIM simulation with given activation threshold'
complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -f -a "to-cycle" -d 'Convert simulation stats to cycle counts (auto-runs simulation if needed)'
complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -f -a "parse-json"
complete -c parse_histogram -n "__fish_parse_histogram_needs_command" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand histogram" -s o -l output -d 'Save histogram as JSON to this file' -r -F
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand histogram" -s h -l help -d 'Print help'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand print" -s n -l count -d 'Number of records to print' -r
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand print" -s h -l help -d 'Print help'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand sparsity" -s h -l help -d 'Print help'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand simulate" -s t -l threshold -d 'Activation threshold (default: 0.0)' -r
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand simulate" -s o -l output -d 'Save result as JSON to this file (auto-derived if omitted)' -r -F
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand simulate" -s r -l remap -d 'Path to remap JSON for balanced bank placement' -r -F
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand simulate" -s h -l help -d 'Print help'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand to-cycle" -s t -l threshold -d 'Activation threshold for simulation (default: 0.0)' -r
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand to-cycle" -s o -l output -d 'Save cycle result as JSON (default: stdout)' -r -F
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand to-cycle" -s r -l remap -d 'Path to remap JSON for balanced bank placement' -r -F
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand to-cycle" -s h -l help -d 'Print help'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand parse-json" -s h -l help -d 'Print help'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand help; and not __fish_seen_subcommand_from histogram print sparsity simulate to-cycle parse-json help" -f -a "histogram" -d 'Compute per-layer histograms of positive-score positions'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand help; and not __fish_seen_subcommand_from histogram print sparsity simulate to-cycle parse-json help" -f -a "print" -d 'Print info of the first N records'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand help; and not __fish_seen_subcommand_from histogram print sparsity simulate to-cycle parse-json help" -f -a "sparsity" -d 'Compute sparsity statistics (overall and per-layer)'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand help; and not __fish_seen_subcommand_from histogram print sparsity simulate to-cycle parse-json help" -f -a "simulate" -d 'Run PIM simulation with given activation threshold'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand help; and not __fish_seen_subcommand_from histogram print sparsity simulate to-cycle parse-json help" -f -a "to-cycle" -d 'Convert simulation stats to cycle counts (auto-runs simulation if needed)'
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand help; and not __fish_seen_subcommand_from histogram print sparsity simulate to-cycle parse-json help" -f -a "parse-json"
complete -c parse_histogram -n "__fish_parse_histogram_using_subcommand help; and not __fish_seen_subcommand_from histogram print sparsity simulate to-cycle parse-json help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
