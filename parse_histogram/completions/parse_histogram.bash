_parse_histogram() {
    local i cur prev opts cmd
    COMPREPLY=()
    if [[ "${BASH_VERSINFO[0]}" -ge 4 ]]; then
        cur="$2"
    else
        cur="${COMP_WORDS[COMP_CWORD]}"
    fi
    prev="$3"
    cmd=""
    opts=""

    for i in "${COMP_WORDS[@]:0:COMP_CWORD}"
    do
        case "${cmd},${i}" in
            ",$1")
                cmd="parse_histogram"
                ;;
            parse_histogram,help)
                cmd="parse_histogram__subcmd__help"
                ;;
            parse_histogram,histogram)
                cmd="parse_histogram__subcmd__histogram"
                ;;
            parse_histogram,print)
                cmd="parse_histogram__subcmd__print"
                ;;
            parse_histogram,sparsity)
                cmd="parse_histogram__subcmd__sparsity"
                ;;
            parse_histogram__subcmd__help,help)
                cmd="parse_histogram__subcmd__help__subcmd__help"
                ;;
            parse_histogram__subcmd__help,histogram)
                cmd="parse_histogram__subcmd__help__subcmd__histogram"
                ;;
            parse_histogram__subcmd__help,print)
                cmd="parse_histogram__subcmd__help__subcmd__print"
                ;;
            parse_histogram__subcmd__help,sparsity)
                cmd="parse_histogram__subcmd__help__subcmd__sparsity"
                ;;
            *)
                ;;
        esac
    done

    case "${cmd}" in
        parse_histogram)
            opts="-f -l -b -h --file --layer --batch --help histogram print sparsity help"
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 1 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --file)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                -f)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --layer)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                -l)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --batch)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                -b)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        parse_histogram__subcmd__help)
            opts="histogram print sparsity help"
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        parse_histogram__subcmd__help__subcmd__help)
            opts=""
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 3 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        parse_histogram__subcmd__help__subcmd__histogram)
            opts=""
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 3 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        parse_histogram__subcmd__help__subcmd__print)
            opts=""
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 3 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        parse_histogram__subcmd__help__subcmd__sparsity)
            opts=""
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 3 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        parse_histogram__subcmd__histogram)
            opts="-o -h --output --help"
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --output)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                -o)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        parse_histogram__subcmd__print)
            opts="-n -h --count --help"
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --count)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                -n)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        parse_histogram__subcmd__sparsity)
            opts="-h --help"
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
    esac
}

if [[ "${BASH_VERSINFO[0]}" -eq 4 && "${BASH_VERSINFO[1]}" -ge 4 || "${BASH_VERSINFO[0]}" -gt 4 ]]; then
    complete -F _parse_histogram -o nosort -o bashdefault -o default parse_histogram
else
    complete -F _parse_histogram -o bashdefault -o default parse_histogram
fi
