#!/bin/sh
# Start praatshell from Git Bash. If nothing echoes as you type, run
# "winpty ./praatshell.sh" instead: MinTTY needs that for some interactive
# programs, though the VS Code terminal usually does not.
cd "$(dirname "$0")" || exit 1
exec python -u -m praatshell "$@"
