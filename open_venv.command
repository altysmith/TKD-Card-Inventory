#!/bin/bash

cd "$(dirname "$0")" || exit 1

if [ ! -f ".venv/bin/activate" ]; then
    echo "Could not find .venv/bin/activate"
    echo
    echo "Create the macOS virtual environment with:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  python -m pip install -r requirements.txt"
    echo
    read -r -p "Press Return to close..."
    exit 1
fi

source ".venv/bin/activate"

echo
echo "Virtual environment activated."
echo "Project folder: $(pwd)"
echo "Type exit when you are finished."
echo

exec "${SHELL:-/bin/zsh}" -i
