#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$SCRIPT_DIR/.."
sudo cp "$ROOT_DIR/config/hosts.template" /etc/hosts
echo "Hosts file synced."
