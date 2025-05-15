#!/bin/bash
# pinet-terminate-mac.sh - Terminates all ncat and tar processes on macOS

echo "Terminating all ncat and tar processes..."

# Kill ncat processes
pkill -f ncat

# Kill tar processes
pkill -f tar

echo "Termination complete."