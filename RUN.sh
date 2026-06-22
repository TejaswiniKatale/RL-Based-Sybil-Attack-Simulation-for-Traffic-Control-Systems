#!/usr/bin/env bash
set -e
netconvert -n nodes.xml -e edges.xml -o intersection.net.xml
echo "Train: python attack_agent.py --train --episodes 50"
echo "Eval : python attack_agent.py --eval --model sybil_attacker.pth --gui"
