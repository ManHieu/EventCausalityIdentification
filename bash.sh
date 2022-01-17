#!/bin/bash
echo "Nice to meet you - Wish you have a great performance :))"
python run.py Causal-TB -g 1 -mle -lr 0.006 -bs 32 -e 15 -w_gen 0.75
python run.py Causal-TB -g 1 -mle -rl -l experiments/last_verson-Causal-TB-lr0.006-eps15-MLE1.0 -lr 0.0005 -bs 16 -e 15 -w_f1 0.5 -w_re 0.1 -w_mle 0.25
