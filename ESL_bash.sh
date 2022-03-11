#!/bin/bash
echo "Nice to meet you!"
echo "ESL training ...."
# python run.py ESL -g 0 -mle -lr 0.006 -bs 32 -e 7 -w_gen 0.75
# python run.py ESL -g 0 -mle -rl -l experiments/ESL-lr0.006-eps7-MLE1.0 -lr 0.0005 -bs 16 -e 5 -w_f1 0.5 -w_re 0.1 -w_mle 0.25

python run.py ESL -g 0 -mle -lr 0.006 -bs 32 -e 15 -w_gen 0.75
python run.py ESL -g 0 -mle -rl -l experiments/ESL-lr0.006-eps15-MLE1.0 -lr 0.0005 -bs 16 -e 15 -w_f1 0.5 -w_re 0.15 -w_mle 0.1

