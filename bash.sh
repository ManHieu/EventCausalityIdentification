#!/bin/bash
echo "Nice to meet you"
# python run.py ESL -g 0 -mle -lr 0.006 -bs 32 -e 7 -w_gen 0.75
# python run.py ESL -g 0 -mle -rl -l experiments/ESL-lr0.006-eps7-MLE1.0 -lr 0.0005 -bs 16 -e 5 -w_f1 0.5 -w_re 0.1 -w_mle 0.25

python run.py ESL -g 0 -mle -lr 0.006 -bs 32 -e 10 -w_gen 0.75
python run.py ESL -g 0 -mle -rl -l experiments/lastest_version-ESL-lr0.006-eps10-MLE1.0 -lr 0.0005 -bs 16 -e 10 -w_f1 0.5 -w_re 0.25 -w_mle 0.1

