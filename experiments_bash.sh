#!/bin/bash
echo "Nice to meet you"

# echo "\n MLE WITHOUT RECONSTRUCT TASK"
# python run.py ESL -g 1 -mle -lr 0.006 -bs 32 -e 7 -w_gen 1.0

# echo "RL ONLY"
# python run.py ESL -g 1 -rl -l experiments/0.558-ESL-lr0.006-eps7-MLE1.0 -lr 0.0005 -bs 16 -e 5 -w_f1 0.5 -w_re 0.1 -w_mle 0.

# echo "------------------------------------------------------------"
# echo "FULL WITHOUT RECONSTRUCT REWARD"
# python run.py ESL -g 1 -mle -rl -l experiments/0.558-ESL-lr0.006-eps7-MLE1.0 -lr 0.0005 -bs 16 -e 5 -w_f1 0.5 -w_re 0. -w_mle 0.25

# echo "------------------------------------------------------------"
# echo "FULL WITHOUT GENERATE REWARD"
# python run.py ESL -g 2 -mle -rl -l experiments/0.558-ESL-lr0.006-eps7-MLE1.0 -lr 0.0005 -bs 16 -e 5 -w_f1 0.5 -w_re 0.5 -w_mle 0.25

echo "------------------------------------------------------------"
echo "FULL WITHOUT F1 REWARD"
python run.py ESL -g 1 -mle -rl -l experiments/0.558-ESL-lr0.006-eps7-MLE1.0 -lr 0.0005 -bs 16 -e 5 -w_f1 0.0 -w_re 0.1 -w_mle 0.25