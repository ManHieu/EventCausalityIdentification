import argparse
import configparser
import itertools
import json
import logging
import os
from collections import defaultdict
from typing import Dict
import optuna
from pytorch_lightning.trainer.trainer import Trainer
import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from pytorch_lightning.utilities.seed import seed_everything
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data_modules.data_modules import load_data_module
from eval import eval_corpus
from model.model import GenEC


def run(defaults: Dict):
    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job
    assert job in config

    print("Hyperparams: {}".format(defaults))
    defaults.update(dict(config.items(job)))
    for key in defaults:
        if defaults[key] in ['True', 'False']:
            defaults[key] = True if defaults[key]=='True' else False
        if defaults[key] == 'None':
            defaults[key] = None
    
    # parse remaining arguments and divide them into three categories
    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    second_parser.set_defaults(**defaults)

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    # print(second_parser.parse_args_into_dataclasses(remaining_args))
    model_args, data_args, training_args = second_parser.parse_args_into_dataclasses(remaining_args)
    if job == 'ESL':
        data_args.input_format = 'ECI_input'
        data_args.output_format = 'ECI_ouput'
        n_fold = 5
        data_dir = 'ESL'
    elif job == 'Causal-TB':
        data_args.input_format = 'ECI_input'
        data_args.output_format = 'ECI_ouput'
        n_fold = 10
        data_dir = 'Causal-TB'

    if data_args.tokenizer == None:
        data_args.tokenizer = model_args.tokenizer_name
    
    record_file_name = './result.txt'
    if args.tuning:
        training_args.output_dir = './tuning_experiments'
        record_file_name = './tuning_result.txt'
        n_fold = 1
    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass
    
    seed_everything(training_args.seed, workers=True)
    # tb_logger = TensorBoardLogger('logs/')
    f1s = []
    ps = []
    rs = []
    for i in range(n_fold):
        print(f"TRAINING AND TESTING IN FOLD {i}: ")
        fold_dir = f'{data_dir}/{i}'
        if args.rl == True:
            model_args.model_name_or_path = f"{args.trained_model}/fold{i}"
        dm = load_data_module(module_name = 'ECI',
                            data_args=data_args,
                            batch_size=training_args.batch_size,
                            data_name=args.job,
                            fold_name=fold_dir)
        
        number_step_in_epoch = len(dm.train_dataloader())/training_args.gradient_accumulation_steps

        # construct name for the output directory
        output_dir = os.path.join(
            training_args.output_dir,
            f'{args.job}'
            f'-lr{training_args.lr}'
            f'-eps{training_args.num_epoches}')
        if args.mle==True:
            output_dir += f'-MLE{defaults["mle_weight"]}'
        if args.rl==True:
            output_dir += f'-RL-w_f1{defaults["f1_reward_weight"]}-w_re{defaults["reconstruct_reward_weight"]}'
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        output_dir = os.path.join(output_dir,f'fold{i}')
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass

        checkpoint_callback = ModelCheckpoint(
                                    dirpath=output_dir,
                                    save_top_k=1,
                                    monitor='f1_dev',
                                    mode='max',
                                    save_weights_only=True,
                                    filename='{epoch}-{f1_dev:.2f}', # this cannot contain slashes 
                                    )
        lr_logger = LearningRateMonitor(logging_interval='step') 
        model = GenEC(
                    tokenizer_name=model_args.tokenizer_name,
                    model_name_or_path=model_args.model_name_or_path,
                    input_type=data_args.input_format,
                    output_type=data_args.output_format,
                    max_input_len=data_args.max_seq_length,
                    max_oupt_len=data_args.max_output_seq_length,
                    mle_train=args.mle,
                    rl_train=args.rl,
                    num_training_step=int(number_step_in_epoch) * training_args.num_epoches,
                    lr=training_args.lr,
                    warmup=training_args.warmup_ratio,
                    adam_epsilon=training_args.adam_epsilon,
                    weight_decay=training_args.weight_decay,
                    generate_weight=training_args.generate_weight,
                    f1_reward_weight=training_args.f1_reward_weight,
                    reconstruct_reward_weight=training_args.reconstruct_reward_weight,
                    mle_weight=training_args.mle_weight,
                )
        trainer = Trainer(
            # logger=tb_logger,
            min_epochs=training_args.num_epoches,
            max_epochs=training_args.num_epoches, 
            gpus=[args.gpu], 
            accumulate_grad_batches=training_args.gradient_accumulation_steps,
            gradient_clip_val=training_args.gradient_clip_val, 
            num_sanity_val_steps=5, 
            val_check_interval=1.0, # use float to check every n epochs 
            callbacks = [lr_logger, checkpoint_callback],
        )

        print("Training....")
        dm.setup('fit')
        trainer.fit(model, dm)

        best_model = GenEC.load_from_checkpoint(checkpoint_callback.best_model_path)
        print("Testing .....")
        dm.setup('test')
        trainer.test(best_model, dm)

        if args.mle==True and args.rl==False:
            best_model.t5.save_pretrained(output_dir)

        f1, p, r = eval_corpus()
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        print(f"RESULT IN FOLD {i}: ")
        print(f"F1: {f1}")
        print(f"P: {p}")
        print(f"R: {r}")
        with open(output_dir+f'{f1}', 'w', encoding='utf-8') as f:
            f.write(f"F1: {f1} \n")
            f.write(f"P: {p} \n")
            f.write(f"R: {r} \n")
    
    f1 = sum(f1s)/len(f1s)
    p = sum(ps)/len(ps)
    r = sum(rs)/len(rs)
    print(f"F1: {f1} - P: {p} - R: {r}")
    if f1 > 0.55:
        with open(record_file_name, 'a', encoding='utf-8') as f:
            f.write(f"{'--'*10} \n")
            f.write(f"Hyperparams: \n {defaults}\n")
            f.write(f"F1: {f1} \n")
            f.write(f"P: {p} \n")
            f.write(f"R: {r} \n")
    
    return f1

def objective(trial: optuna.Trial):
    defaults = {
        'lr': trial.suggest_categorical('pretrain_lr', [1e-4, 4e-4, 6e-4, 1e-3]),
        'batch_size': trial.suggest_categorical('batch_size', [32]),
        'warmup_ratio': 0.1,
        'num_epoches': trial.suggest_categorical('num_epoches', [5,  10, 15, 20]),
    }   
    if args.rl==True:
        defaults['f1_reward_weight'] = trial.suggest_categorical('f1_reward_weight', [0.5, 0.75])
        defaults['reconstruct_reward_weight'] = trial.suggest_categorical('reconstruct_reward_weight', [0.1, 0.25])
    
    if args.mle==True and args.rl==False:
        defaults['mle_weight'] = 1.0
        defaults['generate_weight'] = trial.suggest_categorical('generate_weight', [0.5, 0.75, 0.9])
    if args.mle==False and args.rl==True:
        defaults['mle_weight'] = 0.0
    if args.mle==True and args.rl==True:
        defaults['generate_weight'] = 1.0
        defaults['mle_weight'] = trial.suggest_categorical('mle_weight', [0.1, 0.25, 0.5, 0.75])
    
    f1 = run(defaults=defaults)

    return f1


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('-c', '--config_file', type=str, default='config.ini', help='configuration file')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use')
    parser.add_argument('-mle', '--mle', action='store_true', default=True, help='training use mle loss')
    parser.add_argument('-rl', '--rl', action='store_true', default=False, help='training use rl loss')
    parser.add_argument('-l', '--trained_model', type=str, default=None, help='load trained model')
    parser.add_argument('-t', '--tuning', action='store_true', default=False, help='tune hyperparameters')
    parser.add_argument('-lr', '--lr', type=float, default=0.006, help='learning rate')
    parser.add_argument('-bs', '--bs', type=int, default=16, help='batch size')
    parser.add_argument('-e', '--epoches', type=int, default=5, help='number epoches')
    parser.add_argument('-w_gen', '--w_gen', type=float, default=0.5, help='weight of generate loss')
    parser.add_argument('-w_mle', '--w_mle', type=float, default=1.0, help='weight of mle')
    parser.add_argument('-w_f1', '--w_f1', type=float, default=0.5, help='weight of f1 reward')
    parser.add_argument('-w_re', '--w_re', type=float, default=0.25, help='weight of reconstruct reward')

    args, remaining_args = parser.parse_known_args()
    if args.rl:
        assert args.trained_model != None
    
    if args.tuning:
        print("tuning ......")
        sampler = optuna.samplers.TPESampler(seed=1741)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=25)
        trial = study.best_trial
        print('Accuracy: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))
    else:
        defaults = {
            'warmup_ratio': 0.1,
        }

        defaults['lr'] = args.lr
        defaults['batch_size'] = args.bs
        defaults['num_epoches'] = args.epoches
        
        if args.rl==True:
            defaults['f1_reward_weight'] = args.w_f1
            defaults['reconstruct_reward_weight'] = args.w_re

        if args.mle==True and args.rl==False:
            defaults['mle_weight'] = 1.0
            defaults['generate_weight'] = args.w_gen
        if args.mle==False and args.rl==True:
            defaults['mle_weight'] = 0.0
        if args.mle==True and args.rl==True:
            defaults['generate_weight'] = 1.0
            defaults['mle_weight'] = args.w_mle
        
        run(defaults=defaults)


    