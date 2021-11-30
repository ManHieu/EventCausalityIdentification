import argparse
import configparser
import itertools
import json
import logging
import os
from collections import defaultdict
import optuna
from pytorch_lightning.trainer.trainer import Trainer
import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from pytorch_lightning.utilities.seed import seed_everything
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data_modules.data_modules import load_data_module
from eval import eval_corpus
from model.model import GenEERModel


def objective(trial: optuna.Trial):
    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job
    assert job in config

    defaults = {
        'pretrain_lr': trial.suggest_categorical('pretrain_lr', [1e-5, 5e-5, 1e-4, 5e-4]),
        'reinforce_lr': trial.suggest_categorical('reinforce_lr', [1e-5, 5e-5, 1e-4, 5e-4]),
        'batch_size': trial.suggest_categorical('batch_size', [16]),
        'warmup_ratio': 0.1,
        'pretrain_epoches': trial.suggest_categorical('pretrain_epoches', [1, 3]),
        'reinforce_train_epoches': trial.suggest_categorical('reinforce_train_epoches', [3, 5, 7]),
        'margin': trial.suggest_categorical('margin', [1]),
        'generate_weight': trial.suggest_categorical('generate_weight', [1.0, 0.5, 0.9, 0.95]),
        'f1_weight': trial.suggest_categorical('f1_weight', [1.0, 0.5, 0.9, 0.95]),
    }
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

    if data_args.tokenizer == None:
        data_args.tokenizer = model_args.tokenizer_name
    
    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass
    # construct name for the output directory
    # for example: conll04-t5-base-ep200-len256-ratio0-b4-train
    output_dir = os.path.join(
        training_args.output_dir,
        f'{args.job}'
        f'-{model_args.model_name_or_path.split("/")[-1]}'
        f'-ep{round(training_args.num_train_epochs)}'
        f'-len{data_args.max_seq_length}'
        f'-lr{training_args.learning_rate}'
        f'-b{training_args.batch_size}')
    if data_args.output_format is not None:
        output_dir += f'-{data_args.output_format}'
    if data_args.input_format is not None:
        output_dir += f'-{data_args.input_format}'
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    
    seed_everything(training_args.seed)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        save_top_k=1,
        monitor='avg_val_loss',
        mode='min',
        save_weights_only=True,
        filename='{epoch}-{avg_val_loss:.2f}', # this cannot contain slashes 

    )
    lr_logger = LearningRateMonitor(logging_interval='step') 
    tb_logger = TensorBoardLogger('logs/')

    f1s = []
    ps = []
    rs = []
    for i in range(n_fold):
        print(f"TRAINING AND TESTING IN FOLD {i}: ")
        fold_dir = f'{data_dir}/{i}'
        dm = load_data_module(module_name = 'ECI',
                            data_args=data_args,
                            batch_size=training_args.batch_size,
                            data_name=args.job,
                            fold_name=fold_dir)
        
        number_step_in_epoch = len(dm.train_dataloader())/training_args.gradient_accumulation_steps

        model = GenEERModel(
                            tokenizer_name=model_args.tokenizer_name,
                            model_name_or_path=model_args.model_name_or_path,
                            input_format=data_args.input_format,
                            oupt_format=data_args.output_format,
                            max_input_len=data_args.max_seq_length,
                            max_oupt_len=data_args.max_output_seq_length,
                            generate_weight=training_args.generate_weight,
                            f1_weight=training_args.f1_weight,
                            margin=training_args.margin,
                            pretrain_step=int(number_step_in_epoch) * training_args.pretrain_epoches,
                            reinforce_step=int(number_step_in_epoch) * training_args.reinforce_train_epoches,
                            pretrain_lr=training_args.pretrain_lr,
                            reinforce_lr=training_args.reinforce_lr,
                            adam_epsilon=training_args.adam_epsilon,
                            weight_decay=training_args.weight_decay,
                            warmup=training_args.warmup_ratio,
        )

        trainer = Trainer(
            # logger=tb_logger,
            min_epochs=training_args.pretrain_epoches + training_args.reinforce_train_epoches,
            max_epochs=training_args.pretrain_epoches + training_args.reinforce_train_epoches, 
            gpus=[args.gpu], 
            accumulate_grad_batches=training_args.gradient_accumulation_steps,
            gradient_clip_val=training_args.gradient_clip_val, 
            num_sanity_val_steps=5, 
            val_check_interval=0.5, # use float to check every n epochs 
            callbacks = [lr_logger],
        )

        print("Training....")
        dm.setup('fit')
        trainer.fit(model, dm)

        print("Testing .....")
        dm.setup('test')
        trainer.test(model, dm)

        f1, p, r = eval_corpus()
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        print(f"RESULT IN FOLD {i}: ")
        print(f"F1: {f1}")
        print(f"P: {p}")
        print(f"R: {r}")
    
    f1 = sum(f1s)/len(f1s)
    p = sum(ps)/len(ps)
    r = sum(rs)/len(rs)
    print(f"F1: {f1} - P: {p} - R: {r}")
    if f1 > 0.4:
        with open('./results.txt', 'a', encoding='utf-8') as f:
            f.write(f"F1: {f1} \n")
            f.write(f"P: {p} \n")
            f.write(f"R: {r} \n")
            f.write(f"Hyperparams: \n {defaults}\n")
            f.write(f"{'--'*10} \n")

    return f1


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('-c', '--config_file', type=str, default='config.ini', help='configuration file')
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='run evaluation only')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use')
    parser.add_argument('-l', '--trained_model', type=str, default=None, help='load trained model')

    args, remaining_args = parser.parse_known_args()

    sampler = optuna.samplers.TPESampler(seed=1741)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=75)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    