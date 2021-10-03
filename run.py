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

from model.model import GenEERModel


def objective(trial: optuna.trial):
    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job
    assert job in config

    defaults = {
        'learning_rate': 5e-4,
        'batch_size': 4,
        'warmup_ratio': 0.1,
    }
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
    model_args, data_args, training_args = second_parser.parse_args_into_dataclasses(remaining_args)

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
        f'-b{training_args.per_device_train_batch_size}')
    if data_args.output_format is not None:
        output_dir += f'-{data_args.output_format}'
    if data_args.input_format is not None:
        output_dir += f'-{data_args.input_format}'
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    
    seed_everything(args.seed)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        save_top_k=2,
        monitor='avg_val_loss',
        mode='min',
        save_weights_only=True,
        filename='{epoch}-{avg_val_loss:.2f}', # this cannot contain slashes 

    )
    lr_logger = LearningRateMonitor() 
    tb_logger = TensorBoardLogger('logs/')

    model = GenEERModel(model_args=model_args,
                        learning_rate=training_args.learning_rate,
                        adam_epsilon=training_args.adam_epsilon,
                        warmup=training_args.warmup_ratio)

    dm = load_data_module(module_name = args.job,
                        data_args=data_args,
                        batch_size=training_args.batch_size)

    trainer = Trainer(
        logger=tb_logger,
        min_epochs=training_args.num_train_epochs,
        max_epochs=training_args.num_train_epochs, 
        gpus=[args.gpu], 
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        gradient_clip_val=training_args.gradient_clip_val, 
        num_sanity_val_steps=1, 
        val_check_interval=0.5, # use float to check every n epochs 
        precision=16 if args.fp16 else 32,
        callbacks = [lr_logger, checkpoint_callback],
    )

    if args.trained_model:
        model.load_state_dict(torch.load(args.trained_model, map_location=model.device)['state_dict'])
    
    if args.eval: 
        dm.setup('test')
        trainer.test(model, datamodule=dm) #also loads training dataloader 
    else:
        dm.setup('fit')
        trainer.fit(model, dm) 
    
    return 0

if __name__ == '_main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('-c', '--config_file', type=str, default='config.ini', help='configuration file')
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='run evaluation only')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use')
    parser.add_argument('-l', '--trained_model', type=str, default=None, help='load trained model')

    args, remaining_args = parser.parse_known_args()

    sampler = optuna.samplers.TPESampler(seed=1741)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=5)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    