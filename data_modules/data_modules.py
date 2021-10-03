from typing import Dict
from transformers import T5Tokenizer
from torch.utils.data import DataLoader 
import pytorch_lightning as pl
from arguments import DataTrainingArguments
from transformers import T5Tokenizer
from .base_dataset import my_collate
from .datasets import load_dataset

DATA_MODULES: Dict[str, pl.LightningDataModule] = {}


def register_data_module(data_module_class: pl.LightningDataModule):
    DATA_MODULES[data_module_class.name] = data_module_class
    return data_module_class


def load_data_module(module_name, data_args: DataTrainingArguments, batch_size: int = 8) -> pl.LightningDataModule:
    """
    Load a registered data module.
    """
    return DATA_MODULES[module_name](
        data_args=data_args,
        batch_size=batch_size
    )


@register_data_module
class MatresDataModule(pl.LightningDataModule):
    """
    Dataset processing for MATRES.
    """
    SPECIAL_TOKENS = []
    name = 'MATRES'

    def __init__(self, data_args: DataTrainingArguments, batch_size: int=8):
        super().__init__()
        self.save_hyperparameters(data_args, batch_size)
        self.tokenizer = T5Tokenizer.from_pretrained(data_args.tokenizer)
        self.tokenizer.add_tokens(self.SPECIAL_TOKENS)
        self.max_input_len = data_args.max_seq_length
        self.max_ouput_len = data_args.max_output_seq_length
    
    def train_dataloader(self):
        dataset = load_dataset(
            dataset_name=self.name,
            data_args=self.hparams.data_args,
            tokenizer=self.tokenizer,
            max_input_length=self.max_input_len,
            max_output_length=self.max_ouput_len,
            split='train',
        )
        dataloader = DataLoader(
            dataset= dataset,
            batch_size= self.hparams.batch_size,
            shuffle=True,
            collate_fn=my_collate,
        )
        return dataloader
    
    def val_dataloader(self):
        dataset = load_dataset(
            dataset_name=self.name,
            data_args=self.hparams.data_args,
            tokenizer=self.tokenizer,
            max_input_length=self.max_input_len,
            max_output_length=self.max_ouput_len,
            split='dev',
        )
        dataloader = DataLoader(
            dataset= dataset,
            batch_size= self.hparams.batch_size,
            shuffle=True,
            collate_fn=my_collate,
        )
        return dataloader
    
    def train_dataloader(self):
        dataset = load_dataset(
            dataset_name=self.name,
            data_args=self.hparams.data_args,
            tokenizer=self.tokenizer,
            max_input_length=self.max_input_len,
            max_output_length=self.max_ouput_len,
            split='test',
        )
        dataloader = DataLoader(
            dataset= dataset,
            batch_size= self.hparams.batch_size,
            shuffle=True,
            collate_fn=my_collate,
        )
        return dataloader
