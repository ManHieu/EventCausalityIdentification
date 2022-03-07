from os import name
from typing import Dict
from transformers import T5Tokenizer
from torch.utils.data import DataLoader 
import pytorch_lightning as pl
from arguments import DataTrainingArguments
from transformers import T5Tokenizer, AutoTokenizer, AutoModel
from .datasets import load_dataset
import copy

DATA_MODULES: Dict[str, pl.LightningDataModule] = {}


def register_data_module(data_module_class: pl.LightningDataModule):
    DATA_MODULES[data_module_class.name] = data_module_class
    return data_module_class


def load_data_module(module_name, data_name, data_args: DataTrainingArguments, batch_size: int = 8, fold_name=None) -> pl.LightningDataModule:
    """
    Load a registered data module.
    """
    return DATA_MODULES[module_name](
        data_args=data_args,
        batch_size=batch_size,
        data_name=data_name,
        fold_name=fold_name,
    )


@register_data_module
class EEREDataModule(pl.LightningDataModule):
    """
    Dataset processing for Event Event Relation Extraction.
    """
    SPECIAL_TOKENS = []
    name = 'ECI'

    def __init__(self, data_args: DataTrainingArguments, data_name, batch_size: int=8, fold_name=None):
        super().__init__()
        self.save_hyperparameters()
        self.data_name = data_name
        self.tokenizer = T5Tokenizer.from_pretrained(data_args.tokenizer)
        self.tokenizer.add_tokens(self.SPECIAL_TOKENS)

        self.tokenizer_for_generating = T5Tokenizer.from_pretrained(data_args.tokenizer)
        # when generating, we will use the logits of right-most token to predict the next token
        # so the padding should be on the left
        self.tokenizer_for_generating.padding_side = 'left'
        self.tokenizer_for_generating.pad_token = self.tokenizer_for_generating.eos_token # to avoid an error
        
        self.max_input_len = data_args.max_seq_length
        self.max_ouput_len = data_args.max_output_seq_length
        self.fold_name = fold_name
    
    def train_dataloader(self):
        dataset = load_dataset(
            dataset_name=self.data_name,
            data_args=self.hparams.data_args,
            tokenizer=self.tokenizer,
            tokenizer_for_generating=self.tokenizer_for_generating,
            max_input_length=self.max_input_len,
            max_output_length=self.max_ouput_len,
            split='train',
            data_name=self.fold_name
        )
        dataloader = DataLoader(
            dataset= dataset,
            batch_size= self.hparams.batch_size,
            shuffle=True,
            collate_fn=dataset.my_collate,
        )
        return dataloader
    
    def val_dataloader(self):
        dataset = load_dataset(
            dataset_name=self.data_name,
            data_args=self.hparams.data_args,
            tokenizer=self.tokenizer,
            tokenizer_for_generating=self.tokenizer_for_generating,
            max_input_length=self.max_input_len,
            max_output_length=self.max_ouput_len,
            split='test',
            data_name=self.fold_name
        )
        dataloader = DataLoader(
            dataset= dataset,
            batch_size= self.hparams.batch_size,
            shuffle=True,
            collate_fn=dataset.my_collate,
        )
        return dataloader
    
    def test_dataloader(self):
        dataset = load_dataset(
            dataset_name=self.data_name,
            data_args=self.hparams.data_args,
            tokenizer=self.tokenizer,
            tokenizer_for_generating=self.tokenizer_for_generating,
            max_input_length=self.max_input_len,
            max_output_length=self.max_ouput_len,
            split='dev',
            data_name=self.fold_name
        )
        dataloader = DataLoader(
            dataset= dataset,
            batch_size= self.hparams.batch_size,
            shuffle=False,
            collate_fn=dataset.my_collate,
        )
        return dataloader
