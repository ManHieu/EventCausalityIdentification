from arguments import DataTrainingArguments
from .input_formats import INPUT_FORMATS, BaseInputFormat
from .output_formats import OUTPUT_FORMATS, BaseOutputFormat
from typing import Dict, Generator, List, Tuple
from .input_example import InputExample, InputFeatures
import os 
import logging
import random
from abc import ABC, abstractmethod
import torch
from torch.utils.data.dataset import Dataset, T_co
from tqdm import tqdm
from transformers import PreTrainedTokenizer, torch_distributed_zero_first, default_data_collator


def my_collate(batch: List[InputFeatures]):
    '''
    'input_token_ids':input_tokens['input_ids'],
    'input_attn_mask': input_tokens['attention_mask'],
    'tgt_token_ids': tgt_tokens['input_ids'],
    'tgt_attn_mask': tgt_tokens['attention_mask'],
    '''
    input_token_ids = torch.stack([torch.LongTensor(input_feature.input_ids) for input_feature in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(input_feature.input_attention_mask) for input_feature in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(input_feature.label_ids) for input_feature in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(input_feature.tgt_attention_mask) for input_feature in batch])

    return {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
    }


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None         # name of the dataset
    data_name = None    # name of the directory, if different from the name of the dataset
    task_descriptor = None  # string to prepend to every input sentence if multitask=True (default is self.name)

    default_input_format = 'plain'
    default_output_format = None
    default_data_dir = 'datasets'

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_input_length: int,
            max_output_length: int,
            seed: int = None,
            data_args: DataTrainingArguments = None,
            train_subset = 1,
            split = 'train'
        ) -> None:
        super().__init__()
        if seed is not None:
            # set random seed for repeatability
            random.seed(seed)
            torch.manual_seed(seed)
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.input_format: BaseInputFormat = INPUT_FORMATS[
            data_args.input_format if data_args.input_format is not None else self.default_input_format
        ]()
        self.output_format: BaseOutputFormat = OUTPUT_FORMATS[
            data_args.output_format if data_args.output_format is not None else self.default_output_format
        ]()

        self.data_path = data_args.data_dir if data_args.data_dir is not None else self.default_data_dir

        self.load_schema()

        self.examples: List[InputExample] = self.load_data(split=split)
        for example in self.examples:
            example.dataset = self

        self.features: List[InputFeatures] = self.compute_features()

        # compute effective size of the dataset
        self.effective_size = round(train_subset * len(self.examples))
        if train_subset != 1:
            logging.info(f"Effective dataset size reduced to {self.effective_size} ({train_subset * 100:.0f}%)")
    
    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(self, index) -> InputFeatures:
        return self.features[index]

    def data_dir(self):
        if self.data_name is not None:
            return os.path.join(self.data_path, self.data_name)
        else:
            return os.path.join(self.data_path, self.name)
    
    @abstractmethod
    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def _warn_max_sequence_length(self, max_sequence_length: int, sentences: List[str], name: str):
        max_length_needed = max(len(self.tokenizer.tokenize(x)) for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f'Max sequence length is {max_sequence_length} but the longest {name} sequence is '
                f'{max_length_needed} long'
            )
    
    def compute_features(self, multitask: bool = False):
        input_sentences = []
        output_sentences = []
        input_features = []

        for example in self.examples:
            input_sentence  = self.input_format.format_input(example, multitask=multitask)
            output_sentence = self.output_format.format_output(example)

            input_tokens = self.tokenizer.encode_plus(input_sentence, 
                                add_special_tokens=True,
                                max_length=self.max_input_length,
                                truncation=True,
                                padding='max_length')
            output_tokens = self.tokenizer.encode_plus(output_sentence, 
                                add_special_tokens=True,
                                max_length=self.max_output_length,
                                truncation=True,
                                padding='max_length')

            input_features.append(
                InputFeatures(
                    input_ids= input_tokens['input_ids'],
                    input_attention_mask= input_tokens['attention_mask'],
                    label_ids=output_tokens['input_ids'],
                    tgt_attention_mask=output_tokens['attention_mask']
                )
            )
            input_sentences.append(input_sentence)
            output_sentences.append(output_sentence)
        
        self._warn_max_sequence_length(self.max_input_length, input_sentences, "input")
        self._warn_max_sequence_length(self.max_output_length, output_sentences, "output")
        
        return input_features
