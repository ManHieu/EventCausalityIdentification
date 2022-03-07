from arguments import DataTrainingArguments
from .input_formats import INPUT_FORMATS, BaseInputFormat
from .output_formats import OUTPUT_FORMATS, BaseOutputFormat
from typing import List, Tuple
from .input_example import InputExample, ProcessedInputExample
import os 
import logging
import random
from abc import ABC, abstractmethod
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer


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
            tokenizer_for_generating: PreTrainedTokenizer,
            max_input_length: int,
            max_output_length: int,
            seed: int = None,
            data_args: DataTrainingArguments = None,
            train_subset = 1,
            split = 'train',
            data_name = None,
        ) -> None:
        super().__init__()
        if seed is not None:
            # set random seed for repeatability
            random.seed(seed)
            torch.manual_seed(seed)
        
        self.tokenizer = tokenizer
        self.tokenizer_for_generating = tokenizer_for_generating
        self.data_args = data_args

        self.data_name = data_name
        
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

        self.features: List[ProcessedInputExample] = self.compute_features()

        # compute effective size of the dataset
        self.effective_size = round(train_subset * len(self.examples))
        if train_subset != 1:
            logging.info(f"Effective dataset size reduced to {self.effective_size} ({train_subset * 100:.0f}%)")
    
    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(self, index) -> ProcessedInputExample:
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
        # input_sentences = []
        # output_sentences = []
        # input_features = []

        # for example in self.examples:
        #     context_sentence, ED_template  = self.input_format.format_input(example, multitask=multitask)
        #     # print(input_sentence)
        #     output_sentence = self.output_format.format_output(example)
        #     # print(output_sentence)
            
        #     input_features.append(
        #         ProcessedInputExample(
        #             context_sentence=context_sentence,
        #             ED_template=ED_template,
        #             output_sentence=output_sentence,
        #         )
        #     )
        #     input_sentences.append(context_sentence + '\n' + ED_template)
        #     output_sentences.append(output_sentence)

        # self._warn_max_sequence_length(self.max_input_length, input_sentences, "input")
        # self._warn_max_sequence_length(self.max_output_length, output_sentences, "output")
        input_features = self.examples
        
        return input_features
    
    def my_collate(self, batch: List[InputExample]):
        # input_sentences = []
        # output_sentences = []
        # context_sentences = []
        # ED_templates = []
        # for example in batch:
        #     input_sentences.append(example.context_sentence + '\n' + example.ED_template)
        #     output_sentences.append(example.output_sentence)
        #     context_sentences.append(example.context_sentence)
        #     ED_templates.append(example.ED_template)

        # output_sentence_encoding = self.tokenizer_for_generating(output_sentences,
        #                                             padding='longest',
        #                                             max_length=self.data_args.max_output_seq_length,
        #                                             truncation=True,
        #                                             return_tensors="pt")
        # label_token_ids = output_sentence_encoding.input_ids
        # label_token_ids[label_token_ids[:, :] == self.tokenizer_for_generating.pad_token_id] = -100 # replace padding token id's of the labels by -100
            
        # return (input_sentences, output_sentences, context_sentences, ED_templates, label_token_ids)
        return batch
