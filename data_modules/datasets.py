from .input_example import Entity, EntityType, InputExample, Relation, RelationType
import os
import json
import logging
import numpy as np
from typing import Dict, List
from .base_dataset import BaseDataset
import torch
from transformers import PreTrainedTokenizer
from arguments import DataTrainingArguments

DATASETS: Dict[str, BaseDataset] = {}


def register_dataset(dataset_class: BaseDataset):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(
                dataset_name: str,
                data_args: DataTrainingArguments,
                tokenizer: PreTrainedTokenizer,
                max_input_length: int,
                max_output_length: int,
                seed: int = None,
                train_subset = 1,
                split = 'train',
            ) -> BaseDataset:
    """
    Load a registered dataset.
    """
    return DATASETS[dataset_name](
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            seed=seed,
            data_args=data_args,
            train_subset=train_subset,
            split = split
            )


class JointERDataset(BaseDataset):
    event_types = None
    relation_types = None

    natural_event_types = None     # dictionary from entity types given in the dataset to the natural strings to use
    natural_relation_types = None   # dictionary from relation types given in the dataset to the natural strings to use

    default_output_format = None

    def load_schema(self):
        """
        Load event and relation types.

        This is the default implementation which uses the dictionaries natural_event_types and natural_relation_types.
        """
        if self.natural_event_types is not None:
            self.event_types = {short: EntityType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_event_types.items()}

        if self.natural_relation_types is not None:
            self.relation_types = {short: RelationType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_relation_types.items()}
    
    def load_data(self, split:str, seed: int = None) -> List[InputExample]:
        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_easy_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} sentence-pairs for split {split} of {self.name}")

            for i, x in enumerate(data):
                triggers = [
                    Entity(id=j, type=self.event_types[y['type']], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['triggers'])
                ]

                relations = [
                    Relation(
                        type=self.relation_types[y['type']], head=triggers[y['head']], tail=triggers[y['tail']]
                    )
                    for y in x['relations']
                ]

                tokens = x['tokens']

                example = InputExample(
                        id=f'{split}-{i}',
                        tokens=tokens,
                        triggers=triggers,
                        relations=relations,
                    )
                examples.append(example)

        return examples


@register_dataset
class MatresDataset(JointERDataset):
    """
    MATRES dataset (event detection and temporal event extraction)
    """
    name = "MATRES"

    default_input_format = 'plain'
    default_output_format = 'eere_output'

    natural_event_types = {
        'OCCURRENCE': 'occurrence', 
        'REPORTING': 'reporting', 
        'I_ACTION': 'self-action', 
        'STATE': 'state', 
        'I_STATE': 'self-state', 
        'ASPECTUAL': 'aspectual', 
        'PERCEPTION': 'perception'
        }
    natural_relation_types = {
        0: 'before', 
        1: 'after', 
        2: 'simultaneous', 
        3: 'vague'
        }

