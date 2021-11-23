from .input_example import Entity, EntityType, InputExample, Relation, RelationType
import os
import json
import logging
import numpy as np
from typing import Dict, List
from .base_dataset import BaseDataset
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from arguments import DataTrainingArguments

DATASETS: Dict[str, BaseDataset] = {}


def register_dataset(dataset_class: BaseDataset):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(
                dataset_name: str,
                data_args: DataTrainingArguments,
                tokenizer: PreTrainedTokenizer,
                tokenizer_for_generating: PreTrainedTokenizer,
                max_input_length: int,
                max_output_length: int,
                seed: int = None,
                train_subset = 1,
                split = 'train',
                data_name = None,
            ) -> BaseDataset:
    """
    Load a registered dataset.
    """
    return DATASETS[dataset_name](
            tokenizer=tokenizer,
            tokenizer_for_generating=tokenizer_for_generating,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            seed=seed,
            data_args=data_args,
            train_subset=train_subset,
            split = split,
            data_name=data_name
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
        name = self.name
        file_path = os.path.join(self.data_dir(), f'{name}_intra_{split}.json')
        print(file_path)

        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} sentence-pairs for split {split} of {self.name}")

            for i, x in enumerate(data):
                triggers = [
                    Entity(id=j, type=self.event_types[y['type']], mention=y['mention'], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['triggers'])
                ]

                relations = []
                for y in x['relations']:
                    # print(y)
                    relations.append(Relation(
                            type=self.relation_types[y['type']], head=triggers[y['head']], tail=triggers[y['tail']]
                        ))
                    

                tokens = x['tokens']

                example = InputExample(
                        id=f'{split}-{i}',
                        tokens=tokens,
                        triggers=triggers,
                        relations=relations,
                        dep_path=x['dep_path']
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


@register_dataset
class ESLDataset(JointERDataset):

    name = "ESL"

    default_input_format = 'ECI_input'
    default_output_format = 'ECI_ouput'

    natural_event_types = {
        'action_occurrence': 'action occurrence',
        'action_state': 'action state',
        'action_aspectual': 'action aspectual',
        'action_reporting': 'action reporting',
        'action_perception': 'action perception',
        'action_causative': 'action causative',
        'action_generic': 'action generic',
        'neg_action_state': 'action state',
        'neg_action_occurrence': 'action occurrence',
        'neg_action_aspectual': 'action aspectual',
        'neg_action_reporting': 'action reporting',
        'neg_action_perception': 'action perception',
        'neg_action_causative': 'action causative',
        'neg_action_generic': 'action generic',
        }
    
    natural_relation_types = {
        'FALLING_ACTION': 'falling action', 
        'PRECONDITION': 'precondition', 
        }
