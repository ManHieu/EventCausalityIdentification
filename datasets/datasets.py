from collections import Counter
import os
import json
import logging
import numpy as np
from typing import Dict, List
from utils.utils import get_precision_recall_f1
from datasets.input_example import Entity, EntityType, InputExample, Relation, RelationType
from arguments import DataTrainingArguments
from datasets.base_dataset import BaseDataset
import torch
from transformers import PreTrainedTokenizer

DATASETS = {}


def register_dataset(dataset_class: BaseDataset):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(
        dataset_name: str,
        data_args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        split: str,
        max_input_length: int,
        max_output_length: int,
        train_subset: float = 1,
        seed: int = None,
        shuffle: bool = True,
        is_eval: bool = False
):
    """
    Load a registered dataset.
    """
    return DATASETS[dataset_name](
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        mode=split,
        overwrite_cache=data_args.overwrite_cache,
        train_subset=train_subset,
        seed=seed,
        shuffle=shuffle,
        data_args=data_args,
        is_eval=is_eval,
    )


class JointERDataset(BaseDataset):
    """
    Base class for datasets of joint event and relation extraction.
    """
    event_types = None
    relation_types = None

    natural_event_types = None     # dictionary from entity types given in the dataset to the natural strings to use
    natural_relation_types = None   # dictionary from relation types given in the dataset to the natural strings to use

    default_output_format = 'joint_er'

    def load_cached_data(self, cached_features_file):
        d = torch.load(cached_features_file)
        self.event_types, self.relation_types, self.examples, self.features = \
            d['event_types'], d['relation_types'], d['examples'], d['features']

    def save_data(self, cached_features_file):
        torch.save({
            'event_types': self.event_types,
            'relation_types': self.relation_types,
            'examples': self.examples,
            'features': self.features,
        }, cached_features_file)
    
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
    
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).

        This is the default implementation for datasets in the SpERT format
        (see https://github.com/markus-eberts/spert).
        """
        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentence-pairs for split {split} of {self.name}")

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

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None) -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """
        # extract triggers and relations from output sentence
        res = self.output_format.run_inference(
            example,
            output_sentence,
            event_types=self.event_types,
            relation_types=self.relation_types,
        )
        predicted_triggers, predicted_relations = res[:2]
        if len(res) == 6:
            # the output format provides information about errors
            wrong_reconstruction, label_error, trigger_error, format_error = res[2:]
        else:
            # in case the output format does not provide information about errors
            wrong_reconstruction = label_error = trigger_error = format_error = False

        predicted_triggers_no_type = set([entity[1:] for entity in predicted_triggers])

        # load ground truth entities
        gt_triggers = set(trigger.to_tuple() for trigger in example.triggers)
        gt_triggers_no_type = set([trigger[1:] for trigger in gt_triggers])

        # compute correct entities
        correct_triggers = predicted_triggers & gt_triggers
        correct_triggers_no_type = gt_triggers_no_type & predicted_triggers_no_type

        # load ground truth relations
        gt_relations = set(relation.to_tuple() for relation in example.relations)

        # compute correct relations
        correct_relations = predicted_relations & gt_relations

        assert len(correct_triggers) <= len(predicted_triggers)
        assert len(correct_triggers) <= len(gt_triggers)
        assert len(correct_triggers_no_type) <= len(predicted_triggers_no_type)
        assert len(correct_triggers_no_type) <= len(gt_triggers_no_type)

        assert len(correct_relations) <= len(predicted_relations)
        assert len(correct_relations) <= len(gt_relations)

        res = Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'label_error': 1 if label_error else 0,
            'trigger_error': 1 if trigger_error else 0,
            'format_error': 1 if format_error else 0,
            'gt_triggers': len(gt_triggers),
            'predicted_triggers': len(predicted_triggers),
            'correct_triggers': len(correct_triggers),
            'gt_triggers_no_type': len(gt_triggers_no_type),
            'predicted_triggers_no_type': len(predicted_triggers_no_type),
            'correct_triggers_no_type': len(correct_triggers_no_type),
            'gt_relations': len(gt_relations),
            'predicted_relations': len(predicted_relations),
            'correct_relations': len(correct_relations),
        })

        # add information about each entity/relation type so that we can compute the macro-F1 scores
        if self.event_types is not None:
            for event_type in self.event_types.values():
                predicted = set(trigger for trigger in predicted_triggers if trigger[0] == event_type.natural)
                gt = set(trigger for trigger in gt_triggers if trigger[0] == event_type.natural)
                correct = predicted & gt
                res['predicted_triggers', event_type.natural] = len(predicted)
                res['gt_triggers', event_type.natural] = len(gt)
                res['correct_triggers', event_type.natural] = len(correct)

        if self.relation_types is not None:
            for relation_type in self.relation_types.values():
                predicted = set(relation for relation in predicted_relations if relation[0] == relation_type.natural)
                gt = set(relation for relation in gt_relations if relation[0] == relation_type.natural)
                correct = predicted & gt
                res['predicted_relations', relation_type.natural] = len(predicted)
                res['gt_relations', relation_type.natural] = len(gt)
                res['correct_relations', relation_type.natural] = len(correct)

        return res
    
    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=self.tokenizer,
                )
            results += new_result

        trigger_precision, trigger_recall, trigger_f1 = get_precision_recall_f1(
            num_correct=results['correct_triggers'],
            num_predicted=results['predicted_triggers'],
            num_gt=results['gt_triggers'],
        )

        trigger_precision_no_type, trigger_recall_no_type, trigger_f1_no_type = get_precision_recall_f1(
            num_correct=results['correct_triggers_no_type'],
            num_predicted=results['predicted_triggers_no_type'],
            num_gt=results['gt_triggers_no_type'],
        )

        trigger_precision_by_type = []
        trigger_recall_by_type = []
        trigger_f1_by_type = []

        if macro:
            # compute also entity macro scores
            for event_type in self.event_types.values():
                precision, recall, f1 = get_precision_recall_f1(
                    num_correct=results['correct_triggers', event_type.natural],
                    num_predicted=results['predicted_triggers', event_type.natural],
                    num_gt=results['gt_triggers', event_type.natural],
                )
                trigger_precision_by_type.append(precision)
                trigger_recall_by_type.append(recall)
                trigger_f1_by_type.append(f1)

        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )

        res = {
            'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'],
            'label_error': results['label_error'] / results['num_sentences'],
            'trigger_error': results['trigger_error'] / results['num_sentences'],
            'format_error': results['format_error'] / results['num_sentences'],
            'trigger_precision': trigger_precision,
            'trigger_recall': trigger_recall,
            'trigger_f1': trigger_f1,
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
            'trigger_precision_no_type': trigger_precision_no_type,
            'trigger_recall_no_type': trigger_recall_no_type,
            'trigger_f1_no_type': trigger_f1_no_type,
        }

        if macro:
            res.update({
                'trigger_macro_precision': np.mean(np.array(trigger_precision_by_type)),
                'trigger_macro_recall': np.mean(np.array(trigger_recall_by_type)),
                'trigger_macro_f1': np.mean(np.array(trigger_f1_by_type)),
            })

        return res


@register_dataset
class MATRESDataset(JointERDataset):
    """
    MATRES dataset (event detection and temporal event extraction)
    """
    name = 'MATRES'

    natural_event_types = {}
    natural_relation_types = {}

