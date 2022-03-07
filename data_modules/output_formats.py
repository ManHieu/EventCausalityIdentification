import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict
# from data_modules.templates import TEMPLATES
from utils.utils import get_span
import numpy as np 

from .input_example import EntityType, InputExample, RelationType


class BaseOutputFormat(ABC):
    name = None
    
    @abstractmethod
    def format_output(self, example: InputExample) -> str:
        """
        Format output for feeding into the model.
        """
        raise NotImplementedError


OUTPUT_FORMATS : Dict[str, BaseOutputFormat] = {}


def register_output_format(format_class: BaseOutputFormat):
    OUTPUT_FORMATS[format_class.name] = format_class
    return format_class


@register_output_format
class EEREOutputFormat(BaseOutputFormat):
    """
    Output format for event event relation classification.
    """
    name = 'eere_output'

    def format_output(self, example: InputExample) -> str:
        assert len(example.relations) == 1
        ev1_span = [example.relations[0].head.start, example.relations[0].head.end]
        ev2_span = [example.relations[0].tail.start, example.relations[0].tail.end]
        words = example.tokens
        rel = example.relations[0].type.natural

        return f"{get_span(words, ev1_span)} {rel} {get_span(words, ev2_span)}"


@register_output_format
class ClassOnlyOutputFormat(BaseOutputFormat):
    """
    Only class in output format.
    """
    name = 'class_only_output'
    
    def format_output(self, example: InputExample) -> str:
        return example.relations[0].type.natural


@register_output_format
class IdentifyCausalRelationOutputFormat(BaseOutputFormat):
    """
    Output format uses in ECI task
    """
    name = 'ECI_ouput'
    # templates: List[Tuple[str, str]] = TEMPLATES['eci']

    def format_output(self, example: InputExample) -> str:
        template = "{answer}"
        
        rels = defaultdict(list)
        for relation in example.relations:
            if relation.type.natural == 'falling action':
                rels[relation.tail].append(relation.head)
            else:
                rels[relation.head].append(relation.tail)

        sents = []
        for head, tails in rels.items():
            head_mention = head.mention
            tail_mentions = [f'{tail.mention}' for tail in tails]

            sent = f'{head_mention} causes {" and ".join(tail_mentions)}'
            sents.append(sent)
            
        dep_path = ''
        if len(example.dep_path) == 1:
            dep_path = ', '.join(example.dep_path[0])
        elif len(example.dep_path) > 1:
            dep_path = ', '.join(list(reversed(example.dep_path[0]))[:-1]) + ', ' + ', '.join(example.dep_path[1])     
        
        if len(sents) == 0:
            sent_out = "none"
            oupt = template.format(answer=f"No. {dep_path}", conclusion='None')
        else:
            sent_out = ' '.join(sents)
            oupt = template.format(answer=f"Yes. {sent_out} and {dep_path}", conclusion=sent_out)
        
        # print(oupt)
        
        return oupt