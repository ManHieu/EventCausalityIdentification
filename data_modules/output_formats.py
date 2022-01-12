import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict
from data_modules.templates import TEMPLATES
from utils.utils import get_span
import numpy as np 

from .input_example import EntityType, InputExample, RelationType


class BaseOutputFormat(ABC):
    name = None
    
    @abstractmethod
    def format_output(self, example: InputExample, template_type: int=0) -> str:
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
    templates: List[Tuple[str, str]] = TEMPLATES['eci']

    def format_output(self, example: InputExample, template_type: int) -> str:
        template = self.templates[template_type][1]
        
        rels = defaultdict(list)
        for relation in example.relations:
            if relation.type.natural == 'falling action':
                # ev A falling action ev B = ev B causes ev A
                rels[relation.tail].append(relation.head)
            else:
                rels[relation.head].append(relation.tail)

        sents = []
        for head, tails in rels.items():
            head_mention = head.mention
            tail_mentions = [f'{tail.mention}' for tail in tails]

            sent = f'{head_mention} causes {" and ".join(tail_mentions)}'
            sents.append(sent)
        
        paths = []
        for path in example.dep_path:
            paths.append(', '.join(path))
        dep_path = f"{'; '.join(paths)}."
        
        # print("Output: {}".format(sent_out))
        # print(f"{sent_out}. {dep_path}")
        
        if len(sents) == 0:
            sent_out = "none"
            oupt = template.format(answer=f"No. Because {dep_path}", conclusion='None')
        else:
            sent_out = ' '.join(sents)
            oupt = template.format(answer=f"Yes. Because {sent_out} and {dep_path}", conclusion=sent_out)
        
        return oupt
        
        # return f"{sent_out}. {dep_path}"