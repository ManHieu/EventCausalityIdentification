import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict
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
