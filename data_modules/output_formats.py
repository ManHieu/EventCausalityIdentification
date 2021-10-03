import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict
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

    @abstractmethod
    def run_inference(self, example: InputExample, output_sentence: str):
        """
        Process an output sentence to extract whatever information the task asks for.
        """
        raise NotImplementedError


OUTPUT_FORMATS: Dict(str, BaseOutputFormat) = {}


def register_output_format(format_class: BaseOutputFormat):
    OUTPUT_FORMATS[format_class.name] = format_class
    return format_class

