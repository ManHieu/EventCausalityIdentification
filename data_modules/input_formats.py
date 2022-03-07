from abc import ABC, abstractmethod
from collections import defaultdict
# from data_modules.templates import TEMPLATES
from .input_example import InputExample
from typing import Dict, List, Tuple
from utils.utils import get_span


class BaseInputFormat(ABC):
    name = None

    QUERY_SEPARATOR_TOKEN = ':'

    def format_input(self, example: InputExample, task_descriptor: str=''):
        res = self._format_input(example=example)
        return res
    
    @abstractmethod
    def _format_input(self, example: InputExample, task_prefix: str='') -> str:
        raise NotImplementedError


INPUT_FORMATS : Dict[str, BaseInputFormat] = {}


def register_input_format(format_class: BaseInputFormat):
    INPUT_FORMATS[format_class.name] = format_class
    return format_class


@register_input_format
class PlainInputFormat(BaseInputFormat):
    """
    This format uses the plain sentence as input.
    """
    name = 'plain'

    def _format_input(self, example: InputExample) -> str:
        return ' '.join(example.tokens)


@register_input_format
class IdentifyCausalRelationInputFormat(BaseInputFormat):
    """
    The input format used for ECI task
    """
    name = 'ECI_input'
    
    # templates: List[Tuple[str, str]] = TEMPLATES['eci']
    
    def _format_input(self, example: InputExample):
        context = ' '.join(example.tokens)
        triggers = [trigger.mention for trigger in example.triggers]

        template = "{context}\n\nIs there a causal relation between {head} and {tail}?"
        template = template.format(
                                context=context,
                                head=triggers[0],
                                tail=triggers[1])
        # print(template)
        return context, triggers, f"{template}"