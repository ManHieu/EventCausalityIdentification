from abc import ABC, abstractmethod
import copy

from datasets.input_example import InputExample

INPUT_FORMATS = {}


def register_input_format(format_class):
    INPUT_FORMATS[format_class.name] = format_class
    return format_class


class BaseInputFormat(ABC):
    name = None

    BEGIN_ENTITY_TOKEN = '['
    END_ENTITY_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_SEPARATOR_TOKEN = '='
    QUERY_SEPARATOR_TOKEN = ':'

    def format_input(self, example: InputExample, multitask=False, task_descriptor=None):
        res = self._format_input(example=example)
        if multitask:
            name = task_descriptor or example.dataset.task_descriptor or example.dataset.name
            res = f'{name} {self.QUERY_SEPARATOR_TOKEN} ' + res
        return res
    
    @abstractmethod
    def _format_input(self, example: InputExample) -> str:
        raise NotImplementedError

@register_input_format
class PlainInputFormat(BaseInputFormat):
    """
    This format uses the plain sentence as input.
    """
    name = 'plain'

    def _format_input(self, example: InputExample) -> str:
        return ' '.join(example.tokens)
