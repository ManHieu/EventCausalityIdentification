from abc import ABC, abstractmethod
from .input_example import InputExample
from typing import Dict
from utils.utils import get_span


class BaseInputFormat(ABC):
    name = None

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
    
    def _format_input(self, example: InputExample) -> str:
        context = ' '.join(example.tokens)
        ED_template = "\n Event triggers are "
        triggers = [f"'{get_span(example.tokens, [trigger.start, trigger.end])}'" for trigger in example.triggers]
        ED_template = ED_template + ', '.join(triggers)
        template = f"\n Causual relation between {' and '.join(triggers)} is "
        # "something cause something"
        options = f"\n OPTIONS: \n - {triggers[0]} causes {triggers[1]} \n - {triggers[1]} causes {triggers[0]} \n - none relation"
        

        return context + '</s>' + ED_template + '</s>' + template + '</s>' + options