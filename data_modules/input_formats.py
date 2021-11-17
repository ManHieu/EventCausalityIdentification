from abc import ABC, abstractmethod
from .input_example import InputExample
from typing import Dict, Tuple
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
    
    def _format_input(self, example: InputExample) -> Tuple[str, str]:
        context = ' '.join(example.tokens)
        ED_template = "\n Event triggers are "
        triggers = [trigger.mention for trigger in example.triggers]
        ED_template = ED_template + ', '.join(triggers)

        return context, ED_template
    
    def format_input_for_selector(self, ctx: str, task_prefix: str) -> str:
        return f"{task_prefix} context: {ctx}"
    
    def format_input_for_predictor(self, ctx: str, task_prefix: str, additional_info: str, ) -> str:
        return f"{task_prefix} context: {ctx} \nAdditional infomations: {additional_info}"