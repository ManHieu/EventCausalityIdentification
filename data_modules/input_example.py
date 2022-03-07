from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Union
from torch.tensor import Tensor
from torch.utils.data.dataset import Dataset


@dataclass
class EntityType:
    """
    An entity type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences

    def __hash__(self):
        return hash(self.short)


@dataclass
class RelationType:
    """
    A relation type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences

    def __hash__(self):
        return hash(self.short)


@dataclass
class Entity:
    """
    An entity in a training/test example.
    """
    start: int                  # start index in the sentence
    end: int                    # end index in the sentence
    mention: str                # mention of entity
    type: Optional[EntityType] = None   # entity type
    id: Optional[int] = None    # id in the current training/test example

    def to_tuple(self):
        return self.type.natural, self.start, self.end

    def __hash__(self):
        return hash((self.id, self.start, self.end))


@dataclass
class Relation:
    """
    An (asymmetric) relation in a training/test example.
    """
    type: RelationType  # relation type
    head: Entity        # head of the relation
    tail: Entity        # tail of the relation

    def to_tuple(self):
        return self.type.natural, self.head.to_tuple(), self.tail.to_tuple()


@dataclass
class InputExample:
    """
    A single training/ testing example
    """
    id: Optional[str] = None                  # unique id in the dataset
    tokens: Optional[Union[List[str],str]] = None   # list of tokens (words)
    dataset: Optional[Dataset] = None   # dataset this example belongs to

    # Event extraction (Event detection, Event-Event relation, Event arguments extraction)
    triggers: List[Entity] = None
    relations: List[Relation] = None
    arguments: List[Relation] = None
    dep_path: List[List[str]] = None


@dataclass
class InputFeatures:
    """
    A single set of features of data
    Property names are the same names as the corresponding inputs to model.
    """
    input_ids: List[int]
    input_attention_mask: List[int]
    label_ids: Optional[List[int]] = None
    tgt_attention_mask: Optional[List[int]] = None


@dataclass
class ProcessedInputExample:
    context_sentence: str
    ED_template: str
    output_sentence: str

    def __hash__(self):
        return hash((self.context_sentence, self.ED_template, self.output_sentence))


