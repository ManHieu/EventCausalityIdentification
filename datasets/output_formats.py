import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict
import numpy as np 

from utils.utils import augment_sentence, get_span
from input_example import EntityType, InputExample, RelationType

OUTPUT_FORMATS = {}


def register_output_format(format_class):
    OUTPUT_FORMATS[format_class.name] = format_class
    return format_class


class BaseOutputFormat(ABC):
    name = None

    BEGIN_ENTITY_TOKEN = '['
    END_ENTITY_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_SEPARATOR_TOKEN = '='

    @abstractmethod
    def format_output(self, example: InputExample) -> str:
        """
        Format output in augmented natural language.
        """
        raise NotImplementedError
    
    @abstractmethod
    def run_inference(self, example: InputExample, output_sentence: str):
        """
        Process an output sentence to extract whatever information the task asks for.
        """
        raise NotImplementedError

    def parse_output_sentence(self, example: InputExample, output_sentence: str) -> Tuple[list, bool]:
        """
        Parse an output sentence in augmented language and extract inferred entities and tags.
        Return a pair (predicted_entities, wrong_reconstruction), where:
        - each element of predicted_entities is a tuple (entity_name, tags, start, end)
            - entity_name (str) is the name as extracted from the output sentence
            - tags is a list of tuples, obtained by |-splitting the part of the entity after the entity name
            - this entity corresponds to the tokens example.tokens[start:end]
            - note that the entity_name could differ from ' '.join(example.tokens[start:end]), if the model was not
              able to exactly reproduce the entity name, or if alignment failed
        - wrong_reconstruction (bool) says whether the output_sentence does not match example.tokens exactly

        An example follows.

        example.tokens:
        ['Tolkien', 'wrote', 'The', 'Lord', 'of', 'the', 'Rings']

        output_sentence:
        [ Tolkien | person ] wrote [ The Lord of the Rings | book | author = Tolkien ]

        output predicted entities:
        [
            ('Tolkien', [('person',)], 0, 1),
            ('The Lord of the Rings', [('book',), ('author', 'Tolkien')], 2, 7)
        ]
        """
        output_tokens = []
        unmatched_predicted_entities = []

        # add spaces around special tokens, so that they are alone when we split
        padded_output_sentence = output_sentence
        for special_token in [
            self.BEGIN_ENTITY_TOKEN, self.END_ENTITY_TOKEN,
            self.SEPARATOR_TOKEN, self.RELATION_SEPARATOR_TOKEN,
        ]:
            padded_output_sentence = padded_output_sentence.replace(special_token, ' ' + special_token + ' ')

        entity_stack = []   # stack of the entities we are extracting from the output sentence
        # this is a list of lists [start, state, entity_name_tokens, entity_other_tokens]
        # where state is "name" (before the first | separator) or "other" (after the first | separator)

        for token in padded_output_sentence.split():
            if len(token) == 0:
                continue

            elif token == self.BEGIN_ENTITY_TOKEN:
                # begin entity
                start = len(output_tokens)
                entity_stack.append([start, "name", [], []])

            elif token == self.END_ENTITY_TOKEN and len(entity_stack) > 0:
                # end entity
                start, state, entity_name_tokens, entity_other_tokens = entity_stack.pop()

                entity_name = ' '.join(entity_name_tokens).strip()
                end = len(output_tokens)

                tags = []

                # split entity_other_tokens by |
                splits = [
                    list(y) for x, y in itertools.groupby(entity_other_tokens, lambda z: z == self.SEPARATOR_TOKEN)
                    if not x
                ]

                if state == "other" and len(splits) > 0:
                    for x in splits:
                        tags.append(tuple(' '.join(x).split(' ' + self.RELATION_SEPARATOR_TOKEN + ' ')))

                unmatched_predicted_entities.append((entity_name, tags, start, end))

            else:
                # a normal token
                if len(entity_stack) > 0:
                    # inside some entities
                    if token == self.SEPARATOR_TOKEN:
                        x = entity_stack[-1]

                        if x[1] == "name":
                            # this token marks the end of name tokens for the current entity
                            x[1] = "other"
                        else:
                            # simply add this token to entity_other_tokens
                            x[3].append(token)

                    else:
                        is_name_token = True

                        for x in reversed(entity_stack):
                            # check state
                            if x[1] == "name":
                                # add this token to entity_name_tokens
                                x[2].append(token)

                            else:
                                # add this token to entity_other tokens and then stop going up in the tree
                                x[3].append(token)
                                is_name_token = False
                                break

                        if is_name_token:
                            output_tokens.append(token)

                else:
                    # outside
                    output_tokens.append(token)

        # check if we reconstructed the original sentence correctly, after removing all spaces
        wrong_reconstruction = (''.join(output_tokens) != ''.join(example.tokens))

        # now we align self.tokens with output_tokens (with dynamic programming)
        cost = np.zeros((len(example.tokens) + 1, len(output_tokens) + 1))  # cost of alignment between tokens[:i]
        # and output_tokens[:j]
        best = np.zeros_like(cost, dtype=int)  # best choice when aligning tokens[:i] and output_tokens[:j]

        for i in range(len(example.tokens) + 1):
            for j in range(len(output_tokens) + 1):
                if i == 0 and j == 0:
                    continue

                candidates = []

                # match
                if i > 0 and j > 0:
                    candidates.append(
                        ((0 if example.tokens[i - 1] == output_tokens[j - 1] else 1) + cost[i - 1, j - 1], 1))

                # skip in the first sequence
                if i > 0:
                    candidates.append((1 + cost[i - 1, j], 2))

                # skip in the second sequence
                if j > 0:
                    candidates.append((1 + cost[i, j - 1], 3))

                chosen_cost, chosen_option = min(candidates)
                cost[i, j] = chosen_cost
                best[i, j] = chosen_option

        # reconstruct best alignment
        matching = {}

        i = len(example.tokens) - 1
        j = len(output_tokens) - 1

        while i >= 0 and j >= 0:
            chosen_option = best[i + 1, j + 1]

            if chosen_option == 1:
                # match
                matching[j] = i
                i, j = i - 1, j - 1

            elif chosen_option == 2:
                # skip in the first sequence
                i -= 1

            else:
                # skip in the second sequence
                j -= 1

        # update predicted entities with the positions in the original sentence
        predicted_entities = []

        for entity_name, entity_tags, start, end in unmatched_predicted_entities:
            new_start = None  # start in the original sequence
            new_end = None  # end in the original sequence

            for j in range(start, end):
                if j in matching:
                    if new_start is None:
                        new_start = matching[j]

                    new_end = matching[j]

            if new_start is not None:
                # predict entity
                entity_tuple = (entity_name, entity_tags, new_start, new_end + 1)
                predicted_entities.append(entity_tuple)

        return predicted_entities, wrong_reconstruction


@register_output_format
class JointEROutputFormat(BaseOutputFormat):
    """
    Output format for joint event and event-event relation extraction.
    """
    name = 'joint_er'
    
    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, for example:
        [ Tolkien | person | born in = here ] was born [ here | location ]
        """
        # organize relations by head entity
        relations_by_entity = {entity: [] for entity in example.triggers}
        for relation in example.relations:
            relations_by_entity[relation.head].append((relation.type, relation.tail))
        
        augmentations = []
        for event in example.triggers:
            tags = [(event.type.natural, )]
            for relation_type, tail in relations_by_entity[event]:
                tags.append((relation_type.natural, ' '.join(example.tokens[tail.start: tail.end])))
            
            augmentations.append((tags, event.start, event.end,))
        
        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)
    
    def run_inference(self, example: InputExample, output_sentence: str,
                      event_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None) \
            -> Tuple[set, set, bool, bool, bool, bool]:
        """
        Process an output sentence to extract predicted entities and relations (among the given entity/relation types).

        Return the predicted entities, predicted relations, and four booleans which describe if certain kinds of errors
        occurred (wrong reconstruction of the sentence, label error, entity error, augmented language format error).
        """
        label_error = False     # whether the output sentence has at least one non-existing entity or relation type
        entity_error = False    # whether there is at least one relation pointing to a non-existing head entity
        format_error = False    # whether the augmented language format is invalid

        if output_sentence.count(self.BEGIN_ENTITY_TOKEN) != output_sentence.count(self.END_ENTITY_TOKEN):
            # the parentheses do not match
            format_error = True

        event_types = set(event_type.natural for event_type in event_types.values())
        relation_types = set(relation_type.natural for relation_type in relation_types.values()) \
            if relation_types is not None else {}
        
        # parse output sentence
        raw_predicted_events, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)

        # update predicted entities with the positions in the original sentence
        predicted_events_by_name = defaultdict(list)
        predicted_events = set()
        raw_predicted_relations = []
        # process and filter entities
        for event_name, tags, start, end in raw_predicted_events:
            if len(tags) == 0 or len(tags[0]) > 1:
                # we do not have a tag or more than a tag_type for the entity type
                format_error = True
                continue

            event_type = tags[0][0]

            if event_type in event_types:
                event_tuple = (event_type, start, end)
                predicted_events.add(event_tuple)
                predicted_events_by_name[event_name].append(event_tuple)

                # process tags to get relations
                for tag in tags[1:]:
                    if len(tag) == 2:
                        relation_type, related_event = tag
                        if relation_type in relation_types:
                            raw_predicted_relations.append((relation_type, event_tuple, related_event))
                        else:
                            label_error = True

                    else:
                        # the relation tag has the wrong length
                        format_error = True

            else:
                # the predicted entity type does not exist
                label_error = True
        
        predicted_relations = set()
        for relation_type, event_tuple, related_event in raw_predicted_relations:
            if related_event in predicted_events_by_name:
                # look for the closest instance of the related entity
                # (there could be many of them)
                _, head_start, head_end = event_tuple
                candidates = sorted(
                    predicted_events_by_name[related_event],
                    key=lambda x:
                    min(abs(x[1] - head_end), abs(head_start - x[2]))
                )

                for candidate in candidates:
                    relation = (relation_type, event_tuple, candidate)

                    if relation not in predicted_relations:
                        predicted_relations.add(relation)
                        break

            else:
                # cannot find the related entity in the sentence
                entity_error = True
        
        return predicted_events, predicted_relations, wrong_reconstruction, label_error, entity_error, format_error
