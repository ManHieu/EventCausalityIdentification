from typing import Dict, List, Tuple
import random
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from rouge import Rouge

rouge = Rouge()

sim_evaluator = SentenceTransformer('../all-MiniLM-L12-v1')


def get_span(l: List[str], span: List[int]):
    assert len(span) == 2
    return " ".join([l[i] for i in range(span[0], span[1]) if i < len(l)])


def expand_tokens(tokens: List[str], augmentations: List[Tuple[List[tuple], int, int]],
                  entity_tree: Dict[int, List[int]], root: int,
                  begin_entity_token: str, sep_token: str, relation_sep_token: str, end_entity_token: str) \
        -> List[str]:
    """
    Recursively expand the tokens to obtain a sentence in augmented natural language.

    Used in the augment_sentence function below (see the documentation there).
    """
    new_tokens = []
    root_start, root_end = augmentations[root][1:] if root >= 0 else (0, len(tokens))
    i = root_start  # current index

    for entity_index in entity_tree[root]:
        tags, start, end = augmentations[entity_index]

        # add tokens before this entity
        new_tokens += tokens[i:start]

        # expand this entity
        new_tokens.append(begin_entity_token)
        new_tokens += expand_tokens(tokens, augmentations, entity_tree, entity_index,
                                    begin_entity_token, sep_token, relation_sep_token, end_entity_token)

        for tag in tags:
            if tag[0]:
                # only append tag[0] if it is a type, otherwise skip the type
                new_tokens.append(sep_token)
                new_tokens.append(tag[0])

            for x in tag[1:]:
                new_tokens.append(relation_sep_token)
                new_tokens.append(x)

        new_tokens.append(end_entity_token)
        i = end

    # add tokens after all entities
    new_tokens += tokens[i:root_end]

    return new_tokens


def augment_sentence(tokens: List[str], augmentations: List[Tuple[List[tuple], int, int]], begin_entity_token: str,
                     sep_token: str, relation_sep_token: str, end_entity_token: str) -> str:
    """
    Augment a sentence by adding tags in the specified positions.

    Args:
        tokens: Tokens of the sentence to augment.
        augmentations: List of tuples (tags, start, end).
        begin_entity_token: Beginning token for an entity, e.g. '['
        sep_token: Separator token, e.g. '|'
        relation_sep_token: Separator token for relations, e.g. '='
        end_entity_token: End token for an entity e.g. ']'

    An example follows.

    tokens:
    ['Tolkien', 'was', 'born', 'here']

    augmentations:
    [
        ([('person',), ('born in', 'here')], 0, 1),
        ([('location',)], 3, 4),
    ]

    output augmented sentence:
    [ Tolkien | person | born in = here ] was born [ here | location ]
    """
    # sort entities by start position, longer entities first
    augmentations = list(sorted(augmentations, key=lambda z: (z[1], -z[2])))

    # check that the entities have a tree structure (if two entities overlap, then one is contained in
    # the other), and build the entity tree
    root = -1   # each node is represented by its position in the list of augmentations, except that the root is -1
    entity_tree = {root: []}        # list of children of each node
    current_stack = [root]          # where we are in the tree

    for j, x in enumerate(augmentations):
        tags, start, end = x
        if any(augmentations[k][1] < start < augmentations[k][2] < end for k in current_stack):
            # tree structure is not satisfied!
            print(f'Tree structure is not satisfied! Dropping annotation {x}')
            continue

        while current_stack[-1] >= 0 and \
                not (augmentations[current_stack[-1]][1] <= start <= end <= augmentations[current_stack[-1]][2]):
            current_stack.pop()

        # add as a child of its father
        entity_tree[current_stack[-1]].append(j)

        # update stack
        current_stack.append(j)

        # create empty list of children for this new node
        entity_tree[j] = []

    return ' '.join(expand_tokens(
        tokens, augmentations, entity_tree, root, begin_entity_token, sep_token, relation_sep_token, end_entity_token
    ))


def compute_f1(predicts: List[str], golds: List[str]):
    n_predict = 0
    n_gold = 0
    tp = 0
    wrong_struct = 0
    for predict, gold in zip(predicts, golds):
        if predict.startswith('No')==False and gold.startswith('No')==False:
            tp = tp + 1
        if predict.startswith('No')==False:
            n_predict = n_predict + 1
        if gold.startswith('No')==False:
            n_gold = n_gold + 1
        if predict.startswith('Yes')==False and predict.startswith('No')==False:
            wrong_struct = wrong_struct + 1
    
    if wrong_struct == len(predicts):
        return 0.01, 0.01, 0.01, 0, 0, 0
    elif n_predict==n_gold==0:
        return 0.1, 0.1, 0.1, 0, 0, 0
    else:
        p = tp/(n_predict + 1)
        r = tp/(n_gold + 1)
        f1 = 2 * p * r / (p + r + 1e-9)
        return f1, p, r, tp, n_predict, n_gold


def create_distractor(items: List[str]):
    distracted_items = []
    if len(set(items)) == 1:
        return items
    else:
        for item in items:
            distracted_item = random.choice(list(set(items) - set([item])))
            distracted_items.append(distracted_item)
        return distracted_items

@torch.no_grad()
def compute_sentences_similar(sents_A: List[str], sents_B: List[str], metric: str):
    assert len(sents_A) == len(sents_B)
    origins = []
    reconstructs = []
    for i in range(len(sents_A)):
        ori_sent = ' '.join([word.strip() for word in sents_A[i].split()])
        re_sent = ' '.join([word.strip() for word in sents_B[i].split()])
        origins.append(ori_sent)
        reconstructs.append(re_sent)

    if metric=='vector_sim':
        embeddings1 = sim_evaluator.encode(origins, convert_to_tensor=True)
        embeddings2 = sim_evaluator.encode(reconstructs, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        scores = []
        for i in range(len(sents_B)):
            scores.append(abs(float(cosine_scores[i][i])))
    elif metric == 'rouge':
        scores = []
        for ori, rec in zip(origins, reconstructs):
            score = rouge.get_scores(rec, ori)
            # print(score[0]['rouge-2']['f'])
            scores.append(score[0]['rouge-2']['f'])
    elif metric == 'bleu':
        score = []
    else:
        raise ValueError("We haven't yet support this metric")
    return scores
