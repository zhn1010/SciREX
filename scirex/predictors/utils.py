from typing import Dict, List, Tuple
from scirex_utilities.entity_utils import used_entities
from collections import defaultdict
import json

def span_match(span_1, span_2):
    sa, ea = span_1
    sb, eb = span_2
    iou = (min(ea, eb) - max(sa, sb)) / (max(eb, ea) - min(sa, sb))
    return iou


def map_predicted_spans_to_gold(predicted_spans: List[tuple], gold_spans: List[tuple]):
    predicted_to_gold: Dict[tuple, tuple] = {}

    for p in predicted_spans:
        predicted_to_gold[(p[0], p[1])] = (p[0], p[1])
        for g in gold_spans:
            if span_match((p[0], p[1]), (g[0], g[1])) > 0.5:
                predicted_to_gold[(p[0], p[1])] = (g[0], g[1])
                break

    for p in predicted_spans:
        assert (p[0], p[1]) in predicted_to_gold, breakpoint()

    return predicted_to_gold


def intersect_predicted_clusters_to_gold(
    predicted_clusters: Dict[str, List[Tuple[int, int]]], gold_clusters: Dict[str, List[Tuple[int, int]]]
):

    predicted_clusters = {i: set(p) for i, p in predicted_clusters.items()}
    gold_clusters = {j: set(g) for j, g in gold_clusters.items()}

    intersection_scores = [
        [len(p & g) / len(p) for j, g in gold_clusters.items()] for i, p in predicted_clusters.items()
    ]

    intersection_scores_dict = {}
    for i, k in enumerate(predicted_clusters):
        intersection_scores_dict[k] = {}
        for j, l in enumerate(gold_clusters):
            if intersection_scores[i][j] > 0:
                intersection_scores_dict[k][l] = intersection_scores[i][j]


    return intersection_scores_dict


def map_and_intersect_predicted_clusters_to_gold(
    predicted_clusters: Dict[str, List[Tuple[int, int]]],
    gold_clusters: Dict[str, List[Tuple[int, int]]],
    predicted_to_gold_map,
    words,
    types,
    ner,
):
    # print('new document')
    def getNerTypeFromBound(span): 
        found = [x for x in ner if x[0] == span[0] and x[1] == span[1]]
        return found[0][2]
    
    predicted_tags_and_refs={"Material": defaultdict(list), "Method": defaultdict(list), "Metric": defaultdict(list), "Task": defaultdict(list), "Unknown": defaultdict(list)}
    gold_tags_and_refs={"Material": defaultdict(list), "Method": defaultdict(list), "Metric": defaultdict(list), "Task": defaultdict(list)}

    predicted_clusters = {k:[predicted_to_gold_map[tuple(x)] for x in v] for k, v in predicted_clusters.items()}
    gold_clusters = {k:[tuple(x) for x in v] for k, v in gold_clusters.items()}

    # predicted_clusters_string = {key: list(set([' '.join(words[span[0]:span[1]]) for span in value])) for (key, value) in predicted_clusters.items()}
    # gold_clusters_string = {key: list(set([' '.join(words[span[0]:span[1]]) for span in value])) for (key, value) in gold_clusters.items()}

    for (key, value) in predicted_clusters.items():
        if(len(value) != 0):
            str = list(set([' '.join(words[bound[0]: bound[1]]).lower() for bound in value]))
            entity_type = types[key]
            predicted_tags_and_refs[entity_type][key] = str

    for (key, value) in gold_clusters.items():
        if(len(value) != 0):
            str = list(set([' '.join(words[bound[0]: bound[1]]).lower() for bound in value]))
            entity_type = getNerTypeFromBound(value[0])
            gold_tags_and_refs[entity_type][key] = str

    with open('cluster_output.jsonl', 'a') as outputFile:
        json.dump({
      "Task": {'Gold': gold_tags_and_refs.get('Task', []), 'Predicted': predicted_tags_and_refs.get('Task', []) },
      "Method": {'Gold': gold_tags_and_refs.get('Method', []), 'Predicted': predicted_tags_and_refs.get('Method', []) },
      "Dataset": {'Gold': gold_tags_and_refs.get('Material', []) , 'Predicted': predicted_tags_and_refs.get('Material', []) },
      "Metric": {'Gold': gold_tags_and_refs.get('Metric', []), 'Predicted': predicted_tags_and_refs.get('Metric', []) }
      }, outputFile)
        outputFile.write('\n')
    # print('predicted_clusters', predicted_clusters_string)
    # print('gold_clusters', gold_clusters_string)

    intersection_scores = intersect_predicted_clusters_to_gold(predicted_clusters, gold_clusters)

    return intersection_scores

def convert_ner_to_dict(ner: Tuple[int, int, str]) :
    return {(x[0], x[1]):x[2] for x in ner}

def convert_ner_to_typed_list(ner: Dict[Tuple[int, int], str]) :
    return [(k[0], k[1], v) for k, v in ner.items()]

def convert_ner_to_list(ner) :
    return list(set([(span[0], span[1]) for span in ner]))

def merge_method_subrelations(doc) :
    true_entities = set([r[e] for r in doc['n_ary_relations'] for e in used_entities])
    method_subrelations = doc['method_subrelations']
    for m, subnames in method_subrelations.items() :
        for sm in subnames :
            if m != sm[1] :
                doc['coref'][m] += doc['coref'][sm[1]]

    new_clusters = {}
    for e in list(true_entities) :
        new_clusters[e] = list(set([tuple(x) for x in doc['coref'][e]]))

    del doc['method_subrelations']

    doc['coref'] = new_clusters

