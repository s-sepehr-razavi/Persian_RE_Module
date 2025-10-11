import os
import os.path
import json
import numpy as np
from collections import defaultdict
import pandas as pd

rel2id = json.load(open('meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}
top10 = ('P131', 'P17', 'P27', 'P150', 'P800', 'P527', 'P361', 'P577', 'P463', 'P175')
# top10 = ('P131', 'P17', 'P27', 'P150', 'P800',)

def to_official(preds, features):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res

def omitting_empty_entities(sample):
    labels = sample['labels']
    vertexSet = sample['vertexSet']

    # Build mapping from old indices to new indices
    index_map = {}
    new_vertexSet = []
    for old_idx, v in enumerate(vertexSet):
        if len(v) > 0:
            new_idx = len(new_vertexSet)
            new_vertexSet.append(v)
            index_map[old_idx] = new_idx

    # Update labels
    new_labels = []
    for label in labels:
        h, t = label['h'], label['t']
        if h not in index_map or t not in index_map:
            continue  # skip labels pointing to removed entities
        new_label = label.copy()
        new_label['h'] = index_map[h]
        new_label['t'] = index_map[t]
        new_labels.append(new_label)

    sample['labels'] = new_labels
    sample['vertexSet'] = new_vertexSet
    return sample

def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        data = omitting_empty_entities(data)
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path, tag, args, save_per_relation_path=None):
    """
    Adapted from the official evaluation code + per-relation performance computation.
    """
    truth_dir = os.path.join(path, 'ref')
    os.makedirs(truth_dir, exist_ok=True)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, args.train_file), truth_dir)

    # Load truth data
    if tag == 'dev':
        truth = json.load(open(os.path.join(path, args.dev_file)))
    else:
        truth = json.load(open(os.path.join(path, args.test_file)))

    std = {}
    tot_evidences = 0
    titleset = set()
    title2vectexSet = {}

    for x in truth:
        x = omitting_empty_entities(x)
        title = x['title']
        titleset.add(title)
        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx, t_idx = label['h'], label['t']
            std[(title, r, h_idx, t_idx)] = set([1])
            tot_evidences += 1

    tot_relations = len(std)

    # Deduplicate predictions
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x, y = tmp[i], tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(x)

    # Overall counters
    correct_re = 0
    correct_evidence = 0
    pred_evi = 0
    correct_in_train_annotated = 0

    # NEW: Per-relation stats
    rel_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

    titleset2 = set()
    for x in submission_answer:
        title, h_idx, t_idx, r = x['title'], x['h_idx'], x['t_idx'], x['r']
        titleset2.add(title)

        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        evi = set(x.get('evidence', []))
        pred_evi += len(evi)

        # Count predictions
        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            rel_stats[r]['TP'] += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)

            # Mark if in train
            in_train_annotated = any(
                (n1['name'], n2['name'], r) in fact_in_train_annotated
                for n1 in vertexSet[h_idx] for n2 in vertexSet[t_idx]
            )
            if in_train_annotated:
                correct_in_train_annotated += 1
        else:
            rel_stats[r]['FP'] += 1

    # FN per relation (ground truth missed)
    relations_set = {(x['title'], x['r'], x['h_idx'], x['t_idx']) for x in submission_answer}
    for (title, r, h_idx, t_idx) in std.keys():
        if (title, r, h_idx, t_idx) not in relations_set:
            rel_stats[r]['FN'] += 1

    # === Overall metrics ===
    re_p = correct_re / len(submission_answer) if submission_answer else 0
    re_r = correct_re / tot_relations if tot_relations else 0
    re_f1 = 2 * re_p * re_r / (re_p + re_r) if (re_p + re_r) > 0 else 0

    evi_p = correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = correct_evidence / tot_evidences if tot_evidences > 0 else 0
    evi_f1 = 2 * evi_p * evi_r / (evi_p + evi_r) if (evi_p + evi_r) > 0 else 0

    re_p_ignore_train_annotated = (correct_re - correct_in_train_annotated) / (
        (len(submission_answer) - correct_in_train_annotated + 1e-5)
    )
    re_f1_ignore_train_annotated = 2 * re_p_ignore_train_annotated * re_r / (
        re_p_ignore_train_annotated + re_r
    ) if (re_p_ignore_train_annotated + re_r) > 0 else 0

    # === Per-relation metrics ===
    rel_metrics = []
    for r, v in rel_stats.items():
        TP, FP, FN = v['TP'], v['FP'], v['FN']
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        rel_metrics.append({
            'relation': r,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': TP,
            'fp': FP,
            'fn': FN
        })

    df = pd.DataFrame(rel_metrics).sort_values(by='f1', ascending=False)

    if save_per_relation_path:
        df.to_csv(os.path.join(save_per_relation_path, "per_relation.csv"), index=False)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_p, re_r
