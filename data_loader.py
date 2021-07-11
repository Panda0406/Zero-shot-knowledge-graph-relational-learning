# -*- coding: utf-8 -*-
import json
import random
from modules import random_pick

# Using
def Extractor_generate(dataset, batch_size, symbol2id, ent2id, e1rel_e2, few, sub_epoch):

    print('\nLOADING PRETRAIN TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    rel2candidates = json.load(open(dataset + '/rel2candidates_all.json'))

    task_pool = train_tasks.keys()

    t_num = list()
    for k in task_pool:
        if len(rel2candidates[k]) <= 20:
            v = 0
        else:
            v = min(len(rel2candidates[k]), 1000)
        t_num.append(v)
    t_sum = sum(t_num)
    probability = [float(item)/t_sum for item in t_num]

    while True:
        support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right = \
           list(), list(), list(), list(), list(), list(), list(), list(), list()
        query = random_pick(task_pool, probability)
        for _ in range(sub_epoch):
            candidates = rel2candidates[query]

            train_and_test = train_tasks[query]

            random.shuffle(train_and_test)

            support_triples = train_and_test[:few]

            support_pairs += [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]

            support_left += [ent2id[triple[0]] for triple in support_triples]
            support_right += [ent2id[triple[2]] for triple in support_triples]

            all_test_triples = train_and_test[few:]

            if len(all_test_triples) == 0:
                continue

            if len(all_test_triples) < batch_size:
                query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
            else:
                query_triples = random.sample(all_test_triples, batch_size)

            query_pairs += [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            query_left += [ent2id[triple[0]] for triple in query_triples]
            query_right += [ent2id[triple[2]] for triple in query_triples]

            for triple in query_triples:
                e_h = triple[0]
                rel = triple[1]
                e_t = triple[2]
                while True:
                    noise = random.choice(candidates)
                    if noise in ent2id.keys():#ent2id.has_key(noise):
                        if (noise not in e1rel_e2[e_h+rel]) and noise != e_t:
                            break
                false_pairs.append([symbol2id[e_h], symbol2id[noise]])
                false_left.append(ent2id[e_h])
                false_right.append(ent2id[noise])

        yield support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right


def centroid_generate(dataset, relation_name, symbol2id, ent2id, train_tasks, rela2label):

    all_test_triples = train_tasks[relation_name]

    query_triples = all_test_triples

    query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

    query_left = [ent2id[triple[0]] for triple in query_triples]
    query_right = [ent2id[triple[2]] for triple in query_triples]

    return query_pairs, query_left, query_right, rela2label[relation_name]


def train_generate_decription(dataset, batch_size, symbol2id, ent2id, e1rel_e2, rel2id, args, rela2label, rela_matrix):
    print('##LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + 'train_tasks.json'))
    print('##LOADING CANDIDATES')
    rel2candidates = json.load(open(dataset + 'rel2candidates_all.json'))
    task_pool = list(train_tasks.keys())

    while True:
        rel_batch, query_pairs, query_left, query_right, false_pairs, false_left, false_right, labels = [], [], [], [], [], [], [], []
        random.shuffle(task_pool)
        for query in task_pool[:args.gan_batch_rela]:
            relation_id = rel2id[query]
            candidates = rel2candidates[query]

            if len(candidates) <= 20:
                # print 'not enough candidates'
                continue

            train_and_test = train_tasks[query]

            random.shuffle(train_and_test)

            all_test_triples = train_and_test

            if len(all_test_triples) == 0:
                continue

            if len(all_test_triples) < batch_size:
                query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
            else:
                query_triples = random.sample(all_test_triples, batch_size)

            query_pairs += [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            query_left += [ent2id[triple[0]] for triple in query_triples]
            query_right += [ent2id[triple[2]] for triple in query_triples]

            label = rela2label[query]

            # generate negative samples
            false_pairs_ = []
            false_left_ = []
            false_right_ = []
            for triple in query_triples:
                e_h = triple[0]
                rel = triple[1]
                e_t = triple[2]
                while True:
                    noise = random.choice(candidates)
                    if noise in ent2id.keys(): # ent2id.has_key(noise):
                        if (noise not in e1rel_e2[e_h+rel]) and noise != e_t:
                            break
                false_pairs_.append([symbol2id[e_h], symbol2id[noise]])
                false_left_.append(ent2id[e_h])
                false_right_.append(ent2id[noise])

            false_pairs += false_pairs_
            false_left += false_left_
            false_right += false_right_


            rel_batch += [rel2id[query] for _ in range(batch_size)]

            labels += [rela2label[query]] * batch_size

        yield rela_matrix[rel_batch], query_pairs, query_left, query_right, false_pairs, false_left, false_right, labels
