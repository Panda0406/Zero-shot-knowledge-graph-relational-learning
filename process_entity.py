# -*- coding: utf-8 -*-
import json
from collections import defaultdict

def type_to_entity():
    '''
    build candiate tail entities for every relation
    '''
    # calculate node degrees
    #with open('../path_graph') as f:
    #    for line in f:
    #        line = line.rstrip()
    #        e1 = line.split('\t')[0]
    #        e2 = line.split('\t')[2]

    ent2ids = json.load(open('../ent2ids'))

    all_entities = ent2ids.keys()

    type2ents = defaultdict(set)
    entity_exception = list()

    for ent in all_entities:
        if len(ent.split(':')) != 3:
            print(ent)
            entity_exception.append(ent)
        try:
            type_ = ent.split(':')[1]
            type2ents[type_].add(ent)
        except Exception as e:
            continue

    for k,v in type2ents.items():
        type2ents[k] = list(v)

    json.dump(type2ents, open('../type2ents.json', 'w'))

    print('Entity Exception:', len(entity_exception))

    print('type2sent:', len(type2ents))

    for k,v in type2ents.items():
        print(k, '        ', len(v))

def process_tasks(train_tasks, mode):
    type2ents = json.load(open('../type2ents.json'))
    type2rela = json.load(open('../e_type2rela.json'))
    rela2relas = dict()

    wrong = set()
    for rela, triples in train_tasks.items():
        e1_types = set()
        e2_types = set()
        related_rela = set()
        e1_rela = set()
        e2_rela = set()
        for triple in triples:
            e1, e2 = triple[0], triple[2]
            e1_t = e1.split(':')[1]
            e2_t = e2.split(':')[1]

            if e1_t in type2ents.keys(): # type2ents.has_key(e1_t):
                e1_types.add(e1_t)
            else:
                wrong.add(e1)

            if e2_t in type2ents.keys(): # type2ents.has_key(e2_t):
                e2_types.add(e2_t)
            else:
                wrong.add(e2)

        for e1_t in e1_types:
            if e1_t in type2rela.keys(): #type2rela.has_key(e1_t):
                for item in type2rela[e1_t]:
                    related_rela.add(item)
                    e1_rela.add(item)

        for e2_t in e2_types:
            if e2_t in type2rela.keys(): # type2rela.has_key(e2_t):
                for item in type2rela[e2_t]:
                    related_rela.add(item)
                    e2_rela.add(item)

        rela2relas[rela] = list(related_rela)



        print('RELATION: ', rela, len(related_rela), len(e1_rela), len(e2_rela))
        #print 'ENTITY_1: ', str(list(e1_types))
        #print 'ENTITY_2: ', str(list(e2_types))

    print("WRONG: ", len(wrong))

    json.dump(rela2relas, open('../' + mode + '_rela2relas.json', 'w'))

def e1e2_type2rela(pretrain_tasks):

    e1e2_type2rela = defaultdict(set)

    for rela, triples in pretrain_tasks.items():
        for triple in triples:
            e1 = triple[0]
            e2 = triple[2]
            if len(e1.split(':')) != 3 or len(e2.split(':')) != 3:
                continue
            e1_type = e1.split(':')[1]
            e2_type = e2.split(':')[1]
            e1e2_type2rela[e1_type+'#'+e2_type].add(rela)
            e1e2_type2rela[e2_type+'#'+e1_type].add(rela)

    print('length: ', len(e1e2_type2rela))
    count2 = 0
    relations = list()
    for k,v in e1e2_type2rela.items():
        relations += v
        if len(v) > 1:
            count2 += 1

    print('larger than 2: ', count2)
    print(len(set(relations)))

    for k,v in e1e2_type2rela.items():
        e1e2_type2rela[k] = list(v)
    json.dump(e1e2_type2rela, open('../e1e2_type2rela.json', 'w'))

def e_type2rela(pretrain_tasks):

    e_type2rela = defaultdict(set)

    for rela, triples in pretrain_tasks.items():
        if len(triples) < 500:
            continue
        for triple in triples:
            e1 = triple[0]
            e2 = triple[2]
            if len(e1.split(':')) != 3 or len(e2.split(':')) != 3:
                continue
            e1_type = e1.split(':')[1]
            e2_type = e2.split(':')[1]
            e_type2rela[e1_type].add(rela)
            e_type2rela[e2_type].add(rela)

    print('length: ', len(e_type2rela))
    #count2 = 0
    #relations = list()
    #for k,v in e_type2rela.items():
    #    relations += v
    #    if len(v) > 1:
    #        count2 += 1

    count = 0
    for k,v in e_type2rela.items():
        if len(v) > 28:
            count += 1
            del e_type2rela[k]
            continue
        e_type2rela[k] = list(v)
    print('Common Entity: ', count)
    json.dump(e_type2rela, open('../e_type2rela.json', 'w'))



# if __name__ == "__main__":
#
#     #type_to_entity()
#
#     pretrain_tasks = json.load(open('../pretrain_tasks.json'))
#     e_type2rela(pretrain_tasks)
#
#     train_tasks = json.load(open('../train_tasks.json'))
#     dev_tasks = json.load(open('../dev_tasks.json'))
#     test_tasks = json.load(open('../test_tasks.json'))
#
#     reldes2id = json.load(open(self.data_path + 'reldes2ids'))
#
#     test_tasks = json.load(open(self.data_path + 'pretrain_tasks.json'))
#     rela2label = dict()
#     rela_sorted = sorted(list(test_tasks.keys()))
#     for i, rela in enumerate(rela_sorted):
#         rela2label[rela] = int(i)
#     rela_matrix = np.
#
#     process_tasks(train_tasks, 'train')
#     print('\n')
#     process_tasks(dev_tasks, 'dev')
#     print('\n')
#     process_tasks(test_tasks, 'test')



