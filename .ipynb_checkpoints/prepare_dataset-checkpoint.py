import os
import sys

import random

from loader.topics import *
from loader.features import *
from loader.qrels import *
from tools.normalization import *

def get_docids_random_sampling(number, doc_ids):
    '''
        Randomly sample n unique items from a list
    '''
    if len(doc_ids) < number:
        number = len(doc_ids)
    sampled_docids = random.sample(doc_ids, number)
    return sampled_docids

def get_docids_distribution(docids, docid_rel):
    '''
        get the distribution of document ids
    '''
    rel_docids = []
    irrel_docids = []
    nonannot_docids = []

    for doc_id in docids:
        relevance_grade = docid_rel.get(doc_id)

        if relevance_grade <= 0:
            irrel_docids.append(doc_id)
        elif relevance_grade > 0:
            rel_docids.append(doc_id)
        else:
            nonannot_docids.append(doc_id)

    return rel_docids, irrel_docids, nonannot_docids

def query_document_relevant_stats():
    '''
        Analyzing the number of relevant, irrelevant, and non-annotated documents
        in the relevance judgements and baseline retrieval
    '''
    queryid_annotation_dist = {}
    for query_id in query_ids:
        docid_rel = queryid_docid_rel.get(query_id)
        docid_features = queryid_docid_features.get(query_id)
        
        if docid_rel is None:
            docid_rel = {}

        docids = docid_features.keys()
        relevant_docids, irrelevant_docids, nonannot_docids = get_docids_distribution(docids, docid_rel)
        num_rel = len(relevant_docids)
        num_irrel = len(irrelevant_docids)
        num_non_annotated = len(nonannot_docids)
        queryid_annotation_dist[query_id] = {'rel':num_rel, 'irrel':num_irrel, 'nonannotated':num_non_annotated}

    did = ['rel','irrel','nonannotated']
    header = ['query_id'] + did
    with open(dist_file_path, 'w') as fr:
        fr.write('\t'.join(header) + '\n')
        for query_id in queryid_annotation_dist:
            dist = queryid_annotation_dist.get(query_id)
            nrel = dist.get(did[0])
            nirel = dist.get(did[1])
            nonannot = dist.get(did[2])
            fr.write(str(query_id) + '\t' + str(nrel) + '\t' + str(nirel) + '\t' + str(nonannot) + '\n')
        fr.close()

def get_naturally_distributed_docids(relevant_docids, irrelevant_docids, nonannot_docids):
    '''
        get the document ids to constitute the training samples based on natural distribution,
        that means, considering all the positive and negative available in the relevance 
        judgements.
    '''
    training_docids = relevant_docids
    num_rel = len(relevant_docids)
    num_irrel = len(irrelevant_docids)
    num_nonannot = len(nonannot_docids)
    
    #if there is no irrelevant documents, then choose the equal number of relevant documents from the non-annotated documents 
    if num_irrel == 0:
        sampled_docids = get_docids_random_sampling(nun_rel, nonannot_docids)
        training_docids.extend(sampled_docids)
    else:
            training_docids.extend(irrelevant_docids)    
    return training_docids

def get_equally_distributed_docids(relevant_docids, irrelevant_docids, nonannot_docids):
    '''
        get the document ids to constitute the training samples based on equal distribution 
        that means, equal number of negative examples to postive examples
    '''
    training_docids = relevant_docids
    num_rel = len(relevant_docids)
    num_irrel = len(irrelevant_docids)
    num_nonannot = len(nonannot_docids)
    
    #if there is no irrelevant documents, then choose the equal number of relevant documents from the non-annotated documents 
    if num_irrel == 0:
        sampled_docids = get_docids_random_sampling(nun_rel, nonannot_docids)
    else:
        sampled_docids = get_docids_random_sampling(num_rel, irrelevant_docids)
    
    training_docids.extend(sampled_docids)
    return training_docids

def get_doublly_distributed_docids(relevant_docids, irrelevant_docids, nonannot_docids):
    '''
        get the document ids to constitute the training samples based on double distribution 
        that means, two times more negative examples than postive examples
    '''
    training_docids = relevant_docids
    num_rel = len(relevant_docids)
    num_irrel = len(irrelevant_docids)
    num_nonannot = len(nonannot_docids)
    
    #if there is no irrelevant documents, then choose the equal number of relevant documents from the non-annotated documents 
    if num_irrel == 0:
        sampled_docids = get_docids_random_sampling(nun_rel*2, nonannot_docids)
    else:
        sampled_docids = get_docids_random_sampling(num_rel*2, irrelevant_docids)
    
    training_docids.extend(sampled_docids)
    return training_docids

def preparing_training_samples(dist_type, query_ids, queryid_docid_rel, queryid_docid_features, ltr_train_file_path):
    '''
        preparing the training samples based on the distribution type
    '''
    with open(ltr_train_file_path, 'w') as fw:
        for query_id in query_ids:
            docid_rel = queryid_docid_rel.get(query_id)
            docid_features = queryid_docid_features.get(query_id)

            if docid_rel is None:
                continue

            docids = docid_features.keys()
            relevant_docids, irrelevant_docids, nonannot_docids = get_docids_distribution(docids, docid_rel)
            
            if dist_type == 'equal_neg':
                training_docids = get_equally_distributed_docids(relevant_docids, irrelevant_docids, nonannot_docids)
            elif dist_type =='double_neg':
                training_docids = get_doublly_distributed_docids(relevant_docids, irrelevant_docids, nonannot_docids)
            else:
                training_docids = get_naturally_distributed_docids(relevant_docids, irrelevant_docids, nonannot_docids)
            
            for doc_id in training_docids:
                rel = docid_rel.get(doc_id)
                raw_features = docid_features.get(doc_id)
                features = get_l2_normalize(raw_features)
                
                #in case of non-annotated documents documents selected as irrelevant documents
                if rel is None:
                    rel = 0
                #in case of negative grade for an irrelevant documents
                if rel < 0:
                    rel = 0
                
                line = str(rel) + " " + "qid:"+str(query_id)
                feature_id = 1
                for idx in range(0, len(features)):
                    feature = features[idx]
                    line = line + " " + str(feature_id) + ":"+str(feature)
                    feature_id = feature_id + 1

                line = line + " # "+ str(doc_id)
                fw.write(line+'\n')
        fw.close()
        
def preparing_testing_samples():
            with open(ltr_test_file_path, 'w') as fw:
            for query_id in query_ids:
                docid_rel = queryid_docid_rel.get(query_id)
                docid_features = queryid_docid_features.get(query_id)

                if docid_rel is None:
                    docid_rel = {}

                for doc_id in docid_features:

                    rel = docid_rel.get(doc_id)
                    raw_features = docid_features.get(doc_id)
                    #print (doc_id)
                    #print (raw_features)
                    features = get_l2_normalize(raw_features)

                    if rel is None:
                        rel = 0 #dummy relevance for unjudged test data
                    if int(rel) < 0:
                        rel = 0

                    line = str(rel) + " " + "qid:"+str(query_id)

                    feature_id = 1
                    for idx in range(0, len(features)):

                        feature = features[idx]
                        line = line + " " + str(feature_id) + ":"+str(feature)
                        feature_id = feature_id + 1

                    line = line + " # "+ str(doc_id)
                    fw.write(line+'\n')
            fw.close()

def prepare_dataset(topics_path, query_doc_features_path, qrels_file_path, dist_type, dist_file_path, ltr_train_file_path, ltr_test_file_path):
        '''Preparing the learning to rank (L2R) datasets training and testing'''
        topics, query_ids = get_topics(topics_path)
        queryid_docid_rel = get_qrels(qrels_file_path)
        queryid_docid_features = get_query_doc_features(query_doc_features_path)

        print ("total topics: ", query_ids)

        #query-document-relevance triple analysis
        query_document_relevance_stats()

        #preparing the training dataset
        preparing_training_samples(dist_type, query_ids, queryid_docid_rel, queryid_docid_features, ltr_train_file_path)
        
        #preparing testing samples


def main():
        topics_path = sys.argv[1]
        query_doc_features_path = sys.argv[2]
        rel_judgment_path = sys.argv[3]
        dist_path = sys.argv[4]
        train_path = sys.argv[5]
        test_path = sys.argv[6]

        prepare_dataset(topics_path, query_doc_features_path, rel_judgment_path, dist_path, train_path, test_path)

if __name__ == '__main__':
    main()
