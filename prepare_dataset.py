import os
import sys

import random
import numpy as np

from loader.topics import *
from loader.features import *
from loader.qrels import *
from tools.normalization import *


def get_minmax_norm(features_mat, index):
    '''
        conducting min-max normalization according to index (1 for row and 0 for column)
    '''
    features_normalized = (features_mat - features_mat.min(axis=index))/(features_mat.max(axis=index) - features_mat.min(axis=index))
    features_normalized[np.isnan(features_normalized)] = 0
    return features_normalized


def get_docids_random_sampling(number, doc_ids):
    '''
        Randomly sample n unique items from a list
    '''
    if len(doc_ids) < number:
        number = len(doc_ids)
    sampled_docids = random.sample(doc_ids, number)
    return sampled_docids

def get_docids_lowranked(num_rel, doc_ids):
    '''
        Take the document ids from the bottom ranked non-annotated documents
    '''
    if len(doc_ids) < num_rel:
        num_rel = len(doc_ids)

    chosen_docids = doc_ids[-num_rel:]
    return chosen_docids


def get_docids_distribution(docids, docid_rel):
    '''
        get the distribution of document ids
    '''
    rel_docids = []
    irrel_docids = []
    nonannot_docids = []

    for doc_id in docids:
        relevance_grade = docid_rel.get(doc_id)

        if relevance_grade is None:
            nonannot_docids.append(doc_id)
        elif relevance_grade <= 0:
            irrel_docids.append(doc_id)
        elif relevance_grade > 0:
            rel_docids.append(doc_id)
        else:
            nonannot_docids.append(doc_id)

    return rel_docids, irrel_docids, nonannot_docids


def query_document_relevance_stats(query_ids, queryid_docid_rel, queryid_docid_features, dist_file_path):
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


def get_natural_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids):
    '''
        get the document ids to constitute the training samples based on natural distribution,
        that means, considering all the positive and negative available in the relevance
        judgements.
    '''
    training_docids = relevant_docids
    num_rel = len(relevant_docids)
    num_irrel = len(irrelevant_docids)
    num_nonannot = len(nonannot_docids)

    #if there is no irrelevant documents, then we choose an equal number of relevant documents from 
    #the non-annotated documents 
    if num_irrel == 0:
        #sampled_docids = get_docids_random_sampling(num_rel, nonannot_docids)
        lowranked_docids = get_docids_lowranked(num_rel, nonannot_docids)
        training_docids.extend(lowranked_docids)
    else:
        training_docids.extend(irrelevant_docids)

    return training_docids


def get_equal_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids):
    '''
        get the document ids to constitute the training samples based on equal distribution
        that means, equal number of negative examples to postive examples
    '''
    training_docids = relevant_docids
    num_rel = len(relevant_docids)
    num_irrel = len(irrelevant_docids)
    num_nonannot = len(nonannot_docids)

    #if there is no irrelevant documents, then we choose an equal number of relevant documents 
    #from the non-annotated documents as irrelevant documents

    if num_irrel == 0:
#        sampled_docids = get_docids_random_sampling(num_rel, nonannot_docids)
        lowranked_docids = get_docids_lowranked(num_rel, nonannot_docids)
        sampled_docids = lowranked_docids
    else:
        sampled_docids = get_docids_random_sampling(num_rel, irrelevant_docids)

    training_docids.extend(sampled_docids)
    return training_docids


def get_double_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids):
    '''
        get two times more negative examples than postive examples
    '''
    training_docids = relevant_docids
    num_rel = len(relevant_docids)
    num_irrel = len(irrelevant_docids)
    num_nonannot = len(nonannot_docids)

    #if there is no irrelevant documents, then we choose an equal number of relevant documents 
    # from the non-annotated documents as irrelevant documents
    if num_irrel == 0:
#        sampled_docids = get_docids_random_sampling(num_rel*2, nonannot_docids)
        lowranked_docids = get_docids_lowranked(num_rel, nonannot_docids)
        sampled_docids = lowranked_docids
    else:
        sampled_docids = get_docids_random_sampling(num_rel*2, irrelevant_docids)

    training_docids.extend(sampled_docids)
    return training_docids


def get_triple_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids):
    '''
        get three times more negative examples than postive examples
    '''
    training_docids = relevant_docids
    num_rel = len(relevant_docids)
    num_irrel = len(irrelevant_docids)
    num_nonannot = len(nonannot_docids)

    #if there is no irrelevant documents, then we choose an equal number of relevant documents 
    # from the non-annotated documents as irrelevant documents
    if num_irrel == 0:
#        sampled_docids = get_docids_random_sampling(num_rel*3, nonannot_docids)
        lowranked_docids = get_docids_lowranked(num_rel, nonannot_docids)
        sampled_docids = lowranked_docids
    else:
        sampled_docids = get_docids_random_sampling(num_rel*3, irrelevant_docids)

    training_docids.extend(sampled_docids)
    return training_docids


def get_quadruple_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids):
    '''
        get three times more negative examples than postive examples
    '''
    training_docids = relevant_docids
    num_rel = len(relevant_docids)
    num_irrel = len(irrelevant_docids)
    num_nonannot = len(nonannot_docids)

    #if there is no irrelevant documents, then we choose an equal number of relevant documents 
    # from the non-annotated documents as irrelevant documents
    if num_irrel == 0:
    #   sampled_docids = get_docids_random_sampling(num_rel*4, nonannot_docids)
        lowranked_docids = get_docids_lowranked(num_rel, nonannot_docids)
        sampled_docids = lowranked_docids
    else:
        sampled_docids = get_docids_random_sampling(num_rel*4, irrelevant_docids)

    training_docids.extend(sampled_docids)
    return training_docids

def get_hexaple_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids):
    '''
        get three times more negative examples than postive examples
    '''
    training_docids = relevant_docids
    num_rel = len(relevant_docids)
    num_irrel = len(irrelevant_docids)
    num_nonannot = len(nonannot_docids)

    #if there is no irrelevant documents, then we choose an equal number of relevant documents 
    # from the non-annotated documents as irrelevant documents
    if num_irrel == 0:
        lowranked_docids = get_docids_lowranked(num_rel, nonannot_docids)
        sampled_docids = lowranked_docids
        #sampled_docids = get_docids_random_sampling(num_rel*8, nonannot_docids)
    else:
        sampled_docids = get_docids_random_sampling(num_rel*8, irrelevant_docids)

    training_docids.extend(sampled_docids)
    return training_docids


def preparing_training_samples(dist_type, query_ids, queryid_docid_rel, queryid_docids, queryid_docid_features, ltr_train_file_path):
    '''
        preparing the training samples based on the distribution type
    '''
    with open(ltr_train_file_path, 'w') as fw:
        for query_id in query_ids:
            print (query_id)
            docid_rel = queryid_docid_rel.get(query_id)
            docids = queryid_docids.get(query_id)
            docid_features = queryid_docid_features.get(query_id)

            if docid_rel is None:
                continue

            #docids = docid_features.keys()
            relevant_docids, irrelevant_docids, nonannot_docids = get_docids_distribution(docids, docid_rel)
            print ("Rel:{}, Irel:{}, Nona:{}".format(len(relevant_docids), len(irrelevant_docids), len(nonannot_docids)))

            if dist_type == 'equal_neg':
                training_docids = get_equal_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids)

            elif dist_type =='double_neg':
                training_docids = get_double_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids)

            elif dist_type =='triple_neg':
                training_docids = get_triple_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids)

            elif dist_type =='quadruple_neg':
                training_docids = get_quadruple_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids)

            elif dist_type =='hexaple_neg':
                training_docids = get_hexaple_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids)

            else:
                training_docids = get_natural_negative_examples(relevant_docids, irrelevant_docids, nonannot_docids)

            qds_rel = []
            qds_features = []
            for doc_id in training_docids:
                rel = docid_rel.get(doc_id)
                qd_feature = docid_features.get(doc_id)
                #features = get_l2_normalize(raw_features)
                qds_features.append(qd_feature)

                #in case of non-annotated documents documents selected as irrelevant documents
                if rel is None:
                    rel = 0
                #in case of negative grade for an irrelevant documents
                if rel < 0:
                    rel = 0

                qds_rel.append(rel)

            #normalize features (not samples, thus, column-wise (0 index))
            qds_features_mat = np.matrix(qds_features)
            qds_features_mat_norm = get_minmax_norm(qds_features_mat, 0)
            #qds_features_mat_norm = qds_features_mat
            qds_features_list_norm = qds_features_mat_norm.tolist()
            #qds_features_list_norm = qds_features_mat.tolist()


            for jdx in range(0, len(qds_rel)):
                rel = qds_rel[jdx]
                features = qds_features_list_norm[jdx]
                doc_id = training_docids[jdx]

                line = str(rel) + " " + "qid:"+str(query_id)
                feature_id = 1
                for idx in range(0, len(features)):
                    feature_val = features[idx]
                    line = line + " " + str(feature_id) + ":"+str(feature_val)
                    feature_id = feature_id + 1
                line = line + " # "+ str(doc_id)
                fw.write(line+'\n')
        #fw.close()


def preparing_testing_samples(query_ids,queryid_docid_rel,queryid_docids,queryid_docid_features,rrank,ltr_test_file_path):
    '''
        preparing the testing samples based on natural distribution
    '''
    with open(ltr_test_file_path, 'w') as fw:
        for query_id in query_ids:
            docid_rel = queryid_docid_rel.get(query_id)
            docids = queryid_docids.get(query_id)
            docid_features = queryid_docid_features.get(query_id)

            if docid_rel is None:
                docid_rel = {}

            qds_rel = []
            qds_features = []

            docids_sel = []
            total_docid = len(docids)
            rrank = min(total_docid, int(rrank))

            for doc_id in docids:
                docids_sel.append(doc_id)
                rel = docid_rel.get(doc_id)
                raw_features = docid_features.get(doc_id)
                #features = get_l2_normalize(raw_features)
                features = raw_features
                qds_features.append(features)

                if rel is None:
                    rel = 0 #dummy relevance for unjudged test data
                if int(rel) < 0:
                    rel = 0
                qds_rel.append(rel)

            qds_features_mat = np.matrix(qds_features)
            qds_features_mat_norm = get_minmax_norm(qds_features_mat, 0)
            qds_features_list_norm = qds_features_mat_norm.tolist()
            #qds_features_list_norm = qds_features_mat.tolist()

            for jdx in range(0, len(docids_sel)):
                rel = qds_rel[jdx]
                features = qds_features_list_norm[jdx]
                doc_id = docids_sel[jdx]

                line = str(rel) + " " + "qid:"+str(query_id)
                feature_id = 1
                for idx in range(0, len(features)):
                    feature_val = features[idx]
                    line = line + " " + str(feature_id) + ":"+str(feature_val)
                    feature_id = feature_id + 1

                line = line + " # "+ str(doc_id)
                fw.write(line+'\n')
        #fw.close()


def prepare_dataset(topics_path,query_doc_features_path,qrels_file_path,dist_type,rrank, dist_file_path,ltr_train_file_path,ltr_test_file_path):
        '''
            Preparing the learning to rank (L2R) datasets for training and testing model
        '''
        topics, query_ids = get_topics(topics_path)
        queryid_docid_rel = get_qrels(qrels_file_path)
        queryid_docids, queryid_docids_bm25, queryid_docid_features = get_query_doc_features(query_doc_features_path)

        #print ("total topics: ", query_ids)

        print('relevance judgement analysis ...')
        #query-document-relevance triple analysis
        query_document_relevance_stats(query_ids, queryid_docid_rel, queryid_docid_features, dist_file_path)
        print ('done')

        print('preparing training samples ...')
        #preparing the training dataset
        preparing_training_samples(dist_type, query_ids, queryid_docid_rel, queryid_docids, queryid_docid_features, ltr_train_file_path)
        print ('done')

        print('preparing testing samples ...')
        #preparing testing samples
        preparing_testing_samples(query_ids, queryid_docid_rel, queryid_docids, queryid_docid_features, rrank, ltr_test_file_path)
        print ('done')


def main():
        topics_path = sys.argv[1]
        query_doc_features_path = sys.argv[2]
        rel_judgment_path = sys.argv[3]
        dist_type = sys.argv[4]
        num_rrank = sys.argv[5]
        dist_path = sys.argv[6]

        training_path = sys.argv[7]
        train_path = training_path + '.' + dist_type + '.' + str(num_rrank)

        test_path = sys.argv[8]
        test_path = test_path + '.' + dist_type + '.' + str(num_rrank)

        print ("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(topics_path, query_doc_features_path, rel_judgment_path,
                                                         dist_type, num_rrank, dist_path, train_path, test_path))
        prepare_dataset(topics_path, query_doc_features_path, rel_judgment_path, dist_type, num_rrank, dist_path,
                        train_path, test_path)

if __name__ == '__main__':
    main()
