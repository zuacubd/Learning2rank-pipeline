import os
import sys
import operator

from loader.topics import *
from loader.features import *
from loader.qrels import *
from tools.normalization import *

def get_topics_list(path):
        with open(path, 'r') as fr:
            lines = fr.readlines()
            fr.close()

        topics = [line.rstrip() for line in lines]
        return topics

def align_topics_docid_mlscore(learner, test_data, test_pred):
        topics_docid_score = {}
        with open(test_data, 'r') as fr:
            test_data_lines = fr.readlines()
            fr.close()

        with open(test_pred, 'r') as fr:
            test_pred_lines = fr.readlines()
            fr.close()

        for itr in range(len(test_data_lines)):
            line = test_data_lines[itr]
            parts = line.rstrip().split("#")

            part = parts[0]
            sub_part = part.split(" ")
            q_qid = sub_part[1]

            qid = q_qid.split(":")[1]
            docid = parts[1].strip()
            pred_line = test_pred_lines[itr].rstrip()

            if learner == "svm":
                pred = pred_line
            else:
                pred = pred_line.split("\t")[2]

            docid_score = topics_docid_score.get(qid)
            if docid_score is None:
                docid_score = {}
            docid_score[docid] = float(pred)
            topics_docid_score[qid] = docid_score

        return topics_docid_score

def prepare_ml_ranked(cmt_path, dist, rrank, learner, nfolds, input_folder, output_folder, run_folder):

        run_path = os.path.join(run_folder, cmt_path+'_'+ dist + '_' + str(rrank) + '_' + learner+'_reranked.res')
        with open(run_path, 'w') as fw:
            for f in range(1, (nfolds+1)):
                topics_fold_path = os.path.join(input_folder, 'f'+str(f))
                topics_id = get_topics_list(topics_fold_path)
                test_data=os.path.join(output_folder,'test/'+cmt_path+'_query.ltr.test.'+dist+'.'+str(rrank)+'.f'+str(f)+'te')
                test_pred=os.path.join(output_folder,'test/'+cmt_path+'_query.ltr.test.'+dist+'.'+str(rrank)+'.f'+str(f)+'te.'+learner+'.pred')

                #aligning prediction score of test data
                topics_docid_score = align_topics_docid_mlscore(learner, test_data, test_pred)

                #writing the run file
                for topic_id in topics_id:
                    docid_score = topics_docid_score.get(topic_id)
                    #sorted_docid_score = sorted(docid_score, key=sortSecond, reverse=True)
                    #rank = 1
                    #for docid, score in sorted_docid_score:
                    #    line = str(topic_id)+' '+'Q0'+' '+str(docid)+' '+str(rank)+' '+str(score)+' svm-rank\n'
                    #    rank = rank + 1
                    #    fw.write(line)
                    #
                    if docid_score is None:
                        print (topic_id)
                        continue
                    sorted_docids = sorted(docid_score, key=docid_score.get, reverse=True)
                    rank = 1
                    for docid in sorted_docids:
                        score = docid_score.get(docid)
                        line = str(topic_id)+' '+'Q0'+' '+str(docid)+' '+str(rank)+' '+str(score)+' '+learner+'\n'
                        rank = rank + 1
                        fw.write(line)
            fw.close()

def sortSecond(val):
        return val[1]

def main():
        coll = sys.argv[1]
        model = sys.argv[2]
        topics = sys.argv[3]
        dist = sys.argv[4]
        rrank = sys.argv[5]
        learner = sys.argv[6]
        #nfolds = int(sys.argv[5])
        nfolds = 5
        #input_folder = sys.argv[6]
        input_folder = "input/data"
        #output_folder = sys.argv[7]
        output_folder = "output/l2r-dataset"
        #run_folder = sys.argv[8]
        run_folder = "output/runs"

        cmt_path = coll + '_' + model + '_' + topics
        prepare_ml_ranked(cmt_path, dist, rrank, learner, nfolds, input_folder, output_folder, run_folder)

if __name__ == '__main__':
        main()
