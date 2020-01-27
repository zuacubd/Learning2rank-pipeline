import os
import sys

def get_run_result(path):

        with open(path, 'r') as fread:
            lines = fread.readlines()

        queryid_docid_rankscore = {}
        for idx in range(1, len(lines)):
            line = lines[idx]
            parts = line.rstrip().split(" ")

            #1 0 clueweb09-en0003-55-31884 0
            query_id = parts[0]
            doc_id = parts[2]
            rank = parts[3]
            score = parts[4]

            docid_rankscore = queryid_docid_rankscore.get(query_id)
            if docid_rankscore is None:
                docid_rankscore = {}
            docid_rankscore[doc_id] = [rank, score]
            queryid_docid_rankscore[query_id] = docid_rankscore

        return queryid_docid_rankscore
