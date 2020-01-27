import os
import sys

def get_qrels(path):

        with open(path, 'r') as fread:
            lines = fread.readlines()

        queryid_docid_rel = {}
        for idx in range(1, len(lines)):
            line = lines[idx]
            parts = line.rstrip().split(" ")

            #1 0 clueweb09-en0003-55-31884 0
            query_id = parts[0]
            doc_id = parts[2]
            rel = parts[3]

            docid_rel = queryid_docid_rel.get(query_id)
            if docid_rel is None:
                docid_rel = {}
            docid_rel[doc_id] = rel
            queryid_docid_rel[query_id] = docid_rel

        return queryid_docid_rel
