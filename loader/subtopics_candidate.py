import os
import sys


def get_reciprocal_rank(ranks):

        weight = 0.0
        for rank in ranks:
            weight = weight + 1.0/float(rank)
        return weight


def weight_subtopics(topicid_subtopics_metadata):

        topicid_subtopics_weight = {}

        #print ('start')
        topic_ids = list(topicid_subtopics_metadata.keys())
        topic_ids_int = [int(topic_id) for topic_id in topic_ids]
        stopic_ids = sorted(topic_ids_int)
        #print (stopic_ids)

        for topic_id in stopic_ids:
            subtopics_metadata = topicid_subtopics_metadata.get(str(topic_id))
            subtopics_list = list(subtopics_metadata.keys())
            #print (topic_id)
            subtopics_weight = {}

            for subtopic in subtopics_list:

                metadata = subtopics_metadata.get(subtopic)
                weight = get_reciprocal_rank(metadata)
                subtopics_weight[subtopic] = weight
                #print (str(subtopic) + '\t' + str(weight))
            topicid_subtopics_weight[str(topic_id)] = subtopics_weight
        #print ('done')
        return topicid_subtopics_weight


def get_subtopics(candidates_path):

        #print('start')
        topicid_query = {}
        topicid_subtopics_metadata = {}
        path = candidates_path
        with open(path, 'r') as reader:
            lines = reader.readlines()
            reader.close()

        for line in lines:
            line_parts = line.lower().split('\t')
            topic_id = line_parts[0]
            topic_txt = line_parts[1]
            subtopic = line_parts[2]
            rank = line_parts[3]
            source = line_parts[4]

            query = topicid_query.get(topic_id)
            if query is None:
                topicid_query[topic_id] = topic_txt

            subtopics_metadata = topicid_subtopics_metadata.get(topic_id)
            if subtopics_metadata is None:
                subtopics_metadata = {}

            ranks = subtopics_metadata.get(subtopic)
            if ranks is None:
                ranks = []

            ranks.append(rank)
            subtopics_metadata[subtopic] = ranks

            topicid_subtopics_metadata[topic_id] = subtopics_metadata
        #print('done')
        topic_subtopic_reciprocal_rank = weight_subtopics(topicid_subtopics_metadata)
        return topic_subtopic_reciprocal_rank


def main():

        folder_path = 'input/subtopics'
        subtopics_name = 'trec_webtrack_2009_2012_suggestions.txt'
        subtopics_path = os.path.join(folder_path, subtopics_name)
        get_subtopics(subtopics_path)

main()
