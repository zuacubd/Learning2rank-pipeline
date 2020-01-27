import os
import sys

def get_topics(topic_path):

        #print('start')
        topic_ids = []
        topicid_txt = {}       

        path = topic_path
        with open(path, 'r') as reader:

            lines = reader.readlines()
            reader.close()

        for line in lines:
            line_parts = line.strip().split(':')
            topic_id = line_parts[0]
            txt = line_parts[1]

            topic_ids.append(topic_id)
            topicid_txt[topic_id] = txt

        return topic_ids, topicid_txt


def main():

        folder_path = 'input/topics'
        topics_file_name = 'wt09_12.topics.queries-only.txt' 
        topics_path = os.path.join(folder_path, topics_file_name)

        topics = get_topics(topics_path)
        #print (topics)


main()
