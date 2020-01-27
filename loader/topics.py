import os
import sys


def get_topics(topic_path):
        topicid_txt = {}
        topics_id = []

        with open(topic_path, 'r') as reader:

            lines = reader.readlines()
            reader.close()

        for line in lines:
            line_parts = line.strip().split(':')
            topicid = line_parts[0]
            txt = line_parts[1]

            topicid_txt[topicid] = txt
            topics_id.append(topicid)

        return topicid_txt, topics_id

