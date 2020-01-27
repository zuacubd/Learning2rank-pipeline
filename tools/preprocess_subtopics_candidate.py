import os
import sys


def aggregated_subtopics(topicid_source_suggestions):

        #print ('start')
        topic_ids = list(topicid_source_suggestions.keys())
        stopic_ids = sorted(topic_ids)

        for topic_id in stopic_ids:
            source_suggestions = topicid_source_suggestions.get(topic_id)
            source_list = list(source_suggestions.keys())
            
            topic_number = int(topic_id) - 250

            for source in source_list:

                suggestions = source_suggestions.get(source)
                rank = 1
                query = suggestions[0]

                for s in range(1, len(suggestions)):
                    
                    suggestion = suggestions[s]
                    print (str(topic_number) + '\t' + query + '\t' + suggestion + '\t' + str(rank) + '\t' + source)
                    rank = rank + 1
        #print ('done')


def aggregate_suggestion_candidate(folder_path, files):

        #print('start')
        topicid_source_suggestions = {}       
 
        for file_name in files:

            path = os.path.join(folder_path, file_name)
            with open(path, 'r') as reader:
                lines = reader.readlines()
                reader.close()
            
            name = file_name.split('.t')[0]
            parts = name.split('_')
            source = parts[2] + '_' + parts[3]

            for line in lines:
                line_parts = line.split(';')
                topicid = line_parts[0]

                suggestions = []
                for s in range(1, len(line_parts)):
                    suggestion = line_parts[s].strip()
                    suggestions.append(suggestion)
                
                source_suggestions = topicid_source_suggestions.get(topicid)
                if source_suggestions is None:
                    source_suggestions = {}

                suggests = source_suggestions.get(source)
                if suggests is None:
                    suggests = []

                suggests = suggests + suggestions
                source_suggestions[source] = suggests
                topicid_source_suggestions[topicid] = source_suggestions
        
        #print('done')
        aggregated_subtopics(topicid_source_suggestions)



def main():

        folder_path = 'input/subtopics'
        source_list = ['ntcir10_intent2_bing_completion.txt', 'ntcir10_intent2_bing_suggestion.txt', 'ntcir10_intent2_google_completion.txt', 'ntcir10_intent2_yahoo_completion.txt']
        
        aggregate_suggestion_candidate(folder_path, source_list)



main()
