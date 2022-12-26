## malek abid dec 4th 2022
## onelook thesaurus revdic model data pre-processing
import sys
sys.path.append("./code/utils")
from data_process import *

# file = "./data/wantWordsdata/en/desc.json"
# out_file  = "./data/wantWordsdata/en/desc1.json"
# process_tsinghua_data(file, out_file)

# file = "./data/wantWordsdata/en/desc1.json"
# out_file = "./data/wantWordsdata/en/desc2.json"
# add_bert_embedding_from_def(file, out_file, "dev")

# file = "./data/wantWordsdata/en/desc2.json" 
# out_file = "./data/wantWordsdata/en/desc3.json"
# add_word2vec_embedding(file, out_file)

# file = "./data/wantWordsdata/en/desc3.json"
# out_file = "./data/wantWordsdata/en/desc_final.json"
# filter_hill_data(file, out_file)
# check_data(out_file)


def process_data(in_file, file):
    print("Processing data:")
    process_tsinghua_data(in_file, file)  #first, remove glosses
    add_bert_embedding_from_def(file, file, "dev") #if pre-processing train, change third arg to "train"
    add_word2vec_embedding(file, file) # add target embedding
    filter_hill_data(file, file)
    check_data(file)
    print("Complete.")
