## malek abid dec 4th 2022
## onelook thesaurus revdic model data pre-processing
import sys
sys.path.append("./code/utils")
from data_process import *

file = "./data/wantWordsdata/en/desc.json"
out_file  = "./data/wantWordsdata/en/desc1.json"
process_tsinghua_data(file, out_file)

file = "./data/wantWordsdata/en/desc1.json"
out_file = "./data/wantWordsdata/en/desc2.json"
add_bert_embedding_from_def(file, out_file, "dev")

file = "./data/wantWordsdata/en/desc2.json" 
out_file = "./data/wantWordsdata/en/desc3.json"
add_word2vec_embedding(file, out_file)

file = "./data/wantWordsdata/en/desc3.json"
out_file = "./data/wantWordsdata/en/desc_final.json"
filter_hill_data(file, out_file)
# with open(out_file, 'r') as f:
#     items = json.load(f)
#     print("JSON loaded")
#     for item in items:
#         item['id'] = item['id'].replace("dev", "train")
#     file_out = "./data/wantWordsdata/en/train_final_fixed.json"
#     with open(file_out, 'w') as wp:
#                     json.dump(items, wp)
check_data(out_file)


