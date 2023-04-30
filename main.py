## malek abid dec 4th 2022 
## last edit april 30 2023
## onelook thesaurus revdic model data pre-processing
import sys
sys.path.append("./code/utils")
from data_process import *

file_types = ["train", "dev", "seen", "unseen", "desc"]


def process_data(ftype):
    print("Processing data:")
    file = f"./data/wwdata/{ftype}.json"
    out_file  = f"./data/wwdata/{ftype}_stripped.json"
    process_tsinghua_data(file, out_file)
    print("Linguistic annotations successfully stripped")

#   file = f"./data/wwdata/{ftype}_stripped.json"
#   out_file  = f"./data/wwdata/{ftype}_ft.json"
#   add_fasttext_embedding(file, out_file)
#   print("Fasttext embeddings completed")

    file = f"./data/wwdata/{ftype}_stripped.json"
    out_file  = f"./data/wwdata/{ftype}_w2v.json"
    add_word2vec_embedding(file, out_file, ftype)
    print("Word2vec embeddings completed.") # w2v works best for revdict

#   file = f"./data/wwdata/{ftype}_stripped.json"
#   out_file  = f"./data/wwdata/{ftype}_bert.json"
#   add_bert_embedding_from_def(file, out_file, ftype)
#   print("BERT embeddings completed.") # BERT for definition modelling

    file = f"./data/wwdata/{ftype}_w2v.json"
    out_file  = f"./data/wwdata/{ftype}_label_list.json"
    
    filter_hill_data(file, out_file)
    print("Duplicates cleaned successfully")

    try:
        check_data(out_file)
    except:
        print("Some errors^")
    print("Complete.")

#   process_data(file_types[0]) # train
#   process_data(file_types[1]) # dev
#   process_data(file_types[2]) # seen
#   process_data(file_types[3]) # unseen
#   process_data(file_types[4]) # human desc
