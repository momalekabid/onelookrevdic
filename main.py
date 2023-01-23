## malek abid dec 4th 2022
## onelook thesaurus revdic model data pre-processing
import sys
sys.path.append("./code/utils")
from data_process import *

file_types = ["train", "dev", "seen", "unseen", "desc"]


def process_data():
    print("Processing data:")
    ftype = file_types[4]
    file = f"./data/wwdata/{ftype}.json"
    out_file  = f"./data/wwdata/{ftype}2.json"
    process_tsinghua_data(file, out_file)

    print("Linguistic annotations successfully stripped")
    file = f"./data/wwdata/{ftype}2.json"
    out_file  = f"./data/wwdata/{ftype}3.json"
    add_fasttext_embedding(file, out_file)
    print("Fasttext embeddings completed")


    # file = f"./data/wwdata/{ftype}2.json"
    # out_file  = f"./data/wwdata/{ftype}3.json"
    # add_word2vec_embedding(file, out_file)
    # print("Word2vec embeddings completed.")

    file = f"./data/wwdata/{ftype}3.json"
    out_file  = f"./data/wwdata/{ftype}4.json"
    add_bert_embedding_from_def(file, out_file, ftype)
    print("BERT embeddings completed.")

    file = f"./data/wwdata/{ftype}4.json"
    out_file  = f"./data/wwdata/{ftype}5.json"
    filter_hill_data(file, out_file)
    print("Duplicates cleaned successfully")

    file = f"./data/wwdata/{ftype}5.json"
    # out_file  = f"./data/wwdata/{ftype}_label_list.json"
    check_data(file)
    print("Complete.")

process_data()
