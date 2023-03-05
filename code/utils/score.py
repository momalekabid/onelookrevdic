# Adapted from https://github.com/TimotheeMickus/codwoe/blob/main/code/score.py
# and also https://github.com/PinzhenChen/unifiedRevdicDefmod/blob/main/code/utils/score.py
# Author: Malek Abid, 20 feb 2023

import argparse
import collections
import itertools
import json
import logging
import os
import pathlib
import sys
logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(handler)
os.environ["MOVERSCORE_MODEL"] = "distilbert-base-multilingual-cased"
import moverscore_v2 as mv_sc
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk import word_tokenize as tokenize
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
# import check_output



# following are used to evaluate definition generation task
# def bleu(pred, target, smoothing_function=SmoothingFunction().method4):
#     return sentence_bleu([pred], target, smoothing_function=smoothing_function)


# def mover_corpus_score(sys_stream, ref_streams, trace=0):
#     """Adapted from the MoverScore github"""

#     if isinstance(sys_stream, str):
#         sys_stream = [sys_stream]
#     if isinstance(ref_streams, str):
#         ref_streams = [[ref_streams]]
#     fhs = [sys_stream] + ref_streams
#     corpus_score = 0
#     pbar = tqdm.tqdm(desc="MvSc.", disable=None, total=len(sys_stream))
#     for lines in itertools.zip_longest(*fhs):
#         if None in lines:
#             raise EOFError("Source and reference streams have different lengths!")
#         hypo, *refs = lines
#         idf_dict_hyp = collections.defaultdict(lambda: 1.0)
#         idf_dict_ref = collections.defaultdict(lambda: 1.0)
#         corpus_score += mv_sc.word_mover_score(
#             refs,
#             [hypo],
#             idf_dict_ref,
#             idf_dict_hyp,
#             stop_words=[],
#             n_gram=1,
#             remove_subwords=False,
#         )[0]
#         pbar.update()
#     pbar.close()
#     corpus_score /= len(sys_stream)
#     return corpus_score


# def eval_defmod(args, summary):
#     # 1. read contents
#     ## define accumulators for lemma-level BLEU and MoverScore
#     reference_lemma_groups = collections.defaultdict(list)
#     all_preds, all_tgts = [], []
#     ## reading data files
#     with open(args.submission_file, "r") as fp:
#         submission = sorted(json.load(fp), key=lambda r: r["id"])
#     with open(args.reference_file, "r") as fp:
#         reference = sorted(json.load(fp), key=lambda r: r["id"])

#     # 2. compute scores
#     ## compute sense-level BLEU
#     assert len(submission) == len(reference), "Missing items in submission!"
#     id_to_lemma = {}
#     pbar = tqdm.tqdm(total=len(submission), desc="S-BLEU", disable=None)
#     for sub, ref in zip(submission, reference):
#         assert sub["id"] == ref["id"], "Mismatch in submission and reference files!"
#         all_preds.append(sub["gloss"])
#         all_tgts.append(ref["gloss"])
#         sub["gloss"] = tokenize(sub["gloss"])
#         ref["gloss"] = tokenize(ref["gloss"])
#         sub["sense-BLEU"] = bleu(sub["gloss"], ref["gloss"])
#         reference_lemma_groups[(ref["word"], ref["pos"])].append(ref["gloss"])
#         id_to_lemma[sub["id"]] = (ref["word"], ref["pos"])
#         pbar.update()
#     pbar.close()
#     ## compute lemma-level BLEU
#     for sub in tqdm.tqdm(submission, desc="L-BLEU", disable=None):
#         sub["lemma-BLEU"] = max(
#             bleu(sub["gloss"], g)
#             for g in reference_lemma_groups[id_to_lemma[sub["id"]]]
#         )
#     lemma_bleu_average = sum(s["lemma-BLEU"] for s in submission) / len(submission)
#     sense_bleu_average = sum(s["sense-BLEU"] for s in submission) / len(submission)
#     ## compute MoverScore
#     moverscore_average = np.mean(mv_sc.word_mover_score(
#         all_tgts,
#         all_preds,
#         collections.defaultdict(lambda:1.),
#         collections.defaultdict(lambda:1.),
#         stop_words=[],
#         n_gram=1,
#         remove_subwords=False,
#         batch_size=1,
#     ))
#     moverscore_average = mover_corpus_score(all_preds, [all_tgts])
#     # 3. write results.
#     logger.debug(f"Submission {args.submission_file}, \n\tMvSc.: " + \
#         f"{moverscore_average}\n\tL-BLEU: {lemma_bleu_average}\n\tS-BLEU: " + \
#         f"{sense_bleu_average}"
#     )
#     with open(args.output_file, "a") as ostr:
#         print(f"MoverScore_{summary.lang}:{moverscore_average}", file=ostr)
#         print(f"BLEU_lemma_{summary.lang}:{lemma_bleu_average}", file=ostr)
#         print(f"BLEU_sense_{summary.lang}:{sense_bleu_average}", file=ostr)
#     return (
#         args.submission_file,
#         moverscore_average,
#         lemma_bleu_average,
#         sense_bleu_average,
#     )


def rank_cosine_codwoe(preds, targets): # for CODWOE; the lower the better
    assocs = F.normalize(preds) @ F.normalize(targets).T
    refs = torch.diagonal(assocs, 0).unsqueeze(1)
    ranks = (assocs >= refs).sum(1).float()
    assert ranks.numel() == preds.size(0)
    ranks = ranks.mean().item()
    return ranks / preds.size(0)


def rank_cosine_hill(preds, targets): # For Hill eval; the higher the better
    # unique_targets = targets.unique(dim=0)
    assocs = F.normalize(preds) @ F.normalize(targets).T
    # unique_assocs = F.normalize(preds) @ F.normalize(unique_targets).T
    refs = torch.diagonal(assocs, 0).unsqueeze(1)
    ranks = (assocs > refs).sum(1).float()
    ranks[ranks > 100] = 1000 # This follows https://github.com/thunlp/MultiRD/blob/fe72148c00a72eaebcd22e58104e9588dfb72fa4/EnglishReverseDictionary/code/evaluate.py#L26

    return [ranks.mean().item(),
            ranks.median().item(),
            ranks.var(unbiased=False).sqrt().item(),
            sum([n.item() == 0 for n in ranks]) / len(ranks),
            sum([n.item() <= 9 for n in ranks]) / len(ranks),
            sum([n.item() <= 99 for n in ranks]) / len(ranks)
            ]


def eval_revdict(fsubmission, freference, summary = None):
    # 1. read contents
    ## read data files
    with open(fsubmission, "r") as fp:
        submission = sorted(json.load(fp), key=lambda r: r["id"])
    with open(freference, "r") as fp:
        reference = sorted(json.load(fp), key=lambda r: r["id"])
    vec_archs = sorted(
        set(submission[0].keys()) - {
            "id",
            "gloss",
            "word",
            "pos",
            "concrete",
            "example",
            "f_rnk",
            "counts",
            "polysemous",
            "type",
            "bert"
        }
    )
    ## define accumulators for rank-cosine
    all_preds = collections.defaultdict(list)
    all_refs = collections.defaultdict(list)

    assert len(submission) == len(reference), "Missing items in submission!"
    ## retrieve vectors
    for sub, ref in zip(submission, reference):
        assert sub["id"] == ref["id"], "Mismatch in submission and reference files!"
        for arch in vec_archs:
            all_preds[arch].append(sub[arch])
            all_refs[arch].append(ref[arch])

    torch.autograd.set_grad_enabled(False)
    all_preds = {arch: torch.tensor(all_preds[arch]) for arch in vec_archs}
    all_refs = {arch: torch.tensor(all_refs[arch]) for arch in vec_archs}

    # 2. compute scores
    MSE_scores = {
        arch: F.mse_loss(all_preds[arch], all_refs[arch]).item() for arch in vec_archs
    }
    cos_scores = {
        arch: F.cosine_similarity(all_preds[arch], all_refs[arch]).mean().item()
        for arch in vec_archs
    }
    print("Please check if you are using the correct scoring func for HILL or CODWOE")
    rnk_scores = {
        arch: rank_cosine_hill(all_preds[arch], all_refs[arch]) for arch in vec_archs
    }
    # 3. display results
    logger.debug(f"Submission {fsubmission}, \n\tMSE: " + \
        ", ".join(f"{a}={MSE_scores[a]}" for a in vec_archs) + \
        # ", \n\tcosine: " + \
        # ", ".join(f"{a}={cos_scores[a]}" for a in vec_archs) + \
        ", \n\tcosine ranks: " + \
        ", ".join(f"{a}={rnk_scores[a]}" for a in vec_archs) + \
        "."
    )
    all_archs = sorted(set(reference[0].keys()) - {"id", "gloss", "word", "pos"})
    with open("score_output.txt", "a") as ostr:
        for arch in vec_archs:
            print(arch + ":", file=ostr)
            print(f"MSE_{summary}_{arch}:{MSE_scores[arch]}", file=ostr)
            print(f"cos_{summary}_{arch}:{cos_scores[arch]}", file=ostr)
            print(f"rnk_{summary}_{arch}:{rnk_scores[arch]}", file=ostr)
    return (
        fsubmission,
        *[MSE_scores.get(a, None) for a in vec_archs],
        *[cos_scores.get(a, None) for a in vec_archs],
    )


def main(run, dir = "../../test/",):
    label_list_fname = dir + run + "_label_list.json"
    pred_list_fname = dir + run + "_pred_list.json"
    print('load file : f"{label_list_fname}"\n')
    print('load file : f"{pred_list_fname}"\n')
    eval_revdict(label_list_fname, pred_list_fname, "revdict")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default='unseen')
    args = parser.parse_args()
    main(args.type)
