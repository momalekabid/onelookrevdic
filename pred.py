import sys
import gensim
import json
import numpy as np
# given a file of predictions (jsons), for each entry look at predicted embedding and find top 1, 3, 10 similar words with 
# loaded word2vec model OR using cosine_similarity on contextual embeddings. 

model_path = "GoogleNews-vectors-negative300.bin" # target words were embedded using this model

model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
print('w2v model loaded')

f = open("code/preds/predunseenbert.json")
items = json.load(f)
for item in items:
    unseen_vector = item['bert']
    unseen_vector = list(map(float, unseen_vector))
    unseen_vector = np.array(unseen_vector)
    results = model.most_similar(positive=[unseen_vector], topn=10)
    break

print(results)


# def rank_cosine_hill(preds, targets): # For Hill eval; the higher the better
#     # unique_targets = targets.unique(dim=0)
#     assocs = F.normalize(preds) @ F.normalize(targets).T
#     # unique_assocs = F.normalize(preds) @ F.normalize(unique_targets).T
#     refs = torch.diagonal(assocs, 0).unsqueeze(1)
#     ranks = (assocs > refs).sum(1).float()
#     ranks[ranks > 100] = 1000 # This follows https://github.com/thunlp/MultiRD/blob/fe72148c00a72eaebcd22e58104e9588dfb72fa4/EnglishReverseDictionary/code/evaluate.py#L26

#     return [ranks.mean().item(),
#             ranks.median().item(),
#             ranks.var(unbiased=False).sqrt().item(),
#             sum([n.item() == 0 for n in ranks]) / len(ranks),
#             sum([n.item() <= 9 for n in ranks]) / len(ranks),
#             sum([n.item() <= 99 for n in ranks]) / len(ranks)
#             ]


# def eval_revdict(args, summary):
#     # 1. read contents
#     ## read data files
#     with open(args.submission_file, "r") as fp:
#         submission = sorted(json.load(fp), key=lambda r: r["id"])
#     with open(args.reference_file, "r") as fp:
#         reference = sorted(json.load(fp), key=lambda r: r["id"])
#     vec_archs = sorted(
#         set(submission[0].keys())
#         - {
#             "id",
#             "gloss",
#             "word",
#             "pos",
#             "concrete",
#             "example",
#             "f_rnk",
#             "counts",
#             "polysemous",
#             "type"
#         }
#     )
#     ## define accumulators for rank-cosine
#     all_preds = collections.defaultdict(list)
#     all_refs = collections.defaultdict(list)

#     assert len(submission) == len(reference), "Missing items in submission!"
#     ## retrieve vectors
#     for sub, ref in zip(submission, reference):
#         assert sub["id"] == ref["id"], "Mismatch in submission and reference files!"
#         for arch in vec_archs:
#             all_preds[arch].append(sub[arch])
#             all_refs[arch].append(ref[arch])

#     torch.autograd.set_grad_enabled(False)
#     all_preds = {arch: torch.tensor(all_preds[arch]) for arch in vec_archs}
#     all_refs = {arch: torch.tensor(all_refs[arch]) for arch in vec_archs}

#     # 2. compute scores
#     MSE_scores = {
#         arch: F.mse_loss(all_preds[arch], all_refs[arch]).item() for arch in vec_archs
#     }
#     cos_scores = {
#         arch: F.cosine_similarity(all_preds[arch], all_refs[arch]).mean().item()
#         for arch in vec_archs
#     }
#     print("Please check if you are using the correct scoring func for HILL or CODWOE")
#     rnk_scores = {
#         arch: rank_cosine_hill(all_preds[arch], all_refs[arch]) for arch in vec_archs
#     }
#     # 3. display results
#     logger.debug(f"Submission {args.submission_file}, \n\tMSE: " + \
#         ", ".join(f"{a}={MSE_scores[a]}" for a in vec_archs) + \
#         ", \n\tcosine: " + \
#         ", ".join(f"{a}={cos_scores[a]}" for a in vec_archs) + \
#         ", \n\tcosine ranks: " + \
#         ", ".join(f"{a}={rnk_scores[a]}" for a in vec_archs) + \
#         "."
#     )
#     all_archs = sorted(set(reference[0].keys()) - {"id", "gloss", "word", "pos"})
#     with open(args.output_file, "a") as ostr:
#         for arch in vec_archs:
#             print(arch + ":", file=ostr)
#             print(f"MSE_{summary.lang}_{arch}:{MSE_scores[arch]}", file=ostr)
#             print(f"cos_{summary.lang}_{arch}:{cos_scores[arch]}", file=ostr)
#             print(f"rnk_{summary.lang}_{arch}:{rnk_scores[arch]}", file=ostr)
#     return (
#         args.submission_file,
#         *[MSE_scores.get(a, None) for a in vec_archs],
#         *[cos_scores.get(a, None) for a in vec_archs],
#     )
