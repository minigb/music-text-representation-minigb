import os
import json
import pickle
import torch
from torch import nn
import numpy as np
import pandas as pd
import IPython.display as ipd
from IPython.display import Audio, HTML

import argparse
from mtr.utils.demo_utils import get_model
from mtr.utils.eval_utils import _text_representation
import warnings
warnings.filterwarnings(action='ignore')

# check https://github.com/seungheondoh/msd-subsets
# your_msd_path = "dataset"
# msd_path = os.path.join(your_msd_path, "msd-subsets/dataset")
msd_path = '/home/minhee/userdata/music-text-representation-minigb/dataset'

global msd_to_id
global id_to_path

msd_to_id = pickle.load(open(os.path.join(msd_path, "lastfm_annotation", "MSD_id_to_7D_id.pkl"), 'rb'))
id_to_path = pickle.load(open(os.path.join(msd_path, "lastfm_annotation", "7D_id_to_path.pkl"), 'rb'))
annotation = json.load(open(os.path.join(msd_path, "ecals_annotation/annotation.json"), 'r'))


def pre_extract_audio_embedding(framework, text_type, text_rep):
    ecals_test = torch.load(f"../mtr/{framework}/exp/transformer_cnn_cf_mel/{text_type}_{text_rep}/audio_embs.pt")
    msdid = [k for k in ecals_test.keys()]
    audio_embs = [ecals_test[k] for k in msdid]
    audio_embs = torch.stack(audio_embs)
    return audio_embs, msdid


def retrieve(framework, text_type, text_rep, annotation, query_list):
    # get audio embedding info
    audio_embs, msdid = pre_extract_audio_embedding(framework, text_type, text_rep)
    # get pretrained model
    model, tokenizer, _ = get_model(framework=framework, text_type=text_type, text_rep=text_rep)

    meta_results = []
    for query in query_list:
        text_input = tokenizer(query, return_tensors="pt")['input_ids']
        with torch.no_grad():
            text_embs = model.encode_bert_text(text_input, None)
        
        audio_embs = nn.functional.normalize(audio_embs, dim=1)
        text_embs = nn.functional.normalize(text_embs, dim=1)
        logits = text_embs @ audio_embs.T
        ret_item = pd.Series(logits.squeeze(0).numpy(), index=msdid).sort_values(ascending=False)
        
        metadata = {}
        for idx, _id in enumerate(ret_item.sort_values(ascending=False).head(3).index):
            meta_obj = annotation[_id]
            metadata[f'top{idx+1} music'] = meta_obj['tag']

        meta_results.append(metadata)
    return meta_results


if __name__ == "__main__":
    tag_query = "banjo"
    query = [tag_query]

    framework='contrastive' # triplet
    text_type='bert' # tag, caption
    text_rep="stochastic"

    retrieve(framework, text_type, text_rep, annotation, query)