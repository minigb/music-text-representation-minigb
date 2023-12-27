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
import warnings

from mtr.utils.demo_utils import get_model
from mtr.utils.eval_utils import _text_representation
from mtr.utils.demo_utils import get_model
from mtr.utils.audio_utils import load_audio, STR_CH_FIRST

warnings.filterwarnings(action='ignore')


msd_path = '/home/minhee/userdata/music-text-representation-minigb/dataset'

global msd_to_id
global id_to_path

msd_to_id = pickle.load(open(os.path.join(msd_path, "lastfm_annotation", "MSD_id_to_7D_id.pkl"), 'rb'))
id_to_path = pickle.load(open(os.path.join(msd_path, "lastfm_annotation", "7D_id_to_path.pkl"), 'rb'))
annotation = json.load(open(os.path.join(msd_path, "ecals_annotation/annotation.json"), 'r'))


# framework='contrastive' 
# text_type='bert'
# text_rep="stochastic"
# # load model
# model, tokenizer, config = get_model(framework=framework, text_type=text_type, text_rep=text_rep)

def text_infer(query, model, tokenizer):
    text_input = tokenizer(query, return_tensors="pt")['input_ids']
    with torch.no_grad():
        text_embs = model.encode_bert_text(text_input, None)
    return text_embs


def audio_infer(audio_path, model, sr=16000, duration=9.91):
    audio, _ = load_audio(
            path=audio_path,
            ch_format= STR_CH_FIRST,
            sample_rate= sr,
            downmix_to_mono= True
    )
    input_size = int(duration * sr)
    hop = int(len(audio) // input_size)
    audio = np.stack([np.array(audio[i * input_size : (i + 1) * input_size]) for i in range(hop)]).astype('float32')
    audio_tensor = torch.from_numpy(audio)
    with torch.no_grad():
        z_audio = model.encode_audio(audio_tensor)
    audio_embs = z_audio.mean(0).detach().cpu()
    return audio_embs

# query = "fusion jazz with synth, bass, drums, saxophone"
# audio_path = "your_audio"
# text_embs = text_infer(query, model, tokenizer)
# audio_embs = audio_infer(audio_path, model)


def pre_extract_audio_embedding(audio_path, framework, text_type, text_rep):
    ecals_test = torch.load(f"../mtr/{framework}/exp/transformer_cnn_cf_mel/{text_type}_{text_rep}/audio_embs.pt")
    msdid = [k for k in ecals_test.keys()]
    audio_embs = [ecals_test[k] for k in msdid]
    audio_embs = torch.stack(audio_embs)
    return audio_embs, msdid

def model_load(framework, text_type, text_rep):
    audio_embs, msdid = pre_extract_audio_embedding(framework, text_type, text_rep)
    model, tokenizer, config = get_model(framework=framework, text_type=text_type, text_rep=text_rep)
    return model, audio_embs, tokenizer, msdid

def retrieval_fn(query, tokenizer, model, audio_embs, msdid, annotation):
    text_input = tokenizer(query, return_tensors="pt")['input_ids']
    with torch.no_grad():
        text_embs = model.encode_bert_text(text_input, None)
    audio_embs = nn.functional.normalize(audio_embs, dim=1)
    text_embs = nn.functional.normalize(text_embs, dim=1)
    logits = text_embs @ audio_embs.T
    ret_item = pd.Series(logits.squeeze(0).numpy(), index=msdid)
    instance = {}
    metadata = {}
    for idx, _id in enumerate(ret_item.sort_values(ascending=False).head(3).index):
        music_obj = ipd.Audio(os.path.join(msd_path, 'songs', id_to_path[msd_to_id[_id]]) , rate=22050)
        meta_obj = annotation[_id]
        metadata[f'top{idx+1} music'] = meta_obj['tag']
        music_src = music_obj.src_attr()
        instance[f'top{idx+1} music'] = f"""<audio controls><source src="{music_src}" type="audio/wav"></audio></td>"""
    return instance, metadata

def retrieval_show(framework, text_type, text_rep, annotation, query, is_audio=True):    
    model, audio_embs, tokenizer, msdid = model_load(framework, text_type, text_rep)
    meta_results, retrieval_results = [], []
    for i in query:
        instance, metadata = retrieval_fn(i, tokenizer, model, audio_embs, msdid, annotation)
        retrieval_results.append(instance)
        meta_results.append(metadata)
    if is_audio:
        inference = pd.DataFrame(retrieval_results, index=query)
        html = inference.to_html(escape=False)
    else:
        inference = pd.DataFrame(meta_results, index=query)
        html = inference.to_html(escape=False)
    ipd.display(HTML(html))

if __name__ == "__main__":
    tag_query = "banjo"
    caption_query = "fusion jazz with synth, bass, drums, saxophone"
    unseen_query = "music for meditation or listen to in the forest"
    query = [tag_query, caption_query, unseen_query]

    framework='classification' # triplet
    text_type='bert' # tag, caption
    text_rep="stochastic"
    
    retrieval_show(framework, text_type, text_rep, annotation, query, is_audio=False)
