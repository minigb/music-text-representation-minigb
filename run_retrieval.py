# Modified the provided demo.ipynb
import torch
from torch import nn
import numpy as np
import pandas as pd
import warnings
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import json
from tqdm import tqdm
import os

from mtr.utils.demo_utils import get_model
from utils.embeddings import audio_infer, text_infer, pre_extract_audio_embedding, pre_extract_text_embedding
from utils.dataset import MusicCaps, SongDescriber


def get_sim(audio_embs_dict, text_emb) -> torch.Tensor:
    audio_embs = torch.stack([audio_embs_dict[k] for k in audio_embs_dict.keys()])
    
    audio_embs = nn.functional.normalize(audio_embs, dim=1)
    text_emb = nn.functional.normalize(text_emb, dim=-1)
    logits = text_emb @ audio_embs.T
    
    return logits


def get_audio_embed_dict(audio_embs_path, dataset) -> dict:
    if not audio_embs_path.exists():
        audio_embs_dict = pre_extract_audio_embedding(
            [dataset.get_identifier(i) for i in range(len(dataset))],
            [audio_dir / dataset[i]['path'] for i in range(len(dataset))],
            model,)
        torch.save(audio_embs_dict, audio_embs_path)
    else:
        audio_embs_dict = torch.load(audio_embs_path)
    
    assert len(dataset) == len(audio_embs_dict), f'{len(dataset)} != {len(audio_embs_dict)}'
    return audio_embs_dict


def get_tag_list(tag_path, dataset) -> list:
    assert tag_path.exists()

    with open(tag_path, 'r') as f:
        tag_list = json.load(f)
    tag_list = [tag_list[i] for i in list(dataset.df.index)]
    
    assert len(dataset) == len(tag_list), f'{len(dataset)} != {len(tag_list)}'
    
    return tag_list


def get_text_embed_dict(text_embs_path, dataset) -> dict:
    if not text_embs_path.exists():
        text_embs_dict = pre_extract_text_embedding(
            set([tag for tags in dataset.df['tag'] for tag in tags ]),
            model,
            tokenizer)
        torch.save(text_embs_dict, text_embs_path)
    else:
        text_embs_dict = torch.load(text_embs_path)
    
    assert len(text_embs_dict) == len(set([tag for tags in dataset.df['tag'] for tag in tags ])),\
        f'{len(text_embs_dict)} != {len(set([tag for tags in dataset.df["tag"] for tag in tags ]))}'

    return text_embs_dict


def calculate_and_save_sim(sim_dir, dataset, audio_embs_dict, text_embs_dict) -> None:
    if sim_dir.exists():
        assert len(os.listdir(sim_dir)) == len(dataset), f'{len(os.listdir(sim_dir))} != {len(dataset)}'
        return
    
    os.makedirs(sim_dir, exist_ok=True)
    for i, cur_tags in tqdm(enumerate(dataset.df['tag'])): # cur_tags is a list of string tags
        query_list = cur_tags + [dataset[i]['caption']]
        sim_result = {}

        identifier = dataset.get_identifier(i)
        result_save_path = sim_dir / f'{identifier}.pt'
        
        for query in query_list:
            if query in text_embs_dict.keys():
                text_emb = text_embs_dict[query]
            else:
                text_emb = text_infer(query, model, tokenizer)

            sim = get_sim(audio_embs_dict, text_emb)

            sim_result[query] = sim
        
        torch.save(sim_result, result_save_path)

    return


def calculate_and_save_rank(result_dir, dataset, audio_embs_dict, sim_dir, portion_list):
    if not result_dir.exists():
        os.makedirs(result_dir, exist_ok=True)

    for portion in portion_list:
        rank_dict = {}
        if portion == 'random':
            # this will be set randomly in calculate_recall_at_k
            continue
        else:
            recall_result_path = Path(result_dir/f'{portion}.json')
            if not recall_result_path.exists():
                for i in tqdm(range(len(dataset))):
                    identifier = dataset.get_identifier(i)
                    sim_result_path = sim_dir / f'{identifier}.pt'
                    sim_result = torch.load(sim_result_path)

                    # stack all
                    sim_result = torch.stack(list(sim_result.values()), axis=0)
                        
                    # average
                    if portion == -1:
                        sim_avg = sim_result.mean(axis=0)
                    else:
                        sim_avg = portion * sim_result[:-1].mean(axis=0) + (1 - portion) * sim_result[-1]
                    ret_item = pd.Series(sim_avg.squeeze(0).numpy(), index=audio_embs_dict.keys()).sort_values(ascending=False)
                    rank = ret_item.index.get_loc(identifier)
                    rank_dict[str(identifier)] = rank
                
                with open(recall_result_path, 'w') as f:
                    json.dump(rank_dict, f, indent = 4)

            assert recall_result_path.exists()
            with open(recall_result_path, 'r') as f:
                rank_dict = json.load(f)
            assert len(dataset) == len(rank_dict), f'{len(dataset)} != {len(rank_dict)}'


def calculate_recall_at_k(result_dir, dataset, k_list, portion_list):
    for portion in portion_list:
        rank_dict = {}
        if portion == 'random':
            for i in range(len(dataset)):
                identifier = dataset.get_identifier(i)
                rank_dict[str(identifier)] = np.random.randint(0, len(dataset) - 1)
        else:
            recall_result_path = Path(result_dir/f'{portion}.json')
            assert recall_result_path.exists()
            with open(recall_result_path, 'r') as f:
                rank_dict = json.load(f)
        assert len(dataset) == len(rank_dict), f'{len(dataset)} != {len(rank_dict)}'

        for k in k_list:
            recall = 0
            for identifier in rank_dict.keys():    
                if rank_dict[identifier] < k:
                    recall += 1
        #TODO(minigb): Save the result instead of printing
        print(f'portion: {portion}')
        print(recall)
        print(f'recall@{k}: {recall/len(dataset)*100:.2f}')
        print('-' * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type = str, required = True)
    parser.add_argument("--dataset_config_path", type = str, required = True)
    parser.add_argument("--dataset_name", type = str, required = True)
    parser.add_argument("--tag_type", type = str, default = 'matching')
    args = parser.parse_args()

    # config
    config_path = Path(args.config_path)
    config = OmegaConf.load(config_path)
    model, tokenizer, _ = get_model(config.framework, config.text_type, config.text_rep)
    dir_by_config = Path(f'{config.framework}/{config.text_type}/{config.text_rep}')

    # dataset config
    dataset_config_path = Path(args.dataset_config_path)
    tag_type = args.tag_type
    dataset_config = OmegaConf.load(dataset_config_path)
    
    # dataset
    dataset_name = args.dataset_name
    if dataset_name == 'musiccaps':
        dataset = MusicCaps(dataset_config.musiccaps.csv_path)
        audio_dir = Path(dataset_config.musiccaps.audio)
    elif dataset_name == 'song_describer':
        dataset = SongDescriber(dataset_config.song_describer.csv_path)
        audio_dir = Path(dataset_config.song_describer.audio)
    else:
        raise NotImplementedError
    
    # path
    tag_path = Path(dataset_config.tags_dir)/tag_type/dataset_name/f'tags.json'
    audio_embs_path = Path(dir_by_config/'preprocessing'/dataset_name/'audio_embs.pt')
    text_embs_path = Path(dir_by_config/'preprocessing'/dataset_name/f'text_embs_{tag_type}.pt')
    sim_dir = Path(dir_by_config/'sim_result'/dataset_name/tag_type)
    result_dir = Path('result'/dir_by_config/dataset_name/tag_type)

    # do retrieval
    audio_embs_dict = get_audio_embed_dict(audio_embs_path, dataset)
    dataset.df['tag'] = get_tag_list(tag_path, dataset)
    text_embs_dict = get_text_embed_dict(text_embs_path, dataset)
    calculate_and_save_sim(sim_dir, dataset, audio_embs_dict, text_embs_dict)

    # recall at k
    portion_list = [-1, 0, 0.3, 0.5, 0.7, 'random'] # -1: just average, 0: caption only
    calculate_and_save_rank(result_dir, dataset, audio_embs_dict, sim_dir, dir_by_config, dataset_name, portion_list, k_list=[10])

    k_list = [10]
    calculate_recall_at_k(result_dir, dataset, k_list, portion_list)