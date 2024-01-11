# Modified the provided demo.ipynb

import argparse
from pathlib import Path
from omegaconf import OmegaConf
import shutil
import logging

from mtr.utils.demo_utils import get_model
from utils.retrieval import *


def get_dirs(dataset_config, dataset_name, tag_type, dir_by_model_info, init_previous, run_name) -> tuple:
    dirs_to_save_process = {
        'embeds': Path(dir_by_model_info/'preprocessing'/f'{dataset_name}{run_name}'),
        'sim': Path(dir_by_model_info/'sim_result'/dataset_name/f'{tag_type}{run_name}'),
        'rank': Path('rank'/dir_by_model_info/dataset_name/f'{tag_type}{run_name}'),
        'result': Path('result'/dir_by_model_info/dataset_name/f'{tag_type}{run_name}'),
    }

    init = init_previous
    for dir_path in dirs_to_save_process.values():
        if not dir_path.exists():
            # If a dir doesn't exist, the dirs that comes after that should be initialized to apply the updated result
            init = True
        if init and dir_path.exists():
            shutil.rmtree(dir_path)

    tag_path = Path(dataset_config.tags_dir)/tag_type/f'{dataset_name}_tags.json'

    return dirs_to_save_process, tag_path

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type = str, default = 'contrastive')
    parser.add_argument("--text_type", type = str, default = 'bert')
    parser.add_argument("--text_rep", type = str, required = True)

    parser.add_argument("--dataset_config_path", type = str, required = True)
    parser.add_argument("--dataset_name", type = str, required = True)
    parser.add_argument("--tag_type", type = str, default = 'matching')

    parser.add_argument("--init", type = bool, default = False)
    parser.add_argument("--run_name", type = str, default = '')
    
    args = parser.parse_args()

    framework = args.framework
    text_type = args.text_type
    text_rep = args.text_rep
    dataset_config_path = Path(args.dataset_config_path)
    tag_type = args.tag_type
    dataset_name = args.dataset_name

    # model
    model, tokenizer, _ = get_model(framework, text_type, text_rep)
    dir_by_model_info = Path(f'{framework}/{text_type}/{text_rep}')

    # dataset
    dataset_config = OmegaConf.load(dataset_config_path)
    dataset, audio_dir = get_dataset(dataset_config, dataset_name)
    
    # init
    init_previous = args.init
    run_name = args.run_name
    
    # path
    logging.info('get_dirs')
    dirs, tag_path = get_dirs(dataset_config, dataset_name, tag_type, dir_by_model_info, init_previous, run_name)

    # calculate sim
    logging.info('get_audio_embed_dict')
    audio_embs_dict = get_audio_embed_dict(dirs['embeds'] / 'audio_embs.pt', audio_dir, model, dataset)

    dataset.df['text'] = get_text_all(tag_path, dataset)
    logging.info('get_text_embed_dict')
    text_embs_dict = get_text_embed_dict(dirs['embeds'] / f'text_embs_{tag_type}.pt', model, tokenizer, dataset)

    logging.info('calculate_sim_and_save')
    calculate_sim_and_save(dirs['sim'], dataset, audio_embs_dict, text_embs_dict)

    # calculate rank
    # TODO(minigb): Take portion_list and k_list out to config or args
    portion_list = [-1, 0, 0.3, 0.5, 0.7, 'random'] # -1: just average, 0: caption only
    logging.info('calculate_rank_and_save')
    calculate_rank_and_save(dirs['rank'], dataset, audio_embs_dict, dirs['sim'], portion_list)

    # calculate recall@k
    k_list = [1, 5, 10]
    logging.info('calculate_recall_at_k_and_save')
    calculate_recall_at_k_and_save(dirs['rank'], dataset, k_list, portion_list, dirs['result'])