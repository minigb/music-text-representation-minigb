# Modified the provided code in README
import torch
import numpy as np

from mtr.utils.audio_utils import load_audio, STR_CH_FIRST

def audio_infer(audio_path, model, sr=16000, duration=None):
    assert duration is not None, "audio duration must be specified"

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