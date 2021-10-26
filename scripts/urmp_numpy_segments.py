import fire
import gin
from tqdm import tqdm
import os

from vq_timbre.models import lightning_run
from vq_timbre.utils import (
  create_numpy_files, 
  make_directory, 
  gin_register_and_parse,
  load_audio_file,
  get_urmp_files, 
  torch_float32, 
  load_urmp_f0s,
  resample_f0
)

@gin.configurable
def generate_numpy_segments(
    data_dir, 
    output_dir,
    segment_seconds: float = 3, # in seconds
    sr: float = 16000,
    gin_config="gin_configs/data_gen_urmp.gin"
):

    gin.parse_config_file(os.path.join(os.getcwd(), gin_config))

    file_paths = get_urmp_files(data_dir)

    # load audio clips
    audio_clips = [load_audio_file(f['audio']) for f in tqdm(file_paths)] 

    # load f0s and resample to match audio length
    f0s_np = [load_urmp_f0s(f['f0'], sr) for f in tqdm(file_paths)] 
    f0s_np = [resample_f0(f0, audio_clips[i].shape[-1]) for i, f0 in enumerate(f0s_np)]

    split_set = 'np-segments-' + str(segment_seconds)
    output_dir_seg = os.path.join(output_dir, split_set)
    make_directory(output_dir_seg)

    # save audio and f0 np files
    audio_fnames = [os.path.basename(f['audio']) for f in file_paths]
    create_numpy_files(audio_clips, 
                       audio_fnames, 
                       output_dir_seg, 
                       segment_length = sr * segment_seconds)

    f0_fnames = [os.path.basename(f['f0']) for f in file_paths]
    create_numpy_files(f0s_np, 
                       f0_fnames, 
                       output_dir_seg, 
                       segment_length = sr * segment_seconds)

def main():
  fire.Fire(generate_numpy_segments)

if __name__ == "__main__":
    main()