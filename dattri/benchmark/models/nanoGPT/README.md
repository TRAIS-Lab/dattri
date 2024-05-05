
# nanoGPT
Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## data prepare

The upload version does not include the dataset in the folder, so run the follewing commend first. It will download the input.txt.

```
$ python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. 


## retrain 
You can run the following commend to give your custom parameters to retrain the nanoGPT given run times on Shakespeare_char dataset.

```
python shakespeare.py --num_runs 3 --seed 43 --config_path path/to/config.py --base_out_dir path/to/output
```

