"""Test MAESTRO functions."""

import zipfile
from pathlib import Path

import requests
import torch

from dattri.benchmark.maestro.train import (
    create_loaders,
    eval_maestro,
    train_maestro,
)
from dattri.benchmark.models.MusicTransformer.dataset.preprocess_midi import (
    prep_maestro_midi,
)

# check whether the zip file is there
zip_dataset_filename = "maestro-v2.0.0-midi.zip"
zip_dataset_path = Path(zip_dataset_filename)

if not zip_dataset_path.exists():
    # data downloading
    dataset_urls = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
    response = requests.get(dataset_urls, stream=True, timeout=20)
    with zip_dataset_path.open("wb") as midi:
        for chunk in response.iter_content(chunk_size=1024):
            # writing one chunk at a time to midi file
            if chunk:
                midi.write(chunk)

# check whether the unzipped file is there
dataset_filename = "maestro-v2.0.0"
dataset_path = Path(dataset_filename)

if not dataset_path.exists():
    with zipfile.ZipFile(zip_dataset_filename, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            zip_ref.extract(file_info, "")

# preprocess the file
output_path = Path("./maestro-v2.0.0-processed")
if not output_path.exists():
    prep_maestro_midi(dataset_path, output_path)

train_loader, val_loader, test_loader = create_loaders()
# train the MusicTransformer on 5000 subset
# overfit on the train_loader: val_loader = train_loader
model = train_maestro(train_dataloader=train_loader, val_dataloader=train_loader)
torch.save(model.state_dict(), "model_weights.pth")

# eval the MusicTransformer on 5000 subset
model_path = Path("model_weights.pth")
eval_loss = eval_maestro(model_path, train_loader)

# remove all downloaded files and generatad files if needed
