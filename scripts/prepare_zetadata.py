"""prepare script for the zeta report builder data used in the 2023 hackathon

Adapted from prepare_alapaca.py

"""
import json
import sys
from pathlib import Path

import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm
import s3fs

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

S3_PROFILE = "iam"
DATA_FILE_URL = "s3://ai-ml-research/hackathon-2023/report_builder_data_augmented_combined.jsonl"
DATA_FILE_NAME = "report_builder_data_augmented_combined.jsonl"
LOCAL_DESTINATION_PATH = Path("data/zeta")
S3_DESTINATION_PATH = "s3://ai-ml-research/hackathon-2023/prepared_data"
CHECKPOINT_DIR = Path("checkpoints/stabilityai/stablelm-base-alpha-3b")
TEST_SPLIT_FRACTION = 0.25
IGNORE_INDEX = -1
MASK_INPUTS = False  # as in alpaca-lora
SEED = 42


def prepare(
    destination_path: Path = LOCAL_DESTINATION_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    test_split_fraction: float = TEST_SPLIT_FRACTION,
    seed: int = SEED,
    mask_inputs: bool = MASK_INPUTS,
    data_file_name: str = DATA_FILE_NAME,
    data_file_url: str = DATA_FILE_URL,
    ignore_index: int = IGNORE_INDEX,
    s3_profile: str = S3_PROFILE,
) -> None:
    """Prepare the Zeta dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    with open(checkpoint_dir / "lit_config.json", "r") as file:
        config = json.load(file)
        max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    data_file_path = destination_path / data_file_name

    # ensure JSONL file is passed
    if not str(data_file_path).endswith(".jsonl"):
        raise ValueError(f"Input file {data_file_name} is not a JSONL file")

    print("Loading data file...")
    download_if_missing(data_file_path, data_file_url, s3_profile=s3_profile)

    with open(data_file_path, "r", encoding="utf-8") as file:
        json_list = list(file)

    data = []
    for json_line in json_list:
        json_contents = json.loads(json_line)
        question = json_contents["question"]
        output = json_contents["output"]

        # lit-gpt scripts expects "instruction", "input" (optional), and "output"
        # see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/finetune_adapter.md#tune-on-your-dataset
        data.append({
            "instruction": question,
            "input": "",
            "output": output,
        })

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    # Partition the dataset into train and test
    train_set, test_set = random_split(
        data, [1.0 - test_split_fraction, test_split_fraction], generator=torch.Generator().manual_seed(seed)
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")

    # sync to s3 bucket
    s3 = s3fs.S3FileSystem(profile=None if s3_profile.lower() == "iam" else s3_profile)
    with s3.open(S3_DESTINATION_PATH + "/train.pt", 'wb') as train_outfile:
        torch.save(train_set, train_outfile)
    with s3.open(S3_DESTINATION_PATH + "/test.pt", 'wb') as test_outfile:
        torch.save(test_set, test_outfile)


def download_if_missing(file_path: Path, file_url: str, s3_profile: str = "iam"):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return

    if file_url.startswith("s3://"):
        print(f"Attempting to sync data file '{file_url}' from s3")
        s3 = s3fs.S3FileSystem(profile=None if s3_profile.lower() == "iam" else s3_profile)
        with s3.open(file_url, 'r') as infile:
            contents = infile.read()
        with open(file_path, "w", encoding="utf-8") as outfile:
            outfile.write(contents)

    else:
        print(f"Attempting to download data file '{file_url}' using requests")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(requests.get(file_url).text)


def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
    max_length: int,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an question from a user. Write a response using JSON markdown which contains the "
        "metrics and dimensions needed to answer the question.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
