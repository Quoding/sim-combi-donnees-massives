import argparse
import datetime
import json
import logging
import math
import os
import random
import time

import numpy as np
import pandas as pd
import scipy as sp
import torch
from faker import Faker
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)

fake = Faker()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate polypharmacy dataset")
    parser.add_argument(
        "--config",
        default="configs_patients/config.json",
        help="Path to the configuration file (JSON)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Set seed of rng. Overrides config file's seed.",
    )
    args = parser.parse_args()
    return args


def set_seed(args, config):
    if args.seed is not None:
        seed = args.seed
    else:
        seed = config["seed"]

    logging.info(f"Setting seed to {seed}")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    fake.seed_instance(seed)

    return seed


def check_gpu(config):
    if torch.cuda.is_available() and config["use_gpu"]:
        logging.info("Torch detected a CUDA device. Using GPU for data generation...")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        logging.info(
            "Torch did not detect a CUDA device or use_gpu=0 in config. Data generation will not use GPU..."
        )


def read_config(path_to_config):
    logging.info(f"Loading configuration file: {path_to_config}")
    with open(path_to_config) as file:
        config = json.load(file)
    return config


def generate_patterns(config):
    """Generate patterns from the loaded config

    Args:
        config (dict (JSON)): configuration

    Returns:
        tuple (torch.Tensor, torch.Tensor): patterns and risks tensors
    """
    n_patterns = config["patterns"]["n_patterns"]
    min_prob_outcome = config["patterns"]["min_prob_outcome"]
    mean_rx = config["patterns"]["mean_rx"]
    alpha = config["patterns"]["alpha"]
    beta = config["patterns"]["beta"]
    n_rx = config["n_rx"]

    logging.info("Generating patterns and their risks...")

    p = mean_rx / n_rx
    size_pattern = (n_patterns, n_rx)

    prob_matrix = np.full(size_pattern, p)

    patterns = np.random.binomial(1, p=prob_matrix)

    prob_outcome = np.random.beta(alpha, beta, len(patterns))
    prob_outcome = np.clip(prob_outcome, min_prob_outcome, np.inf)

    return patterns, prob_outcome


def save_patterns(patterns, patterns_idx, risks, config, seed):
    """Save pattern to file. File path is dictated by output_dir, file_identifier and seed* in config.
    *Seed can be overwritten from arguments.


    Args:
        patterns (torch.Tensor):  tensor of generated patterns
        risks (torch.Tensor): tensor of generated risks
        config (dict): dictionnary containing configuration parameters
    """

    directory = config["output_dir"]
    filename = f"patterns/{config['file_identifier']}_{seed}.json"
    out_path = f"{directory}/{filename}"

    dict_ = {
        f"pattern_{i}": {
            "pattern": patterns[i].tolist(),
            "patterns_idx": list(patterns_idx[i]),
            "risk": round(risks[i].item(), 2),
        }
        for i in range(len(patterns))
    }

    logging.info(f"Saving patterns at {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dict_, f)


def get_prob_outcome(
    v_idx: set, patterns_idx: list, patterns_probs: list, config: dict
):
    # Get distance to closest pattern
    biggest_overlap = 0
    nearest_pattern_idx = None
    for i, p_idx in enumerate(patterns_idx):
        overlap = v_idx & p_idx
        overlap_size = len(overlap)
        if biggest_overlap < overlap_size:
            nearest_pattern_idx = i
            biggest_overlap = overlap_size

    if biggest_overlap == 0:
        return config["disjoint_combinations"]["prob_outcome"]

    # Calculer penalite de distance basée sur la taille de l'intersection
    nearest_pattern = patterns_idx[nearest_pattern_idx]
    union = v_idx | nearest_pattern
    inter = v_idx & nearest_pattern
    disjoint_elem = union - inter
    # Calculer la prob outcome pour la combi
    prob_outcome = patterns_probs[nearest_pattern_idx] * (len(inter) / len(union))

    # print(v_idx, nearest_pattern)
    # print(
    #     patterns_probs[nearest_pattern_idx],
    #     prob_outcome,
    #     len(disjoint_elem),
    #     len(union),
    # )

    return prob_outcome


def generate_patient_database(patterns_idx, patterns_probs, config):
    first_dump = True
    db_save_path = f"{config['output_dir']}/patients/"
    # Prep database
    database = []
    os.makedirs(db_save_path, exist_ok=True)
    total_length = 0
    file_path = db_save_path + f"{config['file_identifier']}_{config['seed']}.csv"
    if os.path.exists(file_path):
        logging.info("Detected dataset with same name / location. Deleting old file...")
        os.remove(file_path)

    available_rx = set(list(range(0, config["n_rx"])))

    for patient in tqdm(range(config["n_patients"])):
        current_combi_idx = set()
        # print(f"{patient=}")
        date = fake.date_between(
            start_date=datetime.datetime(2000, 1, 1),
            end_date=datetime.datetime(2020, 12, 31),
        )
        for t in range(config["n_timesteps"]):
            # print(f"{t=}")

            r1, r2, r3, r4, r5 = np.random.uniform(0, 1, size=5)
            outcome = False

            # Remove Rx avec prob p1
            if r1 < config["p1"] and len(current_combi_idx) > 0:
                idx_to_remove = random.choice(list(current_combi_idx))
                current_combi_idx.remove(idx_to_remove)

            # Add Rx avec prob p2
            if r2 < config["p2"] and len(current_combi_idx) < config["n_rx"]:
                idx_to_add = random.choice(list(available_rx - current_combi_idx))
                current_combi_idx.add(idx_to_add)
            # Add 2e Rx avec prob p3 (p1 < p3 < p2)
            if r3 < config["p3"] and len(current_combi_idx) < config["n_rx"]:
                idx_to_add = random.choice(list(available_rx - current_combi_idx))
                current_combi_idx.add(idx_to_add)

            # Get la prob d'outcome pour cette combinaison
            prob_outcome = get_prob_outcome(
                current_combi_idx, patterns_idx, patterns_probs, config
            )
            # Attribuer un outcome à ce record sur cette prob
            if r4 < prob_outcome:
                outcome = True

            # Add le record (patient_idx, t, combinaison, outcome) dans la bd
            combi_binary = np.zeros(config["n_rx"])
            combi_binary[list(current_combi_idx)] = 1
            entry = combi_binary.astype(int).tolist()
            entry.insert(0, date)
            entry.insert(0, patient)
            entry.append(int(outcome))
            database.append(entry)
            # Simulate loss of patient to death or change of country
            if r5 < config["prob_stop"]:
                break

            date += datetime.timedelta(days=random.randint(5, 60))

        # Append to csv
        if (patient + 1) % 1000000 == 0 or (patient + 1) == config["n_patients"]:
            # header = (patient + 1) == 10000
            column_names = ["patient_id", "timestamp"]
            for i in range(config["n_rx"]):
                column_names.append(f"drug_{i}")
            column_names.append("hospit")

            database = pd.DataFrame(database, columns=column_names)
            total_length += len(database)
            database.to_csv(
                file_path,
                index=False,
                # chunksize=1024,
                header=first_dump,
                mode="a",
            )
            database = []
            first_dump = False

    print(f"Total length of data generated: {total_length}")
    return database


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config)
    # print_warnings(config)
    seed_value = set_seed(args, config)
    # check_gpu(config)
    patterns, p_outcome = generate_patterns(config)
    patterns_idx = [set(np.where(pat == 1)[0].tolist()) for pat in patterns]
    # print(patterns, p_outcome)
    db = generate_patient_database(patterns_idx, p_outcome, config)
    save_patterns(patterns, patterns_idx, p_outcome, config, seed_value)
    # print(len(db))

    # db.to_csv(
    #     db_save_path + f"{config['file_identifier']}_{config['seed']}.csv",
    #     index=False,
    #     chunksize=1024,
    # )
    # save_combinations(
    #     combinations,
    #     c_risks,
    #     config,
    #     seed_value,
    #     c_inter_bool,
    #     c_dists,
    # )

    # print_quick_stats(combinations, c_risks, c_inter_bool, 1.1)
    logging.info("Finished generating dataset!")
