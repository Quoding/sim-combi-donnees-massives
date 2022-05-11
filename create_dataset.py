import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import scipy as sp

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--config", default="config.json", help="Path to the configuration file (JSON)"
    )
    parser.add_argument(
        "--seed",
        help="Set seed of rng. Overrides config file's seed.",
    )
    args = parser.parse_args()
    return args


def read_config(path_to_config):
    logging.info(f"Loading configuration file: {path_to_config}")
    with open(path_to_config) as file:
        config = json.load(file)
    return config


def make_patterns(config):
    """[summary]

    Args:
        config ([type]): [description]

    Returns:
        [type]: [description]
    """
    n_patterns = config["patterns"]["n_patterns"]
    min_rr = config["patterns"]["min_rr"]
    max_rr = config["patterns"]["max_rr"]

    n = 1
    p = config["average_rx"] / config["n_rx"]
    size_pattern = (n_patterns, config["n_rx"])

    patterns = np.random.binomial(n, p, size_pattern)

    risks = np.random.uniform(min_rr, max_rr, n_patterns)

    save_patterns(patterns, risks, config)

    return patterns, risks


def save_patterns(patterns, risks, config):
    """Save pattern to file. File path is dictated by output_dir, file_identifier and seed* in config. Seed can be overwritten from arguments.


    Args:
        patterns (np.array):  array of generated patterns
        risks (np.array): array of generated risks
        config (dict): JSON-like dictionnary containing configuration for the dataset
    """

    directory = config["output_dir"]
    filename = f"patterns/{config['file_identifier']}_{config['seed']}.json"
    out_path = f"{directory}/{filename}"

    dict_ = {
        f"pattern_{i}": {"pattern": patterns[i].tolist(), "risk": risks[i]}
        for i in range(len(patterns))
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dict_, f)


def regen_duplicates(patterns, n, p):
    """Regenerate patterns if there are duplicate entries in the patterns
    Show off recursion skills by adding it to a function

    Args:
        patterns (np.array): array of patterns
        n (n): n in a binomial distribution
        p (p): p in a binomial distribution

    Returns:
        np.array : array of patterns with not duplicates but the right amount of patterns
    """
    uniques = np.unique(patterns, axis=0)
    if uniques.shape == patterns.shape:
        return patterns
    else:
        num_to_regen = len(patterns) - len(uniques)
        n_rx = patterns.shape[1]
        patterns_to_add = np.random.binomial(n, p, (num_to_regen, n_rx))
        new_patterns = np.concatenate((uniques, patterns_to_add), axis=0)
        return regen_duplicates(new_patterns, n, p)


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config)
    if args.seed is not None:
        seed = args.seed
    else:
        seed = config["seed"]

    patterns, risks = make_patterns(config)
