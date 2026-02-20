#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data_4.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model_4.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # Load data distribution, each line containing a datapoint -- a string.
    counter = 0
    data_freq = dict()
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            counter += 1
            if data_freq.__contains__(line):
                data_freq[line] += 1
            else:
                data_freq[line] = 1

    data_keys = sorted(data_freq.keys())
    data_dist = [data_freq[k] / counter for k in data_keys]
    np_data_dist = np.array(data_dist, dtype=np.float64)

    # Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.



    # Load model distribution, each line `string \t probability`
    model_probs = dict()

    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # Process the line, aggregating using Python data structures.
            dist , prob = line.split("\t")
            model_probs[dist] = float(prob)

    # Create a NumPy array containing the model distribution aligned to data keys.
    # If the model is missing a probability for any datapoint observed in data,
    # cross-entropy (and KL) should be +inf.
    model_dist = [model_probs.get(k, np.nan) for k in data_keys]
    np_model_dist = np.array(model_dist, dtype=np.float64)




    # Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(np_data_dist * np.log(np_data_dist))

    # Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # the resulting crossentropy should be `np.inf`.

    # Cross-entropy H(p, q) = -sum_x p(x) log q(x).
    # If q(x) is missing or q(x) <= 0 for any x with p(x) > 0, the result is +inf.
    missing_or_invalid = np.isnan(np_model_dist) | (np_model_dist <= 0)
    if np.any(missing_or_invalid):
        crossentropy = np.inf
    else:
        crossentropy = -np.sum(np_data_dist * np.log(np_model_dist))

    # Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = np.inf if not np.isfinite(crossentropy) else (crossentropy - entropy)
    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(main_args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
