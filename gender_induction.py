import argparse
import datetime
import json
import os

from dotenv import load_dotenv

from src.data_loader import DataLoader
from src.gender_inducer import GenderInducer


def get_logdir_name() -> str:
    """
    Create logdir unique name using current date and time.
    :return: Logdir name
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def save_args(logdir: str, args: argparse.Namespace) -> None:
    """
    Save program arguments to file.
    :param logdir: Logdir name
    :param args: Program arguments
    """
    with open(os.path.join(os.getenv("OUT_DIR"), logdir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def parse_args() -> argparse.Namespace:
    """
    Parse program arguments.
    :return: Parsed program arguments
    """
    parser = argparse.ArgumentParser(description="Minimally Supervised Induction of Grammatical Gender for Czech.")
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/ud-treebanks-v2.14/UD_Czech-PDT/cs_pdt-ud-train.conllu",
        help="Path to the Czech Prague Dependency Treebank corpus in CoNLLu format.",
    )
    parser.add_argument("--num_seeds", type=int, default=15, help="Number of seed words for every gender type.")
    parser.add_argument("--context_type", type=str, default="bilateral", choices=["left", "right", "bilateral"], help="Context type.")
    parser.add_argument("--context_word_form", type=str, default="suffix", choices=["word", "suffix"], help="Context word form.")
    parser.add_argument("--context_statistics_type", type=str, default="type", choices=["token", "type"], help="Type of context statistics.")
    parser.add_argument("--morpho_statistics_type", type=str, default="token", choices=["token", "type"], help="How to weight nouns inside trie.")
    parser.add_argument(
        "--default_gender", type=str, default="F", choices=["M", "I", "F", "N"], help="Default gender to assign to words without gender assignment."
    )
    parser.add_argument(
        "--non_noun_to_noun_ratio_threshold",
        type=float,
        default=1.5,
        help="Threshold to filter contexts with ratio of non-noun to noun words with which they co-occur higher than this.",
    )
    parser.add_argument(
        "--gender_prob_threshold",
        type=float,
        default=0.4,
        help="Threshold to assign gender to nouns with gender probability higher than this.",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="Parameter for trie-smoothing.")
    parser.add_argument("--beta", type=float, default=0.99, help="Parameter for trie-smoothing.")

    return parser.parse_args()


def main() -> None:
    # Parse program arguments
    args = parse_args()

    # Load environment variables from file
    load_dotenv()

    # Logdir name
    logdir = get_logdir_name()

    # Create logging directory
    os.makedirs(os.path.join(os.getenv("OUT_DIR"), logdir))

    # Save args
    save_args(logdir=logdir, args=args)

    # Load corpus and create train data and statistics
    data_loader = DataLoader(
        corpus_path=args.corpus_path,
        context_type=args.context_type,
        context_word_form=args.context_word_form,
        non_noun_to_noun_ratio_threshold=args.non_noun_to_noun_ratio_threshold,
        context_statistics_type=args.context_statistics_type,
    )

    # Run gender induction algorithm
    gender_inducer = GenderInducer(
        logdir=logdir,
        data_loader=data_loader,
        num_seeds=args.num_seeds,
        gender_prob_threshold=args.gender_prob_threshold,
        default_gender=args.default_gender,
        morpho_statistics_type=args.morpho_statistics_type,
        alpha=args.alpha,
        beta=args.beta,
    )
    gender_inducer.induce_gender()


if __name__ == "__main__":
    main()
