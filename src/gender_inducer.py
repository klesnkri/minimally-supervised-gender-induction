import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.data_loader import DataLoader, NounContext
from src.pdt_tag_handler import PDTTagHandler
from src.trie_model import TrieModel
from src.util import normalize_probability_distribution


@dataclass
class GenderInducer:
    """
    Class for the gender induction algorithm.
    """

    # Arguments
    logdir: str
    data_loader: DataLoader
    num_seeds: int
    gender_prob_threshold: float
    default_gender: str
    morpho_statistics_type: str
    alpha: float
    beta: float

    # Context to gender assignment
    context_to_gender: defaultdict[NounContext, str | None] = field(default_factory=lambda: defaultdict(lambda: None))
    # Noun to gender assignment
    gender_assignment: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _combine_noun_sortings(*noun_sortings: Iterable[str]) -> list[str]:
        """
        Combine multiple noun sortings into final sorting.
        :param noun_sortings: Noun sortings to combine
        :return: Final noun sorting
        """
        final_sorting = defaultdict(int)

        for noun_sorting in noun_sortings:
            for idx, noun in enumerate(noun_sorting):
                final_sorting[noun] += idx

        return sorted(final_sorting.keys(), key=lambda x: final_sorting[x])

    def _extract_seed_nouns(self) -> None:
        """
        Extract seed nouns from the corpus selected on the basis of frequency and number of contexts with which they co-occur.
        Multi-gender nouns aren't selected.
        :return: Extracted seed nouns
        """

        # Nouns sorted by frequency
        nouns_sorted_by_cnt = [noun for noun, cnt in sorted(self.data_loader.noun_cntr.items(), reverse=True, key=lambda x: x[1])]

        # Nouns sorted by number of contexts
        nouns_context_cnt = [(noun, sum(self.data_loader.noun_to_context_cntr[noun].values())) for noun in self.data_loader.noun_cntr]
        nouns_sorted_by_context_cnt = [noun for noun, cnt in sorted(nouns_context_cnt, reverse=True, key=lambda x: x[1])]

        # TODO: Nouns sorted by suffix patterns?

        nouns_sorted_by_priority = self._combine_noun_sortings(nouns_sorted_by_cnt, nouns_sorted_by_context_cnt)

        # Extract num_seeds seed words for every gender according to priority
        seed_gender_assignment = {}

        for gender in PDTTagHandler.get_all_genders():
            # Eliminate multi-gender seeds
            seed_nouns = [noun for noun in nouns_sorted_by_priority if self.data_loader.ref_gender_noun_list[noun] == {gender}][: self.num_seeds]
            seed_gender_assignment.update({noun: gender for noun in seed_nouns})

        # Update gender assignment
        self.gender_assignment.update(seed_gender_assignment)

    def _compute_contextual_gender_probs(self, noun: str) -> defaultdict[str | None, float]:
        """
        Compute gender probabilities of a given noun by considering genders assigned to the contexts with which the noun co-occurs.
        :param noun: Noun to compute gender probabilities for
        :return: Computed gender probabilities
        """
        gender_probs = defaultdict(lambda: 0.0)

        # Noun without contexts -> questionable context with probability 1
        if len(self.data_loader.noun_to_context_cntr[noun]) == 0:
            gender_probs[None] = 1
        # Noun with contexts
        else:
            gender_cntr = Counter()

            # Compute gender counts
            for context, context_cnt in self.data_loader.noun_to_context_cntr[noun].items():
                context_gender = self.context_to_gender.get(context)
                gender_cntr[context_gender] += context_cnt

            # Normalize gender counts to probability distribution
            gender_probs = normalize_probability_distribution(prob_distribution=gender_cntr)

        return gender_probs

    @staticmethod
    def _get_most_probable_gender(gender_probs: dict[str, float]) -> tuple[str | None, float]:
        """
        Get most probable gender along with its probability from gender probabilities.
        :param gender_probs: Gender probabilities
        :return: Most probable gender along with its probability
        """
        gender = max(gender_probs, key=gender_probs.get)
        gender_prob = gender_probs[gender]
        return gender, gender_prob

    def _get_nouns_without_assigned_gender(self) -> list[str]:
        """
        Get nouns that don't have assigned gender yet.
        Return them sorted so the algorithm is deterministic.
        :return: Nouns that don't have assigned gender yet sorted
        """
        return sorted(self.data_loader.noun_cntr.keys() - self.gender_assignment.keys())

    def _context_bootstrapping(self) -> None:
        """
        Run the context bootstrapping algorithm.
        """
        new_gender_assignment = self.gender_assignment

        while True:
            # No newly added nouns to the gender assignment
            if len(new_gender_assignment) == 0:
                break

            # Obtain gender contexts from newly added words
            for noun, gender in new_gender_assignment.items():
                noun_contexts = self.data_loader.noun_to_context_cntr[noun]

                for context in noun_contexts:
                    # Save context of the gender
                    self.context_to_gender[context] = gender

            # Assign gender to new nouns
            new_gender_assignment = {}

            for noun in self._get_nouns_without_assigned_gender():
                gender_probs = self._compute_contextual_gender_probs(noun=noun)
                gender, gender_prob = self._get_most_probable_gender(gender_probs=gender_probs)

                # Contextual gender probability is above a threshold -> assign it
                if gender is not None and gender_prob >= self.gender_prob_threshold:
                    new_gender_assignment[noun] = gender

            # Update gender assignment
            self.gender_assignment.update(new_gender_assignment)

    def _get_init_trie_gender_probs(self, noun: str) -> dict[str | None, float]:
        """
        Get initial gender probabilities for noun that will be inserted into trie. Weight the contextual gender probabilities if token statistics type is used.
        :param noun: Noun that will be inserted into trie
        :return: Initial gender probabilities for a given noun
        """
        noun_gender_probs = self._compute_contextual_gender_probs(noun=noun)

        # Token weighting
        if self.morpho_statistics_type == "token":
            for gender in noun_gender_probs:
                noun_gender_probs[gender] *= self.data_loader.noun_cntr[noun]

        # Type weighting
        return noun_gender_probs

    def _morphological_analysis(self) -> None:
        """
        Run the morphological analysis algorithm.
        """
        # Insert nouns with their probability distribution into trie
        nouns_to_gender_probs = {noun: self._get_init_trie_gender_probs(noun=noun) for noun in self.data_loader.noun_cntr}
        trie = TrieModel(nouns_to_gender_probs=nouns_to_gender_probs, alpha=self.alpha, beta=self.beta)

        # Compute most probable gender for nouns without assigned gender
        for noun in self._get_nouns_without_assigned_gender():
            gender_probs = trie.compute_noun_gender_probs(noun=noun)
            gender, gender_prob = self._get_most_probable_gender(gender_probs=gender_probs)

            # Update gender assignment if some gender has non-null probability
            if gender_prob > 0:
                self.gender_assignment[noun] = gender

    def _assign_default_gender(self) -> None:
        """
        Assign default gender to nouns that still don't have any gender assigned.
        """
        # Assign default gender
        for noun in self._get_nouns_without_assigned_gender():
            self.gender_assignment[noun] = self.default_gender

    def _save_statistics(self, stats_type: str) -> None:
        """
        Save statistics (coverage and accuracy) to file.
        :param stats_type: Statistics type
        """
        stats = {}

        # Coverage (ratio of nouns with assigned gender)
        coverage = len(self.gender_assignment) / len(self.data_loader.ref_gender_noun_list)
        stats["coverage"] = coverage

        # Accuracy (ratio of nouns that were assigned one of their ref gender)
        accuracy = sum(gender in self.data_loader.ref_gender_noun_list[noun] for noun, gender in self.gender_assignment.items()) / len(self.gender_assignment)
        stats["accuracy"] = accuracy

        # Save statistics to file
        with open(os.path.join(os.getenv("OUT_DIR"), self.logdir, f"{stats_type}_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

    def _save_gender_assignment(self) -> None:
        """
        Save noun gender assignment to file
        """
        # Sort gender assignment so the results are deterministic
        sorted_gender_assignment = dict(sorted(self.gender_assignment.items()))

        # Save gender assignment to file
        with open(os.path.join(os.getenv("OUT_DIR"), self.logdir, "gender_assignment.json"), "w", encoding="utf-8") as f:
            json.dump(sorted_gender_assignment, f, indent=4, ensure_ascii=False)

    def induce_gender(self) -> None:
        """
        Run the gender induction algorithm.
        """

        # Extract seed nouns
        self._extract_seed_nouns()

        # Context bootstrapping
        self._context_bootstrapping()

        # Save coverage and accuracy after context bootstrapping
        self._save_statistics(stats_type="after_context_bootstrapping")

        # Morphological analysis
        self._morphological_analysis()

        # Save coverage and accuracy after morphological analysis
        self._save_statistics(stats_type="after_morpho_analysis")

        # Assign default gender to nouns without assign gender
        self._assign_default_gender()

        # Save final coverage and accuracy
        self._save_statistics(stats_type="final")

        # Save final gender assignment
        self._save_gender_assignment()
