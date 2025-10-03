from collections import defaultdict
from dataclasses import dataclass, field

from src.pdt_tag_handler import PDTTagHandler
from src.util import normalize_probability_distribution


@dataclass
class TrieNode:
    """
    Class to represent a trie node.
    """

    # Child trie nodes
    children: defaultdict[str, "TrieNode"] = field(default_factory=lambda: defaultdict(TrieNode))

    # Gender probabilities at this node
    gender_probs: defaultdict[str | None, float] = field(default_factory=lambda: defaultdict(lambda: 0.0))

    def insert_suffix(self, suffix: str, gender_probs: dict[str | None, float]) -> None:
        """
        Insert noun suffix with noun gender probabilities into trie.
        :param suffix: Noun suffix to insert
        :param gender_probs: Gender probabilities of the noun the suffix belongs to
        """

        # Add gender probabilities to current gender probabilities
        for gender, gender_prob in gender_probs.items():
            self.gender_probs[gender] += gender_prob

        # End of recursion
        if len(suffix) == 0:
            return

        # Insert rest of the suffix
        letter, rest_suffix = suffix[0], suffix[1:]
        self.children[letter].insert_suffix(suffix=rest_suffix, gender_probs=gender_probs)

    def normalize_node_gender_probs(self) -> None:
        """
        Normalize gender probabilities to sum to 1.
        """
        # Normalize gender probabilities
        self.gender_probs = normalize_probability_distribution(prob_distribution=self.gender_probs)

        # Normalize gender probabilities of child nodes
        for children_node in self.children.values():
            children_node.normalize_node_gender_probs()


@dataclass
class TrieModel:
    """
    Class to represent a trie model.
    """

    # Noun to gender probabilities
    nouns_to_gender_probs: dict[str, dict[str | None, float]]

    # Trie-smoothing parameters
    alpha: float
    beta: float

    # Root node
    root: TrieNode = field(default_factory=TrieNode)

    def _insert_noun(self, noun: str, gender_probs: dict[str | None, float]) -> None:
        """
        Insert noun into trie.
        :param noun: Noun to insert
        :param gender_probs: Noun gender probabilities
        """

        # Insert the noun reversed
        self.root.insert_suffix(suffix=noun[::-1], gender_probs=gender_probs)

    def _normalize_nodes_gender_probs(self) -> None:
        """
        Normalize gender probabilities at each node in trie.
        """
        self.root.normalize_node_gender_probs()

    def __post_init__(self) -> None:
        # Insert all nouns
        for noun, gender_probs in self.nouns_to_gender_probs.items():
            self._insert_noun(noun=noun, gender_probs=gender_probs)

        # Normalize gender probs at each trie node
        self._normalize_nodes_gender_probs()

    def _compute_noun_gender_prob(self, noun: str, gender: str) -> float:
        """
        Compute gender probability for a noun and a gender.
        :param noun: Noun to compute gender probability for
        :param gender: Gender to compute gender probability for
        :return: Computed gender probability
        """
        # Noun is empty string -> return probability 0
        if len(noun) == 0:
            return 0

        # Traverse trie and compute probability
        last_letter, rest = noun[-1], noun[:-1]
        node = self.root.children[last_letter]
        computed_prob = node.gender_probs[gender]

        while len(rest) > 0:
            last_letter, rest = rest[-1], rest[:-1]
            node = node.children[last_letter]
            node_gender_prob = node.gender_probs[gender]
            node_quest_prob = node.gender_probs[None]

            # Questionable probability is < 1 -> update the probability computed so far
            if node_quest_prob < 1:
                gamma = (1 - self.beta * node_quest_prob**self.alpha) / (1 - node_quest_prob)
                computed_prob = (gamma * node_gender_prob) + (self.beta * node_quest_prob**self.alpha) * computed_prob

        return computed_prob

    def compute_noun_gender_probs(self, noun: str) -> dict[str, float]:
        """
        Compute gender probabilities for a noun.
        :param noun: Noun to compute gender probabilities for
        :return: Computed gender probabilities
        """
        gender_probs = {}

        # Computed probabilities of all genders
        for gender in PDTTagHandler.get_all_genders():
            gender_prob = self._compute_noun_gender_prob(noun=noun, gender=gender)
            gender_probs[gender] = gender_prob

        # Normalize computed gender probabilities
        return normalize_probability_distribution(prob_distribution=gender_probs)
