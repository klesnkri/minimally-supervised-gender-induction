from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.conllu_handler import CoNLLuHandler
from src.pdt_tag_handler import PDTTagHandler


@dataclass(frozen=True)
class NounContext:
    """
    Class to represent noun context.
    """

    left_context: str
    right_context: str


@dataclass
class DataLoader:
    """
    Class to load data from the Prague Dependency Treebank corpus and store various statistics needed for the gender induction algorithm.
    """

    # Arguments
    corpus_path: str
    context_type: str
    context_word_form: str
    non_noun_to_noun_ratio_threshold: float
    context_statistics_type: str

    # Noun list with ref genders
    ref_gender_noun_list: defaultdict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # Noun counter
    noun_cntr: Counter[str] = field(default_factory=Counter)
    # Noun to context counter mapping
    noun_to_context_cntr: defaultdict[str, Counter[NounContext]] = field(default_factory=lambda: defaultdict(Counter))
    # Context to noun counter mapping
    context_to_noun_cntr: defaultdict[NounContext, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def __post_init__(self) -> None:
        self._compute_stats()

    def _add_new_cnt_to_context_statistics(self, added_context_cnt: int, prev_context_cnt: int) -> int:
        """
        Update context statistics according to the specified context statistics type (type/token).
        :param added_context_cnt: Count to add to the context statistics
        :param prev_context_cnt: Previous context statistics count
        :return: Updated context statistics count
        """
        # Token statistics - add the whole count
        if self.context_statistics_type == "token":
            return prev_context_cnt + added_context_cnt
        # Type statistics - add just 1
        return 1

    def _transform_word_into_contexts(self, word: str) -> list[str]:
        """
        Generate all contexts from a given word.
        :param word: Word to generate contexts from
        :return: Generated contexts
        """
        # Context is just the word form itself
        if self.context_word_form == "word":
            return [word]

        # Context are all the word form suffixes
        return [word[i:] for i in range(len(word) + 1)]

    def _generate_word_contexts(self, left_word: str, right_word: str) -> list[NounContext]:
        """
        Generate all neighbouring contexts of a word.
        :param left_word: Left neighbouring word to generate contexts from
        :param right_word: Right neighbouring word to generate contexts from
        :return: Generated word contexts
        """
        # Starting contexts
        left_word_contexts = [""]
        right_word_contexts = [""]

        # Generate left contexts
        if self.context_type in {"left", "bilateral"}:
            left_word_contexts = self._transform_word_into_contexts(word=left_word)

        # Generate right contexts
        if self.context_type in {"right", "bilateral"}:
            right_word_contexts = self._transform_word_into_contexts(word=right_word)

        # Return all left and right context combinations
        return [
            NounContext(left_context=left_word_context, right_context=right_word_context)
            for left_word_context in left_word_contexts
            for right_word_context in right_word_contexts
        ]

    def _context_should_be_filtered(self, context: NounContext, context_to_non_noun_cntr: dict[NounContext, Counter[str]]) -> bool:
        """
        Test whether a given word context should be filtered.
        :param context: Given word context
        :param context_to_non_noun_cntr: Counter of contexts to non-noun words
        :return: Whether a given word context should be filtered
        """
        # Filter out contexts that occur too much with words outside the noun list
        non_noun_to_noun_ratio = sum(context_to_non_noun_cntr[context].values()) / sum(self.context_to_noun_cntr[context].values())
        if non_noun_to_noun_ratio >= self.non_noun_to_noun_ratio_threshold:
            return True

        # TODO Filter out contexts using a frequency threshold sensitive to both the size of the corpus and seed list?

        return False

    def _compute_stats(self) -> None:
        """
        Read corpus and store statistics needed for the gender induction algorithm.
        """
        # Read corpus sentences
        sentences = CoNLLuHandler.parse_file(self.corpus_path)

        # Obtain noun list with ref genders (should be extracted from some dictionary, here it is extracted from the corpus itself)
        for sentence in sentences:
            for token in sentence:
                if CoNLLuHandler.is_base_noun_token(token=token):
                    noun = CoNLLuHandler.extract_word_form(token=token)
                    tag = CoNLLuHandler.extract_pdt_tag(token=token)
                    gender = PDTTagHandler.extract_ref_gender(tag=tag)

                    self.ref_gender_noun_list[noun].add(gender)

        # Save noun statistics from the corpus
        context_to_non_noun_cntr = defaultdict(Counter)

        for sentence in sentences:
            for left_token, token, right_token in CoNLLuHandler.iterate_sentence_tokens(sentence=sentence):
                word_form = CoNLLuHandler.extract_word_form(token=token)
                left_word = CoNLLuHandler.extract_word_form(token=left_token) if left_token is not None else ""
                right_word = CoNLLuHandler.extract_word_form(token=right_token) if right_token is not None else ""
                word_contexts = self._generate_word_contexts(left_word=left_word, right_word=right_word)

                # Word from noun list
                if word_form in self.ref_gender_noun_list:
                    self.noun_cntr[word_form] += 1

                    for word_context in word_contexts:
                        self.context_to_noun_cntr[word_context][word_form] = self._add_new_cnt_to_context_statistics(
                            added_context_cnt=1, prev_context_cnt=self.context_to_noun_cntr[word_context][word_form]
                        )
                # Word outside of noun list
                else:
                    for word_context in word_contexts:
                        context_to_non_noun_cntr[word_context][word_form] = self._add_new_cnt_to_context_statistics(
                            added_context_cnt=1, prev_context_cnt=context_to_non_noun_cntr[word_context][word_form]
                        )

        # Filter contexts
        for context in list(self.context_to_noun_cntr.keys()):
            if self._context_should_be_filtered(context=context, context_to_non_noun_cntr=context_to_non_noun_cntr):
                # Filter context
                del self.context_to_noun_cntr[context]

        # Add noun-to-contexts mapping
        for context, noun_cntr in self.context_to_noun_cntr.items():
            for noun, cnt in noun_cntr.items():
                self.noun_to_context_cntr[noun][context] = cnt
