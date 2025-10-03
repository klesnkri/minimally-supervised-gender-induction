from collections.abc import Iterable
from dataclasses import dataclass

import conllu

from src.pdt_tag_handler import PDTTagHandler


@dataclass
class CoNLLuHandler:
    """
    Class to handle CoNLL-U code.
    """

    # Constants
    PDT_TAG: str = "xpos"
    FORM: str = "form"

    @classmethod
    def extract_word_form(cls, token: conllu.Token) -> str:
        """
        Extract word form from CoNLL-U token. All word forms are normalized by lowercasing.
        :param token: CoNLL-U token to extract word form from
        :return: Extracted word form
        """
        return token[cls.FORM].lower()

    @classmethod
    def extract_pdt_tag(cls, token: conllu.Token) -> str:
        """
        Extract PDT tag from CoNLL-U token.
        :param token: CoNLL-U token to extract PDT tag from
        :return: Extracted PDT tag
        """
        return token[cls.PDT_TAG]

    @staticmethod
    def is_base_noun_token(token: conllu.Token) -> bool:
        """
        Test whether a give CoNLL-U token is a base noun token.
        :param token: CoNLL-U token to test
        :return: Whether a given CoNLL-U token is a base noun token
        """
        tag = CoNLLuHandler.extract_pdt_tag(token=token)
        return tag is not None and PDTTagHandler.is_base_noun(tag=tag)

    @staticmethod
    def iterate_sentence_tokens(sentence: conllu.TokenList) -> Iterable[tuple[conllu.Token | None, conllu.Token, conllu.Token | None]]:
        """
        Iterate sentence by tuples of previous, current and next tokens.
        :param sentence: CoNLL-U sentence
        :return: Tuples of previous, current and next tokens.
        """
        return zip([None] + sentence[:-1], sentence, sentence[1:] + [None], strict=True)

    @staticmethod
    def parse_file(file_path: str) -> conllu.SentenceList:
        """
        Parse a CoNLL-U file.
        :param file_path: Path of the CoNLL-U file
        :return: CoNLL-U file parsed to sentences
        """
        with open(file_path) as f:
            return conllu.parse(f.read())
