from dataclasses import dataclass


@dataclass
class PDTTagHandler:
    """
    Class to handle Prague Dependency Treebank positional tags.
    """

    # Constants
    # Positions
    POS_POSITION: int = 0
    GENDER_POSITION: int = 2
    NUMBER_POSITION: int = 3
    CASE_POSITION: int = 4

    # Parts of speech
    NOUN_POS: str = "N"

    # Genders
    MASCULINE_INANIMATE_GENDER: str = "I"
    MASCULINE_ANIMATE_GENDER: str = "M"
    FEMININE_GENDER: str = "F"
    NEUTER_GENDER: str = "N"

    # Numbers
    SINGULAR_NUMBER: str = "S"

    # Cases
    FIRST_CASE: str = "1"

    @classmethod
    def get_all_genders(cls) -> list[str]:
        """
        Return all gender tags.
        :return: All gender tags
        """
        return [cls.MASCULINE_ANIMATE_GENDER, cls.MASCULINE_INANIMATE_GENDER, cls.FEMININE_GENDER, cls.NEUTER_GENDER]

    @classmethod
    def is_base_noun(cls, tag: str) -> bool:
        """
        Test whether a given tag is a tag of a base noun (defined gender, singular number, first case).
        :param tag: Tag to test
        :return: Whether a given tag is a tag of a base noun
        """
        pos = tag[cls.POS_POSITION]
        gender = tag[cls.GENDER_POSITION]
        number = tag[cls.NUMBER_POSITION]
        case = tag[cls.CASE_POSITION]

        return (
            pos == cls.NOUN_POS
            and gender in {cls.MASCULINE_INANIMATE_GENDER, cls.MASCULINE_ANIMATE_GENDER, cls.FEMININE_GENDER, cls.NEUTER_GENDER}
            and number == cls.SINGULAR_NUMBER
            and case == cls.FIRST_CASE
        )

    @classmethod
    def extract_ref_gender(cls, tag: str) -> str:
        """
        Extract reference gender from a given tag.
        :param tag: Tag to extract reference gender
        :return: Reference gender
        """
        return tag[cls.GENDER_POSITION]
