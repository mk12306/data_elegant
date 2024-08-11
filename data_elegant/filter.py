import re
from typing import List

from utils import re_utils, str_utils
from utils.ruler import languages, min_map, default_set


def split_on_whitespace(
    document,
    new_line=False,
    tab=False,
):
    """This method also removes concatenated spaces."""
    sep = [" "] + new_line * ["\n"] + tab * ["\t"]
    sep = "|".join(sep)
    split_document = re.split(sep, document)
    split_document = [_ for _ in split_document if _]
    return split_document


def split(
    content: str,
    lang_dataset_id: str,
    lower_case: bool,
    strip_characters: bool
) -> List[str]:
    """
    return words from content
    Args:
        content: content to be split
        lang_dataset_id: lang dataset id
        lower_case: whether to lower case
        strip_characters: whether to strip characters
    """
    if lang_dataset_id in ['ZH']:
        from algorithms.sentence_piece_tokenizer import SentencePieceTokenizer

        document_normalized = re_utils.replace_digits_with_zeros(content.lower())
        words = SentencePieceTokenizer().tokenize(content=document_normalized, join_on_whitespace=False)
    else:
        words = split_on_whitespace(content, new_line=True, tab=True)

    if lower_case:
        words = [word.lower() for word in words]

    if strip_characters:
        words = [str_utils.strip(word, strip_characters) for word in words]
        words = [_ for _ in words if _]

    return words


def calc_num(content, lang_dataset_id='ZH') -> int:
    """
    word count function
    Args:
        content: content
        lang_dataset_id: language id
    """
    words = split(
        content,
        lang_dataset_id=lang_dataset_id,
        lower_case=False,
        strip_characters=default_set,
    )
    return len(words)


def is_short_text(
    content,
    number_words_min_cutoff=None,
    lang_dataset_id='ZH',
) -> bool:
    """ filter content with word's between number_words_min_cutoff and number_words_max_cutoff """
    if lang_dataset_id not in languages:
        return True

    if number_words_min_cutoff is None:
        number_words_min_cutoff = min_map.get(lang_dataset_id, 10)

    length = calc_num(content, lang_dataset_id=lang_dataset_id)
    cond = number_words_min_cutoff <= length

    return cond


if __name__ == '__main__':
    content = 'hello world for my world'
    print(calc_num(content))
    print(is_short_text(content, number_words_min_cutoff=10))
