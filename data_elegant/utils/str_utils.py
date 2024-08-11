"""
Implementation of str utils
"""


def strip(document, strip_characters):
    """
    Way faster than document.strip(strip_characters)
    since strip_characters is now a set instead of a str,
    and it contains a lot of elements (all the emojis).
    """
    if not document:
        return document

    beg_ind = 0
    end_ind = len(document)

    for i in range(len(document)):
        if document[i] in strip_characters:
            beg_ind += 1
        else:
            break

    for i in range(1, len(document) + 1):
        if document[-i] in strip_characters:
            end_ind -= 1
        else:
            break

    document_stripped = document[beg_ind:end_ind]
    return document_stripped
