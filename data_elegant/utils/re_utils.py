import re


def replace_digits_with_zeros(content: str) -> str:
    """
    return content whose digits will be replaced by zeros
    Args:
        content: content to be replaced
    """
    return re.sub(r'\d', '0', content)