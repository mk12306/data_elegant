import opencc
import emoji
import re

from utils.ruler import STYLE_PATTERN, SCRIPT_PATTERN, HTML_TAG_PATTERN

convert_trad2sim = opencc.OpenCC("t2s")
convert_sim2trad = opencc.OpenCC("s2t")


def trad_to_sim(content):
    """
    return content simplified
    Arg:
        content: 待转换的字符串
    """
    return convert_trad2sim.convert(content)


def sim_to_trad(content):
    """
    return content traditional
    Arg:
        content: 待转换的字符串
    """
    return convert_sim2trad.convert(content)


def replace_emoji(content, value=''):
    """
    return content replaced emoji
    Arg:
        content: 待替换的字符串
        value: 替换的字符, 默认空字符串
    """
    return emoji.replace_emoji(content, replace=value)


def remove_web_identifiers(content):
    """
    return content web identifiers, language support ZH
    Arg:
        content: 待转换的字符串
    """
    script_pattern_re = re.compile(SCRIPT_PATTERN, re.I)
    style_pattern_re = re.compile(STYLE_PATTERN, re.I)
    html_tag_pattern_re = re.compile(HTML_TAG_PATTERN, re.S)

    content = script_pattern_re.sub("", content)
    content = style_pattern_re.sub("", content)
    content = html_tag_pattern_re.sub("", content)

    lines = [
        line for line in content.split('\n') if len(line.strip()) > 0
    ]

    return "\n".join(lines)
