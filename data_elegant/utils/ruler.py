"""
Implementation of ruler
"""
import string

# 移除的script标签
SCRIPT_PATTERN = '<\s*script[^>]*>[^<]*<\s*/\s*script\s*>'
# ---------------------------------------------------------------------

# 移除的style标签
STYLE_PATTERN = '<\s*style[^>]*>[^<]*<\s*/\s*style\s*>'
# ---------------------------------------------------------------------

# 移除的html标签
HTML_TAG_PATTERN = r'<[^>]+>'
# ---------------------------------------------------------------------

# 支持的语言
languages = [
    'AR', 'BN', 'CA', 'EN', 'ES', 'EU', 'FR', 'HI', 'ID', 'PT',
    'UR', 'VI', 'ZH'
]
# ---------------------------------------------------------------------

# 有效文本的最短长度
min_map = {
    'AR': 20,
    'BN': 33,
    'CA': 15,
    'EN': 20,
    'ES': 16,
    'EU': 8,
    'FR': 13,
    'HI': 38,
    'ID': 15,
    'PT': 19,
    'UR': 25,
    'VI': 30,
    'ZH': 1
}
# ---------------------------------------------------------------------


# main_special_characters for special_characters_default
main_characters = string.punctuation + string.digits + string.whitespace

# other_special_characters for special_characters_default
other_characters = (
    '    　    ￼’“”–ー一▬…✦�­£​•€«»°·═'
    '×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖'
    '◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚'
    '゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖'
    '」﴾》'
)
default_set = set(main_characters + other_characters)