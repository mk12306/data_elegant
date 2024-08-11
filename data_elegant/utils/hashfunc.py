import hashlib
import struct


def sha1_hash32(data):
    """
     基于SHA1的32-bit哈希函数
    Args:
        data (bytes): 待处理数据
    Returns:
        int: 32-bit整形数据
    """
    assert isinstance(data, bytes), 'data type must be `bytes'
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]


def sha1_hash64(data):
    """
     基于SHA1的64-bit哈希函数
    Args:
        data (bytes): 待处理数据
    Returns:
        int: 64-bit整形数据
    """
    assert isinstance(data, bytes), 'data type must be `bytes'
    return struct.unpack('<Q', hashlib.sha1(data).digest()[:8])[0]
