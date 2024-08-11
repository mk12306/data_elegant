# -*- coding:utf-8 -*-
import logging

import jieba
import jieba.analyse
import numpy as np

logging.basicConfig(level=logging.INFO)


class SimHash(object):
    """
    计算输入内容的SimHash值
    """

    def sim_hash(self, content: str, stop_word_path: str = None) -> str:
        """
        计算文本的simhash值
        Args:
            content: 文本内容
            stop_word_path: 停用词路径
        Returns:
            simhash 字符串
        """
        seg = jieba.cut(content)
        if stop_word_path:
            jieba.analyse.set_stop_words(stop_word_path)

        # jieba基于TF-IDF提取关键词
        key_words = jieba.analyse.extract_tags("|".join(seg), topK=10, withWeight=True)
        key_list = []
        for feature, weight in key_words:
            weight = int(weight)
            bin_str = self.string_hash(feature)
            temp = []
            for c in bin_str:
                if c == '1':
                    temp.append(weight)
                else:
                    temp.append(-weight)
            key_list.append(temp)
        list_sum = np.sum(np.array(key_list), axis=0)
        if not key_list:
            return '00'
        simhash = ''.join(['1' if i > 0 else '0' for i in list_sum])

        return simhash

    @staticmethod
    def string_hash(source: str) -> str:
        """
        计算字符串的64位sim hash值
        """
        if source == "":
            return '0'
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]
            return str(x)

    @staticmethod
    def get_distance(hashstr1: str, hashstr2: str) -> int:
        """
        计算两个simhash的汉明距离
        Args:
            hashstr1: simhash字符串1
            hashstr2: simhash字符串2

        """
        length = 0
        for index, char in enumerate(hashstr1):
            if char == hashstr2[index]:
                continue
            else:
                length += 1
        return length


if __name__ == '__main__':
    simhash = SimHash()
    s1 = simhash.sim_hash('我想洗一张照片')
    s2 = simhash.sim_hash('我可以洗一张照片吗')

    dis = simhash.get_distance(s1, s2)
    logging.info('dis: {}'.format(dis))
    print('dis: {}'.format(dis))
