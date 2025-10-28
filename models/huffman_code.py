import heapq
import struct
from collections import Counter
import numpy as np
from typing import Union


class HuffmanCodecs:
    def __init__(self, original_data: Union[list, np.ndarray]):
        self.float_data = original_data
        self.bin_data = self.float_to_bin(self.float_data)
        self.freq_table = self.build_frequency_table(self.bin_data)
        self.huffman_tree = self.build_huffman_tree(self.freq_table)
        self.huffman_codes = self.build_huffman_codes(self.huffman_tree)

    def float_to_bin(self, f: Union[float, list]):
        # 将浮点数转换为二进制字符串
        if isinstance(f, list):
            return [self.float_to_bin(fi) for fi in f]
        return format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')

    def bin_to_float(self, b: Union[bytes, list]):
        # 将二进制字符串转换为浮点数
        if isinstance(b, list):
            return [self.bin_to_float(bi) for bi in b]
        return struct.unpack('!f', struct.pack('!I', int(b, 2)))[0]

    def get_bin_data(self):
        return self.bin_data

    def get_freq_table(self):
        return self.freq_table

    def get_huffman_tree(self):
        return self.huffman_tree

    def get_huffman_codes(self):
        return self.huffman_codes

    @staticmethod
    def build_frequency_table(bin_data: Union[list, np.ndarray]):
        # 构建字符频率表
        return Counter(bin_data)

    @staticmethod
    def build_huffman_tree(freq):
        # 构建霍夫曼树
        heap = [[wt, [sym, ""]] for sym, wt in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

    @staticmethod
    def build_huffman_codes(tree):
        # 从霍夫曼树构建编码字典
        huffman_codes = {}
        for tup in tree:
            symbol = tup[0]
            code = tup[1]
            huffman_codes[symbol] = code
        return huffman_codes

    def get_encode_data(self):
        # 对数据进行霍夫曼编码
        return ''.join(self.huffman_codes[char] for char in self.bin_data)

    def get_decode_data(self):
        # 解码霍夫曼编码的数据
        rev_huffman_codes = {code: sym for sym, code in self.huffman_codes.items()}
        decoded_data = []
        current_code = ''
        for bit in self.get_encode_data():
            current_code += bit
            if current_code in rev_huffman_codes:
                decoded_data.append(rev_huffman_codes[current_code])
                current_code = ''
        return self.bin_to_float(decoded_data)

    def get_bit_rate(self) -> float:
        # 计算原始数据的总比特数
        original_bit_count = sum(len(bin_str) for bin_str in self.bin_data)

        # 计算压缩后的数据的总比特数
        compressed_bit_count = len(self.get_encode_data())

        # 计算比特率
        return compressed_bit_count / original_bit_count


from FCM import Focus2dCompressNetwork
import torch
device = 'cuda:7'
x = torch.randn(11, 172, 128, 4, device=device)
model_32 = Focus2dCompressNetwork(num_bits=32).to(device)
model_8 = Focus2dCompressNetwork(num_bits=8).to(device)
x_32 = model_32.encoder(x).flatten().tolist()
x_8 = model_8.encoder(x).flatten().tolist()
huffman_codecs_1 = HuffmanCodecs(x_32)
huffman_codecs_2 = HuffmanCodecs(x_8)
bit_rate_1 = huffman_codecs_1.get_bit_rate()
bit_rate_2 = huffman_codecs_2.get_bit_rate()
print(f'32 bit rate: {bit_rate_1:.2%}')
print(f'8 bit rate: {bit_rate_2:.2%}')
