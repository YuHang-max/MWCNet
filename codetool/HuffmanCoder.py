import time
from typing import Union
import struct

import torch
from torch import nn


class Node:
    def __init__(self, symbol: int, prob: float, left=None, right=None, code: str = None):
        self._symbol = symbol
        self._prob = prob
        self._left = left
        self._right = right
        self._code = code

    def __repr__(self):
        return (f"Symbol: {self.symbol}, Prob: {self.prob}, Left: {self.left}, Right: {self.right},"
                f"Code: {self.code}")

    @property
    def symbol(self):
        return self._symbol

    @property
    def prob(self):
        return self._prob

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def code(self):
        return self._code

    @symbol.setter
    def symbol(self, value: int):
        if value >= 0:
            self._symbol = value
        else:
            raise ValueError(f'Expect symbol to be positive integer, but found: {value}.')

    @prob.setter
    def prob(self, value: float):
        if 0 <= value <= 1:
            self._prob = value
        else:
            raise ValueError(f'Expect probability to be between 0 and 1, but found: {value}.')

    @left.setter
    def left(self, node: object):
        if isinstance(node, Node):
            self._left = node
        else:
            raise ValueError(f'Expect node to be Node, but found: {type(node)}.')

    @right.setter
    def right(self, node: object):
        if isinstance(node, Node):
            self._right = node
        else:
            raise ValueError(f'Expect node to be Node, but found: {type(node)}.')

    @code.setter
    def code(self, value: str):
        if set(value).issubset({"0", "1"}):
            self._code = value
        else:
            raise ValueError(f'Expect code to be a string of 0 or 1, but found: {value}.')


class HuffmanTree:
    def __init__(self, probs: list):
        self.probs = probs
        self.root = self.create_tree()
        self.preorder_traversal(self.root, "")

    def __len__(self):
        return len(self.probs)

    def __call__(self, *args, **kwargs):
        return self.root

    def create_tree(self):
        nodes = [Node(idx, prob) for idx, prob in enumerate(self.probs)]
        while True:
            nodes = sorted(nodes, key=lambda node: (node.prob, node.symbol))
            lower_node, upper_node = nodes[0], nodes[1]
            new_node = Node(symbol=-1, prob=lower_node.prob + upper_node.prob, left=lower_node, right=upper_node)
            nodes.remove(lower_node)
            nodes.remove(upper_node)
            nodes.append(new_node)
            if len(nodes) == 1:
                break
        return nodes[0]

    def preorder_traversal(self, node: Node, code: str = ""):
        node.code = code
        if node.left and node.right:
            self.preorder_traversal(node.left, code + "0")
            self.preorder_traversal(node.right, code + "1")


class HuffmanCodex:
    def __init__(self):
        self.tree = None
        self.leaf_nodes = []

    def preorder_traversal(self, node: Node):
        if node.left and node.right:
            self.preorder_traversal(node.left)
            self.preorder_traversal(node.right)
        else:
            self.leaf_nodes.append(node)

    def encode(self, symbol: Union[list, int], probs: list, bit_stream: str) -> str:
        self.tree = HuffmanTree(probs)
        self.leaf_nodes = []
        self.preorder_traversal(self.tree.root)
        if isinstance(symbol, int):
            for node in self.leaf_nodes:
                if node.symbol == symbol:
                    return bit_stream + node.code
        elif isinstance(symbol, list):
            self.leaf_nodes = sorted(self.leaf_nodes, key=lambda node: node.symbol)
            for item in symbol:
                bit_stream += self.leaf_nodes[item].code
            return bit_stream

    def decode(self, probs: list, bit_stream: str, length: int) -> tuple:
        self.tree = HuffmanTree(probs)
        self.leaf_nodes = []
        self.preorder_traversal(self.tree.root)
        decoded_symbols = []
        for j in range(length):
            for i, node in enumerate(self.leaf_nodes):
                if bit_stream.startswith(node.code):
                    bit_stream = bit_stream[len(node.code):]
                    decoded_symbols.append(node.symbol)
                    break
                if i >= len(self.leaf_nodes) - 1:
                    raise BrokenPipeError(f"Expect bit stream to be decoded correctly, but found: {bit_stream}.")

        if length == 1:
            return decoded_symbols[0], bit_stream
        else:
            return decoded_symbols, bit_stream


class EntropyCodex:
    def __init__(
            self, quantize_bit: int = 8, device: str = "cpu", likelihood_lower_bound: float = 1e-8,
            model: nn.Module = None
    ):
        self.huffman_codex = HuffmanCodex()
        self.quantize_bit = quantize_bit
        self.device = device
        self.likelihood_lower_bound = likelihood_lower_bound
        self.model = model

    def set_model(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def get_freq_table(self, int_tensor: torch.Tensor) -> list:
        counts = torch.bincount(int_tensor.flatten())
        probs = counts.to(torch.float32) / int_tensor.numel()
        return torch.clamp(probs, min=self.likelihood_lower_bound, max=1.0).tolist()

    @staticmethod
    def read_bits(bit_stream: str, length: int) -> tuple:
        return bit_stream[:length], bit_stream[length:]

    @staticmethod
    def int8_to_binary(value: int) -> str:
        assert 0 <= value <= 2**8 - 1, f"Expect value of type int8 to be within [0, 255], but found: {value}."
        binary = bin(value)[2:]
        return "0" * (8 - len(binary)) + binary

    @staticmethod
    def binary_to_int8(binary_str: str) -> int:
        assert set(binary_str).issubset({"0", "1"}), f"Expect binary string of 0 or 1, but found: {binary_str}."
        return int(binary_str, 2)

    @staticmethod
    def float32_to_binary(value: float) -> str:
        packed = struct.pack('!f', value)
        return ''.join(f'{byte:08b}' for byte in packed)

    @staticmethod
    def binary_to_float32(binary: str) -> float:
        assert len(binary) == 32, "Binary string must be 32 bits long."
        int_value = int(binary, 2)
        packed = int_value.to_bytes(4, byteorder='big')
        return struct.unpack('!f', packed)[0]

    @staticmethod
    def to_symbol(
            tensor_: torch.Tensor, min_: Union[torch.Tensor, float], quantize_step: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        int_tensor = (tensor_ - min_) / quantize_step
        return int_tensor.round().to(torch.int)

    @staticmethod
    def quantize(
            int_tensor: torch.Tensor, min_: Union[torch.Tensor, float], quantize_step: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        return int_tensor * quantize_step + min_

    @staticmethod
    def compute_bpp(binary_length: int, x_shape: torch.Size) -> float:
        batch_size, channels, height, width = x_shape
        return binary_length / (batch_size * channels * height * width)

    def cat_full_image(self, tensor_: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = tensor_.shape
        reshaped_tensor = torch.zeros(
            (height, width * batch_size, channels), device=self.device, dtype=torch.float32
        )
        for i in range(batch_size):
            reshaped_tensor[:, i * width: (i + 1) * width, :] = tensor_[i].permute(1, 2, 0)
        return reshaped_tensor

    def compute_psnr(self, original_x: torch.Tensor, restored_x: torch.Tensor) -> float:
        original_x = self.cat_full_image(original_x)
        restored_x = self.cat_full_image(restored_x)
        original_x = original_x.reshape(-1, original_x.shape[-1])
        restored_x = restored_x.reshape(-1, restored_x.shape[-1])
        mse = torch.mean((original_x - restored_x) ** 2, dim=0, keepdim=True)
        max_val, _ = torch.max(original_x, dim=0, keepdim=True)
        return torch.mean(10 * torch.log10((max_val ** 2) / mse)).item()
    
    def compute_sam(self, original_x: torch.Tensor, restored_x: torch.Tensor) -> float:
        original_x = self.cat_full_image(original_x)
        restored_x = self.cat_full_image(restored_x)
        num = torch.sum(original_x * restored_x, dim=2)
        den = torch.sqrt(torch.sum(restored_x ** 2, dim=2) * torch.sum(original_x ** 2, dim=2))
        
        cos_theta = num / den
        cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
        angle = torch.acos(cos_theta)
        angle_deg = torch.rad2deg(angle)
        return (torch.sum(angle_deg) / (original_x.shape[0] * original_x.shape[1])).item()
    
    def compute_rmse(self, original_x: torch.Tensor, restored_x: torch.Tensor) -> float:
        original_x = self.cat_full_image(original_x)
        restored_x = self.cat_full_image(restored_x)
        
        mse = torch.mean((restored_x - original_x) ** 2, dim=(0, 1))
        rmse_total = torch.sqrt(torch.mean(mse))
        return rmse_total.item()

    def get_gaussian_freq_table(
            self, mu:torch.Tensor, sigma:torch.Tensor, min_: torch.Tensor, quantize_step: torch.Tensor
    ) -> list:
        dist = torch.distributions.Normal(mu, sigma)
        samples = torch.linspace(
            0, 2 ** self.quantize_bit - 1, steps=2 ** self.quantize_bit, device=self.device,
            dtype=torch.float32
        ) * quantize_step + min_
        likelihoods = dist.cdf(samples + 0.5 * quantize_step) - dist.cdf(samples - 0.5 * quantize_step)
        return torch.clamp(likelihoods, self.likelihood_lower_bound, 1.0).tolist()

    def get_gaussian_freq_tables(
            self, mu: torch.Tensor, sigma: torch.Tensor, min_: torch.Tensor, quantize_step: torch.Tensor
    ) -> list:
        dist = torch.distributions.Normal(mu, sigma)
        samples = torch.linspace(
            0, 2 ** self.quantize_bit - 1, steps=2 ** self.quantize_bit, device=self.device,
            dtype=torch.float32
        ) * quantize_step + min_
        likelihoods = dist.cdf(samples + 0.5 * quantize_step) - dist.cdf(samples - 0.5 * quantize_step)
        if len(likelihoods.shape) == 4:
            likelihoods = likelihoods.unsqueeze(3)
        return torch.clamp(likelihoods, self.likelihood_lower_bound, 1.0).tolist()

    def compress_list(
            self, int_tensor: torch.Tensor, freq_table: list, bit_stream: str,
    ) -> str:
        symbols = int_tensor.flatten().tolist()
        return self.huffman_codex.encode(symbols, freq_table, bit_stream)

    def decompress_list(
            self, freq_table: list, bit_stream: str, min_: torch.Tensor, quantize_step: torch.Tensor, shape: tuple
    ):
        B, C, H, W = shape
        symbols, bit_stream = self.huffman_codex.decode(freq_table, bit_stream, B * C * H * W)
        int_tensor = torch.tensor(symbols, device=self.device, dtype=torch.float32).reshape(*shape)
        return int_tensor * quantize_step + min_, bit_stream

    def compress_points(
            self, int_tensor: torch.Tensor, freq_tables: list, bit_stream: str,
    ) -> str:
        B, C, H, W = int_tensor.shape
        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        symbol = int_tensor[b, c, h, w].item()
                        freq_table = freq_tables[b][c][h][w]
                        len1 = len(bit_stream)
                        bit_stream = self.huffman_codex.encode(symbol, freq_table, bit_stream)
                        len2 = len(bit_stream)
                        if c == 9 and h == 19:
                            dic = [{"symbol": idx, "prob": item} for idx, item in enumerate(freq_tables[0][9][19][0])]
                            dic = sorted(dic, key=lambda x: (x["prob"], x["symbol"]))
                            print(symbol)
                            print([item["symbol"] for idx, item in enumerate(dic)])
                            print(bit_stream[-(len2 - len1):])
        return bit_stream

    def decompress_points(
            self, freq_tables: list, bit_stream: str, min_: torch.Tensor, quantize_step: torch.Tensor, shape: tuple
    ):
        B, C, H, W = shape
        int_tensor = torch.zeros(shape, device=self.device, dtype=torch.float32)
        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):  # 596  1,27,64,1  [0, 9, 19, 0]
                        freq_table = freq_tables[b][c][h][w]
                        len1 = len(bit_stream)
                        bit_stream1 = bit_stream
                        symbol, bit_stream = self.huffman_codex.decode(freq_table, bit_stream, 1)
                        len2 = len(bit_stream)
                        int_tensor[b, c, h, w] = symbol
                        if c == 9 and h == 19:
                            dic = [{"symbol": idx, "prob": item} for idx, item in enumerate(freq_tables[0][9][19][0])]
                            dic = sorted(dic, key=lambda x: (x["prob"], x["symbol"]))
                            print(symbol)
                            print([item["symbol"] for idx, item in enumerate(dic)])
                            print(bit_stream1[:(len1 - len2)])
        # print(int_tensor[0, 9, 19, 0].item())
        # print(int_tensor.to(torch.int).flatten().tolist())
        return int_tensor * quantize_step + min_, bit_stream

    def run_no_hyper(self, x: torch.Tensor, norm_max: torch.Tensor, norm_min: torch.Tensor) -> tuple:
        if self.quantize_bit == 32:
            start_time = time.time()
            y = self.model.encoder(x)
            end_time = time.time()
            if isinstance(y, tuple):
                y = y[0]
            print(f"执行耗时：\n{end_time - start_time}")
            restored = self.model.decoder(y)
            restored = restored * (norm_max - norm_min) + norm_min
            x = x * (norm_max - norm_min) + norm_min
            bpp = y.numel() * 32 / x.numel()
            psnr = self.compute_psnr(x, restored)
            sam = self.compute_sam(x, restored)
            rmse = self.compute_rmse(x, restored)
            return restored, bpp, psnr, sam, rmse
        x = x.contiguous()
        start_time = time.time()
        y = self.model.encoder(x)
        if isinstance(y, tuple):
            y, _, int_y, max_, min_ = y
            int_y = int_y.to(torch.int)
        else:
            max_, min_ = torch.max(y), torch.min(y)
            quantize_step = (max_ - min_) / (2 ** self.quantize_bit - 1)
            int_y = self.to_symbol(y, min_, quantize_step)
        B, C, H, W = y.shape
        binary_b = self.int8_to_binary(B)
        binary_c = self.int8_to_binary(C)
        binary_h = self.int8_to_binary(H)
        binary_w = self.int8_to_binary(W)
        binary_max = self.float32_to_binary(max_.item())
        binary_min = self.float32_to_binary(min_.item())
        bit_stream = binary_b + binary_c + binary_h + binary_w + binary_max + binary_min
        freq_table = self.get_freq_table(int_y)
        for i, freq in enumerate(freq_table):
            bit_stream += self.float32_to_binary(freq)
        bit_stream = self.compress_list(int_y, freq_table, bit_stream)
        end_time = time.time()
        print(f"执行耗时：\n{end_time - start_time}")
        binary_length = len(bit_stream)

        binary_b, bit_stream = self.read_bits(bit_stream, 8)
        binary_c, bit_stream = self.read_bits(bit_stream, 8)
        binary_h, bit_stream = self.read_bits(bit_stream, 8)
        binary_w, bit_stream = self.read_bits(bit_stream, 8)
        binanry_max, bit_stream = self.read_bits(bit_stream, 32)
        binanry_min, bit_stream = self.read_bits(bit_stream, 32)
        B = self.binary_to_int8(binary_b)
        C = self.binary_to_int8(binary_c)
        H = self.binary_to_int8(binary_h)
        W = self.binary_to_int8(binary_w)
        max_ = self.binary_to_float32(binanry_max)
        min_ = self.binary_to_float32(binanry_min)
        max_ = torch.tensor(max_, device=self.device, dtype=torch.float32)
        min_ = torch.tensor(min_, device=self.device, dtype=torch.float32)
        quantize_step = (max_ - min_) / (2 ** self.quantize_bit - 1)
        freq_table = []
        for i in range(2 ** self.quantize_bit):
            binary_prob, bit_stream = self.read_bits(bit_stream, 32)
            freq_table.append(self.binary_to_float32(binary_prob))
        decoded, bit_stream = self.decompress_list(
            freq_table, bit_stream, min_, quantize_step, (B, C, H, W)
        )
        assert len(bit_stream) == 0, f"Expect all bits to be decoded, but found {len(bit_stream)} left."
        restored = self.model.decoder(decoded)
        
        int_tensor = self.to_symbol(restored, torch.min(restored), (torch.max(restored) - torch.min(restored)) / (2 ** self.quantize_bit - 1))
        restored = self.quantize(int_tensor, torch.min(restored), (torch.max(restored) - torch.min(restored)) / (2 ** self.quantize_bit - 1))
        
        restored = restored * (norm_max - norm_min) + norm_min
        x = x * (norm_max - norm_min) + norm_min
        bpp = self.compute_bpp(binary_length, x.shape)
        psnr = self.compute_psnr(x, restored)
        sam = self.compute_sam(x, restored)
        rmse = self.compute_rmse(x, restored)
        return restored, bpp, psnr, sam, rmse

    def run_hyper(self, x: torch.Tensor) -> tuple:
        y = self.model.encoder(x)
        z = self.model.hyper_prior.hyper_analysis(y)
        B_z, C_z, H_z, W_z = z.shape
        max_y, min_y = torch.max(y), torch.min(y)
        max_z = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        min_z = torch.tensor(-1.0, device=self.device, dtype=torch.float32)
        quantize_step_y = (max_y - min_y) / (2 ** self.quantize_bit - 1)
        quantize_step_z = (max_z - min_z) / (2 ** self.quantize_bit - 1)
        int_z = self.to_symbol(z, min_z, quantize_step_z)
        int_y = self.to_symbol(y, min_y, quantize_step_y)
        # print(int_y[0, 9, 19, 0].item())
        # print(int_y.flatten().tolist())
        binary_b_z = self.int8_to_binary(B_z)
        binary_c_z = self.int8_to_binary(C_z)
        binary_h_z = self.int8_to_binary(H_z)
        binary_w_z = self.int8_to_binary(W_z)
        binary_max_y = self.float32_to_binary(max_y.item())
        binary_min_y = self.float32_to_binary(min_y.item())
        bit_stream = binary_b_z + binary_c_z + binary_h_z + binary_w_z + binary_max_y + binary_min_y
        mu_z = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        sigma_z = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        freq_table_z = self.get_gaussian_freq_table(mu_z, sigma_z, min_z, quantize_step_z)
        bit_stream = self.compress_list(int_z, freq_table_z, bit_stream)
        quantized_z = self.quantize(int_z, min_z, quantize_step_z)
        # print(quantized_z.flatten().tolist())
        mu_y, sigma_y = self.model.hyper_prior.hyper_synthesis(quantized_z)
        # print(mu_y.flatten().tolist())
        freq_tables_y = self.get_gaussian_freq_tables(mu_y, sigma_y, min_y, quantize_step_y)
        dic = [{"symbol": idx, "prob": item} for idx, item in enumerate(freq_tables_y[0][9][19][0])]
        dic = sorted(dic, key=lambda x: (x["prob"], x["symbol"]))
        # print([item["symbol"] for idx, item in enumerate(dic)])
        bit_stream = self.compress_points(int_y, freq_tables_y, bit_stream)
        binary_length = len(bit_stream)

        binary_b_z, bit_stream = self.read_bits(bit_stream, 8)
        binary_c_z, bit_stream = self.read_bits(bit_stream, 8)
        binary_h_z, bit_stream = self.read_bits(bit_stream, 8)
        binary_w_z, bit_stream = self.read_bits(bit_stream, 8)
        binary_max_y, bit_stream = self.read_bits(bit_stream, 32)
        binary_min_y, bit_stream = self.read_bits(bit_stream, 32)
        B_z = self.binary_to_int8(binary_b_z)
        C_z = self.binary_to_int8(binary_c_z)
        H_z = self.binary_to_int8(binary_h_z)
        W_z = self.binary_to_int8(binary_w_z)
        max_y = self.binary_to_float32(binary_max_y)
        min_y = self.binary_to_float32(binary_min_y)
        max_z = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        min_z = torch.tensor(-1.0, device=self.device, dtype=torch.float32)
        max_y = torch.tensor(max_y, device=self.device, dtype=torch.float32)
        min_y = torch.tensor(min_y, device=self.device, dtype=torch.float32)
        quantize_step_y = (max_y - min_y) / (2 ** self.quantize_bit - 1)
        quantize_step_z = (max_z - min_z) / (2 ** self.quantize_bit - 1)
        mu_z = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        sigma_z = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        freq_table_z = self.get_gaussian_freq_table(mu_z, sigma_z, min_z, quantize_step_z)
        decoded_z, bit_stream = self.decompress_list(
            freq_table_z, bit_stream, min_z, quantize_step_z, (B_z, C_z, H_z, W_z)
        )
        # print(decoded_z.flatten().tolist())
        mu_y, sigma_y = self.model.hyper_prior.hyper_synthesis(decoded_z)
        # print(mu_y.flatten().tolist())
        freq_tables_y = self.get_gaussian_freq_tables(mu_y, sigma_y, min_y, quantize_step_y)
        dic = [{"symbol": idx, "prob": item} for idx, item in enumerate(freq_tables_y[0][9][19][0])]
        dic = sorted(dic, key=lambda x: (x["prob"], x["symbol"]))
        # print([item["symbol"] for idx, item in enumerate(dic)])
        decoded_y, bit_stream = self.decompress_points(
            freq_tables_y, bit_stream, min_y, quantize_step_y, tuple(mu_y.shape)
        )
        assert len(bit_stream) == 0, f"Expect all bits to be decoded, but found {len(bit_stream)} left."
        restored = self.model.decoder(decoded_y)
        bpp = self.compute_bpp(binary_length, x.shape)
        psnr = self.compute_psnr(x, restored)
        return restored, bpp, psnr


if __name__ == '__main__':
    from trainOps import *
    from dataset_ori import *
    from comparison_methods.URUCH2 import RSCC_AHTH2
    from comparison_methods.BTCNet import BTCNet
    from comparison_methods.RSCC2 import RSCC2, RSCC3
    from CTCSN.model.CTCSN import CTCSN
    from comparison_methods.model import DCSN
    from comparison_methods.LiteRecon_3conv import LiteRecon_3conv
    from comparison_methods.LiteRecon_BTC_detail import LiteRecon_BTC_detail
    from comparison_methods.LiteRecon_MSFB import LiteRecon_MSFA
    from comparison_methods.LiteRecon_CAFE import LiteRecon_CAFE
    
    
    def quantize(tensor_: torch.Tensor, quant_bit: int = 8) -> torch.Tensor:
        max_, min_ = torch.max(tensor_), torch.min(tensor_)
        quant_step = (max_ - min_) / (2 ** quant_bit - 1)
        tensor_ = (tensor_ - min_) / quant_step
        tensor_ = tensor_.round().to(torch.int)
        return tensor_ * quant_step + min_
    
    quant_bit = 8
    device = "cuda:7"
    # model = LiteRecon_3conv().to(device)
    # model.load_state_dict(torch.load("./rscc_param/LiteRecon_3conv_best.pth"))
    
    # model = LiteRecon_BTC_detail(172, 27, ).to(device)
    # model.load_state_dict(torch.load("./rscc_param/LiteRecon_BTC_detail_best.pth"))
    
    # model = LiteRecon_MSFA(172, 27, 32).to(device)
    # model.load_state_dict(torch.load("./rscc_param/LiteRecon_MSFA_best.pth"))
    
    # model = LiteRecon_CAFE(172, 27, 32).to(device)
    # model.load_state_dict(torch.load("./rscc_param/LiteRecon_CAFE_best.pth"))
    
    model = RSCC3(172, 27, 32, quant_bit).to(device)
    # model.load_state_dict(torch.load(f"./rscc_param/RSCC3_{quant_bit}bit_best.pth"))
    model.load_state_dict(torch.load(f"./rscc_param/RSCC3_32bit_best.pth"))
    for param in model.encoder.parameters():
        param.data = quantize(param.data, quant_bit)
    # model.load_state_dict(torch.load("./rscc_param/RSCC_4780_epoch.pth"), strict=False)
    
    # model = CTCSN(cr=1, bit_num=32).to(device)
    # model.load_state_dict(torch.load("./ctcsn_param/CTCSN_33080_epoch.pth"))
    
    # model = DCSN(cr=1).to(device)
    # model.load_state_dict(torch.load("./dcsn_param/RSCC_4880_epoch.pth"))
    
    # model = BTCNet(bit_num=32).to(device)
    # model.load_state_dict(torch.load("./btc_param/BTCNet_best.pth"))
    if isinstance(model, RSCC3):
        params = 0
        params += sum([param.numel() for param in model.encoder.weight]) * quant_bit
        params += sum([param.numel() for param in model.encoder.bias]) * quant_bit
        params += model.encoder.p_relu.weight.numel() * 32
        params += 2 * 2 * 32
        print("total params: ", params)
    valfn = loadTxt('testpath/5HSI_test.txt')
    val_loader = DataLoader(dataset_h5(valfn, mode='Validation', root='', return_min_max=True),
                            batch_size=1, shuffle=False, pin_memory=False)
    x = None
    for ind2, (vx, vfn, min_x, max_x) in enumerate(val_loader):
        vx = vx.view(vx.size()[0] * vx.size()[1], vx.size()[2], vx.size()[3],
                     vx.size()[4])
        vx = vx.to(device).permute(0, 3, 1, 2).float()
        x = vx.to(device)
        if ind2 == 0:
            norm_max = max_x.item()
            norm_min = min_x.item()
            # x = vx.to(device)
            break
    x = quantize(x, quant_bit)
    # print(x[20:24, ...].flatten().tolist())
    # print(x.shape)
    # print(x.flatten().tolist()[-20:])
    model.quant_bit = quant_bit
    coder = EntropyCodex(device=device, quantize_bit=quant_bit)
    coder.set_model(model)
    # restored, bpp, psnr = coder.run_no_hyper(x)
    restored, bpp, psnr, sam, rmse = coder.run_no_hyper(x, norm_max=norm_max, norm_min=norm_min)
    # bpp, psnr = coder.run_freq(x)
    print(f"bpp, psnr, sam, rmse: \n"
          f"{bpp:.12f} \n"
          f"{psnr:.12f} \n"
          f"{sam:.12f} \n"
          f"{rmse:.12f}")



