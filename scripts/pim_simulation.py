import argparse
import argparse
import json
import os
import struct
import sys


from convert_sparse_dump import read_binary


class PimContext:
    def __init__(self):
        self.page_size = 1024  # 1KB
        self.banks = 32 * 32  # 32 banks, 32 channels

        self.activation_width = 4
        self.data_width = 4  # 4 bytes per float

        self.up_dense = 0
        # up.1
        self.up_total_naive_time = 0
        # up.2
        self.up_total_asnc_time = 0
        # up.3.1 单独
        self.up_total_iterleave_time = 0
        # up.3.2 共享
        self.activation_size = 4 * 1024  # 4k elements
        self.neuron_size = 11008  # elements

        self.down_dense = 0
        # down.1
        self.down_total_interproduct_time_single = 0

        # down 2
        self.down_total_interproduct_time_two = 0

        # down.3.1 方法一
        self.down_total_rowwise_bitserial_time_method_1 = 0
        # down.3.2 方法二
        self.down_total_rowwise_bitserial_time_method_2 = 0

        # stats
        self.total_records = 0
        self.total_neurons = 0
        self.total_selected_neurons = 0

        #
        self.last_round_index = None

        self.last_round_index_down = None

    def compute_time(self, total, active, indices: list[int]):
        self.total_records += 1
        self.total_neurons += total
        self.total_selected_neurons += active

        """
        有几种布局模式:
        1. Naive layout: 每次所有的 bank 激活同一行
        2. async layout: 每次 bank 激活不同行
        3. iterleave layout: 同时取 2 个 batch,同时运行
        
        关于 down gate: 4k * 11k, 4k output channel
        1. innerproduct, 11k*11k -> 1, 这种情况下 11k 中的 sparsity 无法轻易跳过,需要一整个 row 都是空才跳过,11k 放在一起
        2. row_wise- bitserial: 11k 个 4k+4k, 4k 放在一起(不行, row 是破坏性的都,无法写回)
        """

        # 4k in a bank, 11k across the banks
        naive_single_neuron_size = self.activation_size * self.data_width

        rows_per_bank = naive_single_neuron_size // self.page_size
        assert (naive_single_neuron_size % self.page_size) == 0, (
            "naive_single_neuron_size must be multiple of page_size"
        )

        self.up_dense += (
            (self.neuron_size + self.banks - 1) // self.banks * rows_per_bank
        )
        # 1. naive layout:所有的bank,只要有一个row 被激活,那么所有的bank 都需要激活这个row,所以只要统计所有的row,然后取max

        valid_rows = [set() for _ in range(32)]
        for i in indices:
            channel_id = (i // 32) % 32
            bank_id = i % 32
            row_id = i // 32 // 32
            valid_rows[channel_id].add(row_id)
            pass
        each_channel_rows = [len(s) for s in valid_rows]
        max_rows = max(each_channel_rows)
        self.up_total_naive_time += max_rows * rows_per_bank

        # 2. async layout: 每个bank可以激活自己的row
        # each bank can activate different rows, so count the number of rows for each bank, and take the max
        rows_per_each_bank = [[0 for _i in range(32)] for _ in range(32)]
        for i in indices:
            channel_id = (i // 32) % 32
            bank_id = i % 32
            rows_per_each_bank[channel_id][bank_id] += 1
        each_channel_max_rows = [max(s) for s in rows_per_each_bank]
        max_rows = max(each_channel_max_rows)

        self.up_total_asnc_time += max_rows * rows_per_bank

        # 3. iterleave layout
        # there are two batch in parallel, so we can merge the same index do the same time, for diffrent index, if they belong to the same bank,
        # they  will be serial,
        if self.last_round_index is None:
            self.last_round_index = indices
        else:
            set_1 = set(self.last_round_index)
            set_2 = set(indices)
            all_set = set_1 | set_2

            rows_per_each_bank = [0 for _i in range(self.banks)]
            for i in all_set:
                bank_id = i % self.banks
                rows_per_each_bank[bank_id] += 1

            self.up_total_iterleave_time += max(rows_per_each_bank) * rows_per_bank

            self.last_round_index = None

        # down gate:

        self.down_dense += (
            (self.neuron_size * self.data_width + self.page_size - 1) // self.page_size
        ) * (self.activation_size // self.banks)
        # inner product:
        # 在做 inner projection 的时候,所有的 output-channel 都是一致的,所以视角只要看到一个bank 就好了
        # 11k在一起, 4k 在banks上均匀分布,11k分布在44行,从中取消一些行不要
        sinle_neuron_size = self.neuron_size * self.data_width  # 44k

        rows_per_bank = (sinle_neuron_size + self.page_size - 1) // self.page_size  # 44
        # 只有一整个 row 都被跳过,这个 row 才不需要加载
        row_index_count = set()
        for i in indices:
            row_index_count.add(i * self.data_width // self.page_size)

        # 最多44
        self.down_total_interproduct_time_single += len(row_index_count) * (
            self.activation_size // self.banks
        )

        # 如果两个batch 同时进行,那么 index 一样的可以共享rowbuffer read,但是无法共享计算,
        row_index_count_set_1 = set()
        row_index_count_set_2 = set()
        if self.last_round_index_down is None:
            self.last_round_index_down = indices
        else:
            for i in self.last_round_index_down:
                row_index_count_set_1.add(i * self.data_width // self.page_size)

            for i in indices:
                row_index_count_set_2.add(i * self.data_width // self.page_size)

            all_set = row_index_count_set_1 | row_index_count_set_2

            self.down_total_interproduct_time_two += (len(all_set)) * (
                self.activation_size // self.banks
            )

            self.last_round_index_down = None

        ## row-wise bitserial
        ## 4k 在一起, 需要累加 11k 个 4k, 然后再 logical die 上面做最终的累加
        ## 把11k 分配到 所有的 bank 上, 每个 bank 上累加
        ## 有两种策略: 1. 4k * 11k 变成 256 * 16 * 11 K,
        # 方法一: 4K 放16行, 11k 放不同的bank, 每行有 16 * 11k/banks, 每个bank 计算完成后得到16行, 需要完成 11k/banks * 16 次累加
        # 方法二: 4K 放不同的16个bank, 那么 bank 被分成banks/16 个组, 11k 放到不同的组上, 每行有 1 * 11k/(banks/16),
        # 每个bank 计算完成后得到一行,需要完成 11k/(banks/16) 次累加: 优点: 每个bank上的11k 分布比较多: 11*16, 可能更平均

        # 方法三: 如果是batch = 2, 在进行row-wise的时候如果两个index不同,在同一个bank上,那么就需要计算两次,如果index相同,那么理论上也要计算两次

        method_1_each_bank_tasks = [0 for _i in range(self.banks)]
        rows_per_bank = (self.activation_size * self.data_width) // self.page_size

        for i in indices:
            bank_id = i % self.banks
            method_1_each_bank_tasks[bank_id] += 1
        self.down_total_rowwise_bitserial_time_method_1 += (
            max(method_1_each_bank_tasks) * rows_per_bank
        )

        rows_per_bank = 1
        group_size = self.activation_size * self.data_width // self.page_size  # 16

        method_2_each_bank_tasks = [0 for _i in range(self.banks // group_size)]
        for i in indices:
            bank_id = i % (self.banks // group_size)
            method_2_each_bank_tasks[bank_id] += 1
        self.down_total_rowwise_bitserial_time_method_2 += (
            max(method_2_each_bank_tasks) * rows_per_bank
        )
        pass

        # 方法3: 如果batch = 2 似乎无法共享,但是可以考虑 bitlevel 共享

    def finish(self):
        if self.last_round_index is not None:
            rows_per_each_bank = [0 for _i in range(self.banks)]
            for i in self.last_round_index:
                bank_id = i % self.banks
                rows_per_each_bank[bank_id] += 1
            rows_per_bank = (self.activation_size * self.data_width) // self.page_size

            self.up_total_iterleave_time += max(rows_per_each_bank) * rows_per_bank
            self.last_round_index = None

        if self.last_round_index_down is not None:
            row_index_count_set_1 = set()
            for i in self.last_round_index_down:
                row_index_count_set_1.add(i * self.data_width // self.page_size)

            self.down_total_interproduct_time += len(row_index_count_set_1) * (
                self.activation_size // self.banks
            )
            self.last_round_index_down = None

    def save_result_json(self, output_file):
        self.finish()
        result = {
            "total_records": self.total_records,
            "total_neurons": self.total_neurons,
            "total_selected_neurons": self.total_selected_neurons,
            "up_dense": self.up_dense,
            "down_dense": self.down_dense,
            "up_total_naive_time": self.up_total_naive_time,
            "up_total_asnc_time": self.up_total_asnc_time,
            "up_total_iterleave_time": self.up_total_iterleave_time,
            "down_total_interproduct_time_single": self.down_total_interproduct_time_single,
            "down_total_interproduct_time_two": self.down_total_interproduct_time_two,
            "down_total_rowwise_bitserial_time_method_1": self.down_total_rowwise_bitserial_time_method_1,
            "down_total_rowwise_bitserial_time_method_2": self.down_total_rowwise_bitserial_time_method_2,
        }

        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)

    pass


def main():
    parser = argparse.ArgumentParser(
        description="Convert PowerInfer binary sparse dump to JSONL"
    )
    parser.add_argument("input", help="Binary dump file (from POWERINFER_DUMP_BINARY)")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,  # pyright: ignore[reportUnusedCallResult]
        help="Neuron activation threshold (default: 0.0)",
    )
    parser.add_argument("-o", "--output", help="Output JSONL file (default: stdout)")  # pyright: ignore[reportUnusedCallResult]
    args = parser.parse_args()
    context = PimContext()

    if os.path.isdir(args.input):
        bins = sorted(f for f in os.listdir(args.input) if f.endswith(".bin"))
        if not bins:
            print(f"No .bin files found in {args.input}", file=sys.stderr)
            sys.exit(1)

        for name in bins:
            bin_path = os.path.join(args.input, name)
            records = read_binary(bin_path)
            for r in records:
                token, layer, batch, n_neurons, scores = r
                active = (scores > args.threshold).nonzero()[0].tolist()

                context.compute_time(n_neurons, len(active), active)

    else:
        records = read_binary(args.input)
        for r in records:
            token, layer, batch, n_neurons, scores = r
            active = (scores > args.threshold).nonzero()[0].tolist()

            context.compute_time(n_neurons, len(active), active)

    context.save_result_json(args.output or sys.stdout)


if __name__ == "__main__":
    main()
