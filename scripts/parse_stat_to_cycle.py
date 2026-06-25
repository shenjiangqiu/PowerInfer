import argparse
import json


sample = {
    "total_records": 10912,
    "total_neurons": 120119296,
    "total_selected_neurons": 39341788,
    "up_total_naive_time": 1908848,
    "up_total_asnc_time": 1342208,
    "up_total_iterleave_time": 765856,
    "down_total_interproduct_time_single": 1838120,
    "down_total_interproduct_time_two": 36056,
    "down_total_rowwise_bitserial_time_method_1": 1342208,
    "down_total_rowwise_bitserial_time_method_2": 743299,
}


class CycleSimulation:
    def __init__(self, stat):
        self.stat = stat
        self.bandwidth = 1024 / 8  # bytes / ns 每个stack 16 byte = 1024 bit per ns
        self.enable_gate = False
        self.data_width = 4
        self.row_open = 56  # ns
        self.row_compute = 1024 / 16  # 64 ns

    def compute_total_cycles(self):
        self.gpu_cycle = (
            self.stat["total_neurons"] * 4096 * self.data_width
        ) // self.bandwidth
        if self.enable_gate:
            self.gpu_cycle = self.gpu_cycle * 3
        else:
            self.gpu_cycle = self.gpu_cycle * 2

        self.gpu_cycle_sparse = (
            self.stat["total_selected_neurons"] * 4096 * self.data_width
        ) // self.bandwidth
        self.gpu_cycle_sparse = self.gpu_cycle_sparse * 2

        self.up_dense_row_open = self.stat["up_dense"] * self.row_open
        self.up_dense_compute = self.stat["up_dense"] * self.row_compute
        self.up_total_naive_time_row_open = (
            self.stat["up_total_naive_time"] * self.row_open
        )
        self.up_total_naive_time_compute = self.stat["up_total_naive_time"] * (
            self.row_compute
        )

        self.up_total_asnc_time_row_open = (
            self.stat["up_total_asnc_time"] * self.row_open
        )
        self.up_total_asnc_time_compute = self.stat["up_total_asnc_time"] * (
            self.row_compute
        )

        self.up_total_iterleave_time_row_open = (
            (self.stat["up_total_iterleave_time"]) * self.row_open
        )
        self.up_total_iterleave_time_compute = (
            self.stat["up_total_iterleave_time"]
        ) * (self.row_compute)

        # down naive

        # down sparse
        self.down_dense_row_open = self.stat["down_dense"] * self.row_open
        self.down_dense_compute = self.stat["down_dense"] * self.row_compute
        # down single
        self.down_total_interproduct_time_single_row_open = (
            self.stat["down_total_interproduct_time_single"] * self.row_open
        )
        self.down_total_interproduct_time_single_compute = self.stat[
            "down_total_interproduct_time_single"
        ] * (self.row_compute)
        # down two

        self.down_total_interproduct_time_two_row_open = (
            (self.stat["down_total_interproduct_time_two"]) * self.row_open
        )

        self.down_total_interproduct_time_two_compute = (
            self.stat["down_total_interproduct_time_two"]
        ) * (self.row_compute)

        # bitserial single bank,  要做一次乘法和加法,假设乘法需要 8 * 16 次操作, 加法需要 16次操作,那么一共需要 9 *16次操作,每次 操作 56 ns
        self.down_total_rowwise_bitserial_time_method_1 = (
            self.stat["down_total_rowwise_bitserial_time_method_1"] * 9 * 16 * 56
        )
        self.down_total_rowwise_bitserial_time_method_2 = (
            self.stat["down_total_rowwise_bitserial_time_method_2"] * 9 * 16 * 56
        )


def parse_stat_to_cycle(stat_file, output_file):
    with open(stat_file, "r") as f:
        stat = json.load(f)
        CycleSim = CycleSimulation(stat)
        CycleSim.compute_total_cycles()
        # 将结果写入输出文件
        with open(output_file, "w") as f:
            json.dump(CycleSim.__dict__, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Parse stat file to cycle file")
    parser.add_argument(
        "--stat_file", type=str, required=True, help="Path to the stat file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output cycle file"
    )
    args = parser.parse_args()
    parse_stat_to_cycle(args.stat_file, args.output_file)


if __name__ == "__main__":
    main()
