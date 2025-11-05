#!/usr/bin/env python
#
# Autotune JVM flags to optimize SPECpower2008 performance
#

import argparse
import sys
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
from pathlib import Path
import os
import subprocess
import re

# JVM 参数定义
JVM_PARAMS = [
    # (参数名, 最小值, 最大值, 步长)
    ('TargetSurvivorRatio', 50, 99, 1),
    ('ParallelGCThreads', 1, 8, 1),
    ('AllocatePrefetchDistance', 128, 512, 32),
    ('AllocatePrefetchLines', 1, 64, 2),
    ('InitialTenuringThreshold', 1, 10, 1),
    ('MaxTenuringThreshold', 1, 16, 1),
    ('InlineSmallCode', 1000, 20000, 1000),
    ('MaxInlineSize', 100, 500, 50),
    ('FreqInlineSize', 1000, 10000, 500),
    ('UseAVX', 0, 3, 1),
    ('LoopUnrollLimit', 1, 500, 5),
    ('InitialHeapSize', 2500*1024*1024, 3500*1024*1024, 100*1024*1024),
    ('NewRatio', 1, 10, 1),
    ('SurvivorRatio', 1, 20, 2),
]

# 布尔类型的 JVM 参数
JVM_FLAGS = [
    'UseParallelGC',
    'CheckIntrinsics',
    'OptimizeFill',
    'AggressiveHeap',
    'AlwaysPreTouch',
    'TieredCompilation',
    'UseFPUForSpilling',
    'UseLargePages',
    'UseHugeTLBFS',
    'UseTransparentHugePages',
    'AggressiveOpts',
]

def executable_file(path):
    path_obj = Path(path)
    if not path_obj.exists():
        raise argparse.ArgumentTypeError(f"file not exists: {path}")
    if not path_obj.is_file():
        raise argparse.ArgumentTypeError(f"not a regular file: {path}")
    if not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(f"file not executable: {path}")
    return os.path.abspath(path_obj)


class SPECpowerTuner(MeasurementInterface):

    def manipulator(self):
        """
        Define the search space for JVM parameters
        """
        manipulator = ConfigurationManipulator()

        # 添加数值型参数
        for param, min_val, max_val, step in JVM_PARAMS:
            manipulator.add_parameter(
                IntegerParameter(param, min_val, max_val)
            )

        # 添加布尔型参数（启用/禁用）
        for flag in JVM_FLAGS:
            manipulator.add_parameter(
                EnumParameter(flag, ['on', 'off'])
            )

        return manipulator

    def build_java_opts(self, cfg):
        """
        根据配置构建 JVMOPTIONS 环境变量
        """
        java_opts = []

        # 处理数值型参数
        for param, min_val, max_val, step in JVM_PARAMS:
            value = cfg[param]
            java_opts.append(f"-XX:{param}={value}")
            if param == 'InitialHeapSize':
                java_opts.append(f"-XX:MaxHeapSize={value}")

        # 处理布尔型参数
        for flag in JVM_FLAGS:
            if cfg[flag] == 'on':
                java_opts.append(f"-XX:+{flag}")
            else:
                java_opts.append(f"-XX:-{flag}")

        return ' '.join(java_opts)

    def run(self, desired_result, input, limit):
        """
        Run SPECpower2008 with the given JVM configuration
        """
        if self.args.trace_level > 2:
            print("-----------------------------------------------------------------")
        cfg = desired_result.configuration.data

        # 构建 JVMOPTIONS
        java_opts = self.build_java_opts(cfg)

        # 设置环境变量
        env = os.environ.copy()
        env['JVMOPTIONS'] = java_opts
        env['PRESET_OPTIONS'] = '-Xmixed'

        try:
            # 运行 SPECpower2008 基准测试
            benchmark_script = self.args.benchmark_script
            cmd = f"{benchmark_script}"

            run_result = self.call_program(cmd, env=env)

            if run_result['returncode'] != 0:
                print(f"Run failed with return code: {run_result['returncode']}")
                print("Test script:", cmd)
                print("STDOUT:", run_result['stdout'])
                print("STDERR:", run_result['stderr'])
                return Result(time=0)

            # 解析性能指标
            if self.args.trace_level > 1:
                print("STDOUT from script:", run_result['stdout'])
                print("STDERR from script:", run_result['stderr'])
            performance = self.parse_specpower_output(run_result['stdout'])
            if self.args.trace_level > 0:
                print(f"Test result is {performance} for testing configuration: {java_opts}")
            return Result(time=1.0/performance)

        except Exception as e:
            print(f"Error during run: {e}")
            return Result(time=0)

    def parse_specpower_output(self, output):
        """
        解析 SPECpower2008 的输出，提取性能指标
        """
        output = output.decode('utf-8') if isinstance(output, bytes) else output
        lines = output.split('\n')
        pattern = r"ssj_ops@100%\s*=\s*([\d,]+)"
        for line in lines:
            if ' run; ssj_ops@100%' in line:
                try:
                    line = line.strip()
                    if self.args.trace_level > 2:
                        print(f"Output line: {line}")
                    match = re.search(pattern, line)
                    if match:
                        performance = match.group(1).replace(',', '')
                        return int(performance)
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
                    continue
        print("Warning: Could not parse performance from output")
        return int(sys.maxsize)

    def save_final_config(self, configuration):
        """
        Save the best configuration found
        """
        best_opts = self.build_java_opts(configuration.data)
        print(f"\nBest JVM configuration found:")
        print(f"JVMOPTIONS='{best_opts}'")

        # 保存到文件
        with open('best_jvm_config.txt', 'w') as f:
            f.write(f"JVMOPTIONS='{best_opts}'\n")

        if self.args.trace_level > 1:
            print("Configuration saved to best_jvm_config.txt")


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    argparser.add_argument(
        '--benchmark-script',
        type=executable_file,
        required=True,
        help='Path to the SPECpower2008 benchmark run script'
    )
    argparser.add_argument(
        '--trace-level', type=int, default=1,
        help='Level of tracing for debugging purposes'
    )
    args = argparser.parse_args()
    SPECpowerTuner.main(args)
