import os
import numpy as np
import config as Config

def load_bandwidth():
    datas = os.listdir(Config.data_dir)
    time_traces = []
    throughput_traces = []
    data_names = []
    if Config.bw_env_version == 0:
        for data in datas:
            file_path = data_dir + data
            time_trace = []
            throughput_trace = []
            time = 0.0
            with open(file_path, 'r') as f:
                for line in f:
                    parse = line.strip('\n')
                    time_trace.append(time)
                    throughput_trace.append(float(parse))
                    time += 1.0
            time_traces.append(time_trace)
            throughput_traces.append(throughput_trace)
            data_names.append(data)
    elif Config.bw_env_version == 1:
        for data in datas:
            file_path = data_dir + data
            time_trace = []
            throughput_trace = []
            with open(file_path, 'r') as f:
                for line in f:
                    parse = line.strip('\n').split()
                    time_trace.append(float(parse[0]))
                    throughput_trace.append(float(parse[1]))
            time_traces.append(time_trace)
            throughput_traces.append(throughput_trace)
            data_names.append(data)
    return time_traces, throughput_traces, data_names

def load_single_trace(data_dir):
    file_path = data_dir
    time_trace = []
    throughput_trace = []
    time = 0.0
    with open(file_path, 'r') as f:
        if Config.bw_env_version == 0:
            for line in f:
                parse = line.strip('\n')
                time_trace.append(time)
                throughput_trace.append(float(parse))
                time += 1.0
        elif Config.bw_env_version == 1:
            for line in f:
                parse = line.strip('\n').split()
                time_trace.append(float(parse[0]))
                throughput_trace.append(float(parse[1]))
    return time_trace, throughput_trace
