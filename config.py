# Configuration for all files
class Config(object):
    model_version = 0           #v0: two outputs,   v1: one (6*7) output
    initial_epsilon = 1.0 
    epsilon_start = 1.0
    epsilon_final = 0.0001
    epsilon_decay = 2000.0          # less, focus faster
    logs_path = './logs_' + str(model_version) + '/'
    reply_buffer_size = 3000
    total_episode = 50000
    discount_factor = 0.99
    save_logs_frequency = 1000
    lr = 1e-3
    momentum = 0.9
    # batch_size = 300
    observe_episode = 5
    sampling_batch_size = 300
    update_target_frequency = 50
    show_loss_frequency = 1000
    maximum_model = 5

    massive_result_files = './all_results/'
    trace_idx = 2

class Env_Config(object):
    # For environment, ms
    bw_env_version = 0              # O for LTE (NYC), 1 for 3G (Norway)
    if bw_env_version == 0:
        data_dir = '../bw_traces/'
    elif bw_env_version == 1:
        data_dir = '../new_traces/train_sim_traces/'
    s_info = 10
    s_len = 15
    a_num = 2
    a_dims = [6, 7] # 6 bitrates and 7 playing speed
    video_terminal_length = 300

    ms_in_s = 1000.0
    kb_in_mb = 1000.0   # in ms
    seg_duration = 1000.0
    chunk_duration = 200.0
    chunk_in_seg = seg_duration/chunk_duration
    chunk_seg_ratio = chunk_duration/seg_duration
    server_init_lat_low = 1
    server_init_lat_high = 10
    start_up_ssh = 2000.0
    freezing_tol = 3000.0 
    
    default_action_1 = 0
    default_action_2 = 3
    skip_segs = 3.0
    repeat_segs = 3.0

    # Reward metrics parameters
    action_reward = 1.0 * chunk_seg_ratio   
    rebuf_penalty = 6.0                         
    smooth_penalty = 1.0
    long_delay_penalty = 4.0 * chunk_seg_ratio
    const = 6.0
    x_ratio = 1.0 
    speed_smooth_penalty = 2.0
    unnormal_playing_penalty = 2.0              
    skip_seg_penalty = 3.0              
    repeat_seg_penalty = 3.0      
    skip_latency = skip_segs * seg_duration + chunk_duration 
