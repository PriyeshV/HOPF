import sys
import itertools
import subprocess
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.tabulate_results import write_results
from src.utils.utils import *
import time

machine = 'F+S_cora_'
get_results_only = False

switch_gpus = False  # For multiple GPUs
default_gpu_id = 0
n_gpus = 1
n_parallel_threads = 1

args = dict()
args['hyper_params'] = ['aggKernel', 'node_features', 'neighbor_features', 'shared_weights', 'max_outer', 'gpu']
args['aggKernel'] = [sys.argv[1]]
args['node_features'] = [sys.argv[2]]
args['neighbor_features'] = [sys.argv[3]]
args['shared_weights'] = [sys.argv[4]]
args['max_outer'] = [sys.argv[5]]
args['gpu'] = [sys.argv[6]]

# Set Hyper-parameters
now = datetime.now()
name = machine + args['aggKernel'][0] + '_' + args['node_features'][0] + '_' + args['neighbor_features'][0] + '_' + args['shared_weights'][0] + '_' + args['max_outer'][0] + '_'
if not get_results_only:
    timestamp = name + str(now.month)+'|'+str(now.day)+'|'+str(now.hour)+':'+str(now.minute)+':'+str(now.second)  # +':'+str(now.microsecond)
else:
    name = machine + 'simple_x_-_0_1_1'
    timestamp = name + '9|8|5:56:26' #  '05|12|03:41'  # Month | Day | hours | minutes (24 hour clock)

args_path = '../Experiments/' + timestamp + '/args/'
stdout_dump_path = '../Experiments/' + timestamp + '/stdout_dumps/'

args['timestamp'] = [timestamp]
if not get_results_only:

    # The names should be the same as argument names in parser.py
    args['hyper_params'] = args['hyper_params'] + ['dataset', 'batch_size', 'dims', 'neighbors', 'max_depth', 'lr', 'l2',
                                                   'drop_in', 'drop_lr', 'wce', 'percents', 'folds',
                                                   'skip_connections', 'propModel', 'timestamp']

    args['dataset'] = ['cora']
    args['batch_size'] = [128]  # 16
    args['dims'] = ['16,16,16,16,16']
    args['neighbors'] = ['all,all,all,all,all']
    args['max_depth'] = [1, 2, 3, 4, 5]  # 1
    args['lr'] = [1e-2]
    args['l2'] = [1e-3]
    args['drop_in'] = [0.5]
    args['drop_lr'] = [True]
    args['wce'] = [True]
    args['percents'] = [10]
    args['folds'] =['1,2,3,4,5']
    args['skip_connections'] = [True]
    args['propModel'] = ['propagation_gated']

    pos = args['hyper_params'].index('dataset')
    args['hyper_params'][0], args['hyper_params'][pos] = args['hyper_params'][pos], args['hyper_params'][0]


    def diff(t_a, t_b):
        t_diff = relativedelta(t_a, t_b)
        return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

    # Create Args Directory to save arguments
    if not path.exists(args_path):
        create_directory_tree(str.split(args_path, sep='/')[:-1])
    name = args['aggKernel'][0] + '_' + args['node_features'][0] + '_' + args['neighbor_features'][0] \
           + '_' + args['shared_weights'][0] + '_' + args['max_outer'][0]

    np.save(path.join(args_path, name), args)

    # Create Log Directory for stdout Dumps
    if not path.exists(stdout_dump_path):
        create_directory_tree(str.split(stdout_dump_path, sep='/')[:-1])

    param_values = []
    this_module = sys.modules[__name__]
    for hp_name in args['hyper_params']:
        param_values.append(args[hp_name])

    combinations = list(itertools.product(*param_values))
    n_combinations = len(combinations)
    print('Total no of experiments: ', n_combinations)

    pids = [None] * n_combinations
    f = [None] * n_combinations
    last_process = False
    for i, setting in enumerate(combinations):
        # Create command
        command = "python __main__.py "
        folder_suffix = ''
        for name, value in zip(args['hyper_params'], setting):
            command += "--" + name + " " + str(value) + " "
            if name != 'dataset':
                folder_suffix += "_" + str(value)
        command += "--" + "folder_suffix " + folder_suffix + '__' + str(i + 1) + '/' + str(n_combinations)
        print(i + 1, '/', n_combinations, command)

        # if switch_gpus and (i % n_gpus) == 0:
        #     gpu_id = str(int(i % n_gpus))
        #     env = dict(environ, **{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu_id})
        # else:
        #     gpu_id = str(default_gpu_id)
        #     env = dict(environ, **{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu_id})

        name = path.join(stdout_dump_path, folder_suffix)
        with open(name, 'w') as f[i]:
            pids[i] = subprocess.Popen(command.split(), stdout=f[i])
        time.sleep(3)
        if i == n_combinations - 1:
            last_process = True
        if ((i + 1) % n_parallel_threads == 0 and i >= n_parallel_threads - 1) or last_process:
            if last_process and not ((i + 1) % n_parallel_threads) == 0:
                n_parallel_threads = (i + 1) % n_parallel_threads
            start = datetime.now()
            print('########## Waiting #############')
            for t in range(n_parallel_threads - 1, -1, -1):
                pids[i - t].wait()
            end = datetime.now()
            print('########## Waiting Over######### Took', diff(end, start), 'for', n_parallel_threads, 'threads')

    write_results(args)
else:

    name = args_path + args['aggKernel'][0] + '_' + args['node_features'][0] + '_' + args['neighbor_features'][0] \
           + '_' + str(args['shared_weights'][0]) + '_' + str(args['max_outer'][0])
    try:
        args = np.load(name+'.npy').item()
        write_results(args)
        print("Done tabulation")
    except:
        print('model not found', name)