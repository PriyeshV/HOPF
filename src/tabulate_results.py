import xlwt
import itertools
import numpy as np
from os import path, mkdir, listdir


def write_results(args):
    expt_path = '../Experiments/' + args['timestamp'][0]
    save_path = expt_path + '/resuts_xls/'
    if not path.exists(save_path):
        mkdir(save_path)
    percents = ['10', 'avg']

    book = xlwt.Workbook(encoding='utf-8')

    sheets = {}
    metrics = ['O_EPOCH', 'I_EPOCH', 'TR_F1', 'VAL_LOSS', 'VAL_F1', 'k-MICRO-F1', 'k-MACRO-F1', 'micro-f1', 'macro-f1', 'MC_ACC', 'ML_ACC', 'BAE']
    n_metrics = len(metrics)

    if args is not None:
        cols = args['hyper_params'][1:] + [''] + metrics

        param_values = []
        for hp_name in args['hyper_params'][1:]:
            param_values.append(args[hp_name])
        combinations = list(itertools.product(*param_values))
        n_combinations = len(combinations)

        for i in range(len(percents)):
            data_name = percents[i]
            sheets[data_name] = book.add_sheet(data_name)
            row0 = sheets[data_name].row(0)
            col_id = -1
            for header in cols:
                col_id += 1
                row0.write(col_id, header)

        row_id = 0
        for i,setting in enumerate(combinations):
            folder_suffix = ''

            for name, value in zip(args['hyper_params'][1:], setting):
                folder_suffix += "_" + str(value)

            prefix = path.join(path.join(expt_path, args['dataset'][0]), args['aggKernel'][0])
            prefix = path.join(prefix, folder_suffix) + '__' + str(i + 1) + '/' + str(n_combinations)
            if not path.exists(prefix):
                continue

            try:
                results = np.loadtxt(path.join(prefix, 'metrics.txt'), skiprows=1)
            except:
                continue

            mean_results = np.zeros((1, n_metrics))
            for i in range(len(percents)):
                for name, value in zip(args['hyper_params'][1:], setting):
                    data_name = percents[i]
                    row = sheets[data_name].row(row_id + 1)
                    row.write(cols.index(name), value)

                if i < len(percents)-1:
                    for pos in range(n_metrics):
                        if len(percents) == 2:
                            val = results[pos]
                        else:
                            val = results[i][pos]
                        row.write(cols.index(metrics[pos]), val)
                        mean_results[0, pos] += val
                else:
                    mean_results = mean_results/(len(percents)-1)
                    for pos in range(n_metrics):
                        row.write(cols.index(metrics[pos]), mean_results[0][pos])
            row_id +=1

    # Save it with a time-stamp
    book.save(path.join(save_path, args['aggKernel'][0] + '_' + args['node_features'][0] + '_' + args['neighbor_features'][0]
                        + '_' + str(args['shared_weights'][0]) + '_' + str(args['max_outer'][0]) + '.xls'))
