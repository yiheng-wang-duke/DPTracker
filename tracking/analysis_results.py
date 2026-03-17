import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

import sys
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'darktrack2021'
trackers.extend(trackerlist(name='dptrack_b', parameter_name='vitb_256_ce_ep30', dataset_name=dataset_name,
                            run_ids=None, display_name='dptrack-b'))
# trackers.extend(trackerlist(name='dptrack_t', parameter_name='vit_tiny_patch16_224', dataset_name=dataset_name,
#                             run_ids=None, display_name='dptrack-t'))

dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

