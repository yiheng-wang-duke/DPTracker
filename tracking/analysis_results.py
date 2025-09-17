import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

import sys
sys.path.insert(0, "/home/user/.conda/envs/wyh/lib/python3.8/site-packages")
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'darktrack2021'
# trackers.extend(trackerlist(name='avtrack', parameter_name='vit_tiny_patch16_224', dataset_name=dataset_name,
#                             run_ids=None, display_name='avtrack-vit'))
trackers.extend(trackerlist(name='avtrack_pt', parameter_name='vit_tiny_patch16_224_e28', dataset_name=dataset_name,
                            run_ids=None, display_name='avtrack-pt-vit'))
# trackers.extend(trackerlist(name='avtrack', parameter_name='deit_tiny_patch16_224', dataset_name=dataset_name,
#                             run_ids=None, display_name='avtrack-deit'))
# trackers.extend(trackerlist(name='avtrack_pt', parameter_name='vit_tiny_patch16_224', dataset_name=dataset_name,
#                             run_ids=None, display_name='avtrack-pt-vit'))
# trackers.extend(trackerlist(name='avtrack_pt', parameter_name='deit_tiny_patch16_224', dataset_name=dataset_name,
#                             run_ids=None, display_name='avtrack-pt-deit'))

# trackers.extend(trackerlist(name='ostrack_pt', parameter_name='vitb_256_ce_ep30', dataset_name=dataset_name,
#                            run_ids=None, display_name='ostrack-256-pt'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='ostrack-256'))
# trackers.extend(trackerlist(name='ostrack_pt', parameter_name='vitb_256_mae_32x4_ep50', dataset_name=dataset_name,
#                             run_ids=None, display_name='ostrack-256-pt'))
# trackers.extend(trackerlist(name='ostrack_pt_v6', parameter_name='vitb_256_mae_32x4_ep50', dataset_name=dataset_name,
#                             run_ids=None, display_name='ostrack-256-pt-v6'))
# trackers.extend(trackerlist(name='ostrack_pt_v2', parameter_name='vitb_256_mae_ce_32x4_ep30', dataset_name=dataset_name,
#                             run_ids=None, display_name='ostrack-256-pt-v2'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='ostrack-384'))
# trackers.extend(trackerlist(name='avtrack', parameter_name='deit_tiny_patch16_224', dataset_name=dataset_name,
#                             run_ids=None, display_name='avtrack-deit'))

dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

