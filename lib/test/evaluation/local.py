from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.darktrack2021_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/DarkTrack2021'
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/got10k_lmdb'
    settings.got10k_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/itb'
    settings.lasot_extension_subset_path_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/lasot_lmdb'
    settings.lasot_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/lasot'
    settings.nat2021_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/NAT2021'
    settings.network_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/nfs'
    settings.otb_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/otb'
    settings.prj_dir = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26'
    settings.result_plot_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/output/test/result_plots'
    settings.results_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/output'
    settings.segmentation_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/output/test/segmentation_results'
    settings.tc128_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/trackingnet'
    settings.uav_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/uav'
    settings.uavdark135_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/UAVDark135'
    settings.vot18_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/vot2018'
    settings.vot22_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/vot2022'
    settings.vot_path = '/mnt/sdc/V4R/WYH/Pytracking/github_ICRA26/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

