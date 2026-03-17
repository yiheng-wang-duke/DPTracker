from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.lasot_extension_subset_path_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.network_path = ''    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = ''
    settings.result_plot_path = ''
    settings.results_path = ''    # Where to store tracking results
    settings.save_dir = ''
    settings.segmentation_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.uav_path = ''

    settings.nat2021_path = ''
    settings.nat2021l_path = ''
    settings.uavdark135_path = ''
    settings.darktrack2021_path = ''
    settings.nat2024_path = ''
    settings.nutl_path = ''
    settings.uav123_10fps_path = ''
    settings.uav123_path = ''
    settings.uavdt_path = ''
    settings.dtb70_path = ''
    settings.uavtrack112_path = ''
    settings.visdrone2018_path = ''

    return settings