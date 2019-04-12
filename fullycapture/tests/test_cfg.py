import totaldensify.utils.config as cfg_util
if __name__=='__main__':
    cfg_file = '../../config/TotalCapture.yaml'
    cfg = cfg_util.load_cfg_from_file(cfg_file)
    print(cfg['FIT_STAGE_0'])