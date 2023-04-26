from utils.utils import Config

def model_config():
    config = Config({
        # MLIC and MLIC+
        "N": 192,
        "M": 320,
        "slice_num": 10,
        "context_window": 5,
        "slice_ch": [8, 8, 8, 8, 16, 16, 32, 32, 96, 96],
        "quant": "ste",
    })

    return config
