# app/utils/cli.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="ictrain.yaml",
        help="Config file name under app/cfgs/"
    )
    return parser.parse_args()


# end of app/utils/cli.py