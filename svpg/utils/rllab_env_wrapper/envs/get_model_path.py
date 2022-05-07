import os.path as osp


def model_path(file_name):
    return osp.abspath(
        osp.join(osp.dirname(osp.dirname(__file__)), f"models/{file_name}")
    )
