import os

from dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderCross, DataLoaderInf
def get_training_data(rgb_dir, target_zs):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, target_zs, None)

def get_validation_data(rgb_dir, target_zs):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, target_zs, None)


def get_test_data(rgb_dir, testdir, target_zs):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, testdir, target_zs, None)

def get_crossval_data(rgb_dir, testdir, target_zs):
    assert os.path.exists(rgb_dir)
    return DataLoaderCross(rgb_dir, testdir, target_zs, None)

def get_inference_data(rgb_dir, testdir, target_zs):
    assert os.path.exists(rgb_dir)
    return DataLoaderInf(rgb_dir, testdir, target_zs, None)