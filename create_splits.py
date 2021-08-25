import argparse
import glob
import os
import random
from sklearn.model_selection import train_test_split
import shutil

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function
    all_files_path =glob.glob(os.path.join(data_dir,"*.tfrecord"))


    train_files_path, val_test_files_path=train_test_split(all_files_path,test_size=0.4)
    val_files_path,test_files_path=train_test_split(val_test_files_path,test_size=0.5)
    
    
    dest_train = os.path.join(data_dir, 'train')
    os.makedirs(dest_train, exist_ok=True)
    for train_path in train_files_path:
        shutil.move(train_path, dest_train)
    print("train dataset finished")


    dest_val = os.path.join(data_dir, 'val')
    os.makedirs(dest_train, exist_ok=True)
    for train_path in train_files_path:
        shutil.move(val_files_path, dest_val)
    print("validation dataset finished")


    dest_test = os.path.join(data_dir, 'test')
    os.makedirs(dest_train, exist_ok=True)
    for train_path in train_files_path:
        shutil.move(test_files_path, dest_test)
    print("test dataset finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
