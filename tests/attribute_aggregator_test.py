import pytest
import os
import h5py
import numpy as np
from pathlib import Path

from nmtest.attribute_aggregator import AttributeAggregator

def test_aggregate_hdf():
    h5_file_path = './data/tmp/test/'
    Path(h5_file_path).mkdir(parents=True, exist_ok=True)

    h5_file = f'{h5_file_path}/test.h5'
    with h5py.File(h5_file, 'w') as hdf:
        np_a = np.zeros((10,10))
        np_b = np.zeros((10,10))
        np_b[3:6, 3:6] = 2
        hdf.create_dataset('a', data = np_a)
        hdf.create_dataset('b', data = np_b)

    attribute_aggregator: AttributeAggregator = AttributeAggregator('./data')
    result = {}
    total = 0
    attribute_aggregator._aggregate_hdf(result, h5_file)
    for k in result:
        total += np.sum(result[k])

    assert total == 9


