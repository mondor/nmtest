#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 22:12:08 2021

@author: mondor
"""

import numpy as np
import h5py # type: ignore
import pandas as pd # type: ignore
import os
import glob
import tarfile
from pathlib import Path
from typing import Dict, Tuple, Any
from tqdm import tqdm # type: ignore
import shutil

class AttributeAggregator:
    def __init__(self, data_path: str):
        self._source_path = os.path.join(data_path, 'New_Data')        
        self._tmp_path = os.path.join(data_path, 'tmp')
        self._aggregated_path = os.path.join(data_path, 'aggregated')
        
        csv = os.path.join(self._source_path, 'attribute_manifest.csv')
        assert os.path.exists(csv)
        df = pd.read_csv(csv)
        # making sure the dataframe is unique by loc_id, date
        df['unique_key'] = df.apply(lambda x: f"{x['loc_id']}_{x['date']}", axis=1)
        df = df.drop_duplicates(subset=['unique_key'])
        self._df = df
        
        # pixel counts per attributes
        self._pixel_counts: Dict[str, int] = {}
        
        
    # extract all the jobs for a given location and date     
    def _extract_hdf_files(self, loc_id: str, date:str) -> Tuple[str, str]:        
        # will extract to path: self.extract_path/loc_id_date
        loc_id_date = f'{loc_id}_{date}'
        extracted_path = os.path.join(self._tmp_path, loc_id_date)
        Path(extracted_path).mkdir(parents=True, exist_ok=True)
        
        # find all the jobs under the location of a given date
        files = glob.glob(os.path.join(self._source_path, loc_id, f'*{loc_id_date}.tar.gz'))
        extracted_img_file = ''
        for file in files:
            with tarfile.open(file, 'r:gz') as archive:
                for member in archive.getmembers():
                    h5_or_jpg = member.name.endswith('.h5') or member.name.endswith('.jpg')
                    if not member.isdir() and h5_or_jpg:
                        # remove the "job_info" path
                        member.name = os.path.basename(member.name)
                                             
                        # extract the .h5 and .jpg under the extracted path
                        archive.extract(member, extracted_path)

                        if member.name.endswith('.jpg'):
                            extracted_img_file = member.name
                                                
        return extracted_path, extracted_img_file


    # iterate through the hdf metadata, and save the aggregated masks
    def _aggregate_hdf(self, aggregated_attributes: Dict[str, Any], h5_file: str) -> None:
        with h5py.File(h5_file) as hdf:
            for k in hdf.keys():
                dataset = np.array(hdf.get(k))
                # only interested the 1s, everything else set as zero
                dataset = np.where(dataset > 0, 1, 0)

                if k not in aggregated_attributes:
                    aggregated_attributes[k] = np.zeros(dataset.shape)

                # aggregate
                aggregated_attributes[k] += dataset


    # create a dictionary of the aggregated attributes from a extracted path    
    def _get_aggregated_attributes(self, extracted_path: str) -> Dict:        
        h5_files = glob.glob(os.path.join(extracted_path, '*.h5'))

        # a dict of aggregated attributes
        aggregated_attributes: Dict[str, Any] = {}
        for h5_file in h5_files:        
            self._aggregate_hdf(aggregated_attributes, h5_file)

        # normalise the final masks, record some metrics
        for k in aggregated_attributes:            
            # assuming the ML model is classification  rather than regression
            # replace >= 1 to just 1
            aggregated_attributes[k] = np.where(aggregated_attributes[k] > 0, 1, 0)

            if k not in self._pixel_counts:
                self._pixel_counts[k] = 0
                
            self._pixel_counts[k] += np.sum(aggregated_attributes[k])

                                                           
        return aggregated_attributes
        
    
    # create the aggregated h5 file and create an archive
    def _create_dist_folder(self, loc_id: str, date: str, aggregated_attributes: Dict[str, Any], src_image_file: str):
        loc_path = os.path.join(self._aggregated_path, loc_id)
        loc_id_date = f'{loc_id}_{date}'
        archive_path = os.path.join(loc_path, loc_id_date)
        Path(archive_path).mkdir(parents=True, exist_ok=True)
        
        h5_file = os.path.join(archive_path, f'{loc_id_date}.h5')        
        with h5py.File(h5_file, 'w') as hdf:
            for k in aggregated_attributes:
                hdf.create_dataset(k, data = aggregated_attributes[k])

        # copy the image to archive path
        shutil.copy(src_image_file, archive_path)
        
        output_filename = os.path.join(loc_path, f'{loc_id_date}.tar.gz')
        with tarfile.open(output_filename, "w:gz") as archive:
            archive.add(archive_path, arcname='.')        
            
        shutil.rmtree(archive_path)
    
    
    # extract the h5 files for the loc_id and date from data/New_Data folder 
    # compute the aggregated attributes and archive it to the data/aggregated folder 
    def aggregate_one(self, loc_id: str, date: str) -> None:
        extracted_path, extracted_img_file = self._extract_hdf_files(loc_id, date)
        aggregated_attributes = self._get_aggregated_attributes(extracted_path)
        src_image_file = os.path.join(extracted_path, extracted_img_file)
        self._create_dist_folder(loc_id, date, aggregated_attributes, src_image_file)
        shutil.rmtree(extracted_path)
                
    
    # iterate through the csv and create a new h5 file for each (location, date) pair
    def aggregate(self) -> None:
        total = self._df.shape[0]
        for index, row in tqdm(self._df.iterrows(), total=total):
            loc_id = str(row['loc_id'])
            date = str(row['date'])
            self.aggregate_one(loc_id, date)
            
            
    def print_metrics(self) -> None:
        roof = self._pixel_counts['2'] if '2' in self._pixel_counts else 0
        solar = self._pixel_counts['3'] if '3' in self._pixel_counts else 0        
        print(f'Solar panel coverage on roof is about {solar*100/(roof+1)}%')
        
        

        
        
        
        
    