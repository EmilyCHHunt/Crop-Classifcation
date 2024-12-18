import os
import torch
from torch.utils.data import Dataset
import gzip
import zipfile
#from sh import gunzip
from glob import glob
import pickle
import sentinelhub # this import is necessary for pickle loading
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features
from tqdm import tqdm

CLASSES = ["Wheat", "Rye", "Barley", "Oats", "Corn", "Oil Seeds", "Root Crops", "Meadows", "Forage Crops"]
CROP_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

class S1Dataset(Dataset):
    def __init__(self, zippath, labelgeojson, transform=None):
        npzcache = zippath.replace(".zip", ".npz")

        self.data_transform = transform

        if not os.path.exists(npzcache):
            self.tsdata, self.fids, self.crop_ids = setup(zippath, labelgeojson)
            print(f"saving extracted time series with label data to {npzcache}")
            np.savez(npzcache, tsdata=self.tsdata, fids=self.fids, crop_ids=self.crop_ids)
        else:
            self.tsdata = np.load(npzcache)["tsdata"]
            self.fids = np.load(npzcache)["fids"]
            self.crop_ids = np.load(npzcache)["crop_ids"]

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, item):
        X = self.tsdata[item]
        y = self.crop_ids[item]
        fid = self.fids[item]

        if self.data_transform is not None:
            X = self.data_transform(X)

        return X, CROP_IDS.index(y), fid

def setup(zippath, labelgeojson):
    """
    This utility function unzipps a dataset from Sinergize and performs a field-wise aggregation.
    results are written to a .npz cache with same name as zippath
    """
    datadir = os.path.dirname(zippath)
    rootpath = zippath.replace(".zip", "")
    if not (os.path.exists(rootpath) and os.path.isdir(rootpath)):
        print(f"unzipping {zippath} to {datadir}")
        with zipfile.ZipFile(zippath, 'r') as zip_ref:
            zip_ref.extractall(datadir)
    else:
        print(f"found folder in {rootpath}, no need to unzip")

    # find all .gz-ipped files and unzip
    for gz in glob(os.path.join(rootpath,"*","*.gz")) + glob(os.path.join(rootpath,"*.gz")):
        print(f"unzipping {gz}")
        gunzip(gz)

    with open(os.path.join(rootpath, "bbox.pkl"), 'rb') as f:
        bbox = pickle.load(f)
        crs = str(bbox.crs)
        minx, miny, maxx, maxy = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y

    labels = gpd.read_file(labelgeojson)
    # project to same coordinate reference system (crs) as the imagery
    labels = labels.to_crs(crs)

    vv = np.load(os.path.join(rootpath, "data", "VV.npy"))
    vh = np.load(os.path.join(rootpath, "data", "VH.npy"))
    bands = np.stack([vv[:,:,:,0],vh[:,:,:,0]], axis=3)
    #clp = np.load(os.path.join(rootpath, "data", "CLP.npy"))
    #bands = np.concatenate([bands, clp], axis=-1) # concat cloud probability
    _, width, height, _ = bands.shape

    transform = rio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    fid_mask = features.rasterize(zip(labels.geometry, labels.fid), all_touched=True,
                              transform=transform, out_shape=(width, height))
    assert len(np.unique(fid_mask)) > 0, f"vectorized fid mask contains no fields. " \
                                         f"Does the label geojson {labelgeojson} cover the region defined by {zippath}?"

    crop_mask = features.rasterize(zip(labels.geometry, labels.crop_id), all_touched=True,
                              transform=transform, out_shape=(width, height))
    assert len(np.unique(crop_mask)) > 0, f"vectorized fid mask contains no fields. " \
                                          f"Does the label geojson {labelgeojson} cover the region defined by {zippath}?"

    fids = []
    crop_ids = []
    tsdata = []
    for fid, crop_id in tqdm(zip(labels.fid.unique(), labels.crop_id.values), total=len(labels), desc="extracting time series"):
        field_mask = fid_mask == fid
        if field_mask.sum() > 0:
            data = bands.transpose(0, 3, 1, 2)[:, :, field_mask].mean(-1)
            tsdata.append(data)
            crop_ids.append(crop_id)
            fids.append(fid)
        else:
            print(f"field {fid} contained no pixels. Is it too small with {labels.loc[labels.fid==fid].geometry.area}m2 ? skipping...")

    tsdata = np.stack(tsdata)
    return tsdata, fids, crop_ids

if __name__ == '__main__':
    zippath = r"C:\Users\emily\Documents\Uni stuff\Proj\sentinel-1\s1-asc-utm-33N-17E-243N-2019.zip"
    labelgeojson = r"C:\Users\emily\Documents\Uni stuff\Proj\br-17E-243N-crop-labels-test-2019.geojson"

    ds = S1Dataset(zippath, labelgeojson)
    len(ds)
    X,y,fid = ds[0]