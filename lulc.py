from osgeo import gdal
import pandas as pd, xarray as xr, os
import scipy.misc as sm
import rasterio, numpy as np, glob
from rasterio.merge import merge 
from rasterio import plot 
from PIL import Image
import matplotlib.pyplot as plt

class Landsat():
    def __init__(self, files):
        self.infiles = files
    
    def sort_files(self):
        files = self.infiles
        x = int(files[0].split('.')[1].split('/')[-1].split('_')[-1][1:])
        files = sorted(files, key = lambda x: x.split('.')[1].split('/')[-1].split('_')[-1])
        lst = []
        
        for filename in files:
            try:
                x = int(filename.split('.')[1].split('/')[-1].split('_')[-1][1:])
                lst.append(filename)
            except:
                #print filename
                pass
            
        self.filenames = sorted(lst, key = lambda x: int(x.split('.')[1].split('/')[-1].split('_')[-1][1:]))
        return self
    
    def layer_stacking(self, ofile='stack.tif'):
        file_list = self.filenames
        self.stackfile = ofile
        # Read metadata of first file
        with rasterio.open(file_list[0]) as src0:
            meta = src0.meta

        # Update meta to reflect the number of layers
        meta.update(count = len(file_list))

        # Read each layer and write it to stack
        with rasterio.open(ofile, 'w', **meta) as dst:
            for id, layer in enumerate(file_list, start=1):
                with rasterio.open(layer) as src1:
                    dst.write_band(id, src1.read(1))
        return self

    def composite(self, band = [5, 4, 3], ofile='fcc.tif', cmin=2, cmax=98):
        self.composite_file = ofile
        dataset = rasterio.open(self.stackfile)
        meta = dataset.meta
        meta.update(count = len(band))
        #print meta

        red = dataset.read([band[0]])
        red = (255*self.norm(red)).astype('int')
        green = dataset.read([band[1]])
        green = (255*self.norm(green)).astype('int')
        blue = dataset.read([band[2]])
        blue = (255*self.norm(blue)).astype('int')

        rgb = np.vstack((red, green, blue)).astype('uint16')
        #rgb = 255*rgb/np.max(rgb)
#         sm.toimage(rgb, cmin=np.percentile(rgb, cmin), \
#                    cmax=np.percentile(rgb, cmax)).save(self.composite_file)
        
        with rasterio.open(self.composite_file, "w", **meta) as dest:
                dest.write(rgb)
        return self
    
    @staticmethod                
    def norm(band):
        band = band.astype('float')
        band = (band-band.min())/(band.max()-band.min())
        return band
    
    @staticmethod
    def imshow(filename):
        temp = 'temp.tif'
        src = rasterio.open(filename)
        data = np.squeeze(src.read())
        sm.toimage(data).save(temp)
        
        Image.MAX_IMAGE_PIXELS = data.size + 100
        im = Image.open(temp)
        im.show()
        
class Analysis():
    def __init__(self, files):
        self.files = files
        
    def mosaic_images(self, ofile = 'mosaic.tif', bbox=True):
        tif_files = self.files
        self.mosaic_file = ofile
        files_to_mosaic = []
        
        for tif_file in tif_files:
            try:
                src = rasterio.open(tif_file)
                if bbox == True:
                    print(src.bounds)
                files_to_mosaic.append(src)
            except:
                print('%s does not exist.' % tif_file)
                
        mosaic, out_trans = merge(files_to_mosaic)  
        out_meta = src.meta.copy()
        out_meta.update({"driver":"GTiff", "height":mosaic.shape[1],"width":mosaic.shape[2], "transform":out_trans})
        
        with rasterio.open(self.mosaic_file, "w", **out_meta) as dest:
            dest.write(mosaic)
        return self

