# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 04:49:15 2019

@author: rheil
"""

# =============================================================================
# Imports
# =============================================================================
import ee
ee.Initialize()
import numpy as np
import hcs_database as hcs_db
import pandas as pd

# =============================================================================
# Define paths
# =============================================================================
in_path = 'users/rheilmayr/indonesia/'
out_path = 'users/eggenm/hcs_out/'

# =============================================================================
# Define date range
# =============================================================================
year='ALL2' 
date_start = ee.Date.fromYMD(2014, 1, 1)
date_end = ee.Date.fromYMD(2018, 12, 31)
#year=2017
#date_start = ee.Date.fromYMD(year, 1, 1)
#date_end = ee.Date.fromYMD(year, 12, 31)
# =============================================================================
# Define functions
# =============================================================================
def genClassifierMultisite(img_dict, bands, n_site = 2000):
    training = ee.FeatureCollection([])
    for site, img in img_dict.items():
        bounds = img.geometry().bounds()
        img = img.cast(ee.Dictionary({'remapped': "int8"}))
        site_training = img.stratifiedSample(numPoints = n_site, seed = 0, 
                                             scale = 30, region = bounds, classBand = 'remapped')
        training = training.merge(site_training)
#    classifier = ee.Classifier.cart(prune = True).train(training, 'remapped', inputProperties = bands)
    classifier = ee.Classifier.randomForest(100).train(training, 'remapped', inputProperties = bands)
    return classifier    

def genClassifier(class_img, n = 10000):
    study_area = class_img.geometry().bounds()
    training = class_img.sample(numPixels = n, seed = 0, scale = 30, 
                                region = study_area)
#     classifier = ee.Classifier.cart(prune = True).train(training, 'remapped')
#     classifier = ee.Classifier.svm(kernelType = 'RBF', gamma = 0.5, 
#                                    cost = 10).train(training, 'remapped')
    classifier = ee.Classifier.randomForest(200).train(training, 'remapped')
    return classifier

def validate(classifier, val_img, n = 5000):
    study_area = val_img.geometry().bounds()
    validation = val_img.sample(numPixels = n, seed = 1, scale = 30, 
                                region = study_area)
    validated = validation.classify(classifier)
    testAccuracy = validated.errorMatrix('remapped', 'classification')
    return testAccuracy

class validate_map:
    def __init__(self, class_site, class_n = 10000, val_n = 5000):
        self.class_site = class_site
        self.class_img = ee.Image(out_path + class_site + '_toClass')
        self.classifier = genClassifier(self.class_img, class_n)
        self.val_n = val_n
    def __call__(self, val_img):
        val_site = val_img.get('site')
        study_area = val_img.geometry().bounds()
        validation = val_img.sample(numPixels = self.val_n, seed = 1, scale = 30, 
                                    region = study_area)
        validated = validation.classify(self.classifier)
        testAccuracy = validated.errorMatrix('remapped', 'classification')
#         val_img = val_img.set(self.class_site, testAccuracy)
        out_point = ee.Feature(ee.Geometry.Point(0, 0))
        out_dict = {str(n) + str(m): testAccuracy.array().get([n,m]) for n in range(4) for m in range(4)}
        out_dict.update({'test_site': val_site,
                         'train_site': self.class_site,
                         'kappa': testAccuracy.kappa(),
                         'acc': testAccuracy.accuracy(),
                         'p_acc': testAccuracy.producersAccuracy(),
                         'c_acc': testAccuracy.consumersAccuracy()})
        out_point = out_point.set(out_dict)
        return out_point
    
def maskCloudsLandsat8(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get the pixel QA band.
    qa = image.select('pixel_qa')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask)

l8_band_dict =  {'B1': 'ublue',
              'B2': 'blue',
              'B3': 'green',
              'B4': 'red',
              'B5': 'nir',
              'B6': 'swir1',
              'B7': 'swir2',
              'B10': 'tir1',
              'B11': 'tir2',
              'sr_aerosol': 'sr_aerosol'
#             ,'nd': 'ndvi_l8'
              } 

s2_band_dict = {'B1': 'S2_ublue',
              'B2': 'S2_blue',
              'B3': 'S2_green',
              'B4': 'S2_red',
              'B5': 'rededge1',
              'B6': 'rededge2',
              'B7': 'rededge3',
              'B8': 'S2_nir',
              'B8A': 'S2_nir2',
              'B9': 'S2_vape',
              'B10': 'S2_swir1',
              'B11': 'S2_swir2',
              'B12': 'S2_swir3',
              'nd': 'ndvi_s2'
} 

s1_band_dict = {'VH': 'VH',
              'VV': 'VV'} 

def maskS2clouds(image):
  qa = image.select('QA60')
  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloudBitMask = (1 << 10)
  cirrusBitMask = (1 << 11)
  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000)

def prep_ls8(img):
    """
    Used to map the initialized class onto an imagecollection. Will mask out clouds,
    add an ndvi band, and rename all bands.
    """
    # Mask out flagged clouds
    img = maskCloudsLandsat8(img)
    
    # Rename bands
    old_names = list(l8_band_dict.keys())
    new_names = list(l8_band_dict.values())
    img = img.select(old_names, new_names)
        # Add ndvi
    img = img.addBands(img.normalizedDifference(['nir', 'red']))
    
    # Rename ndvi
    newer_names = new_names.copy()
    newest_names = new_names.copy()
    newer_names.append('nd')
    newest_names.append('ndvi_l8')
    img = img.select(newer_names, newest_names)

    return img

def prep_sar(image_collection):
  composite = ee.Image.cat([
    image_collection.select('VH').mean(),
    image_collection.select('VV').mean()
  #  sentinel1.select('VH').reduce(ee.Reducer.stdDev()).rename('VH_vari'), There are string artifacts with this operation
  # sentinel1.select('VV').reduce(ee.Reducer.stdDev()).rename('VV_vari')
  ]).focal_median();
  composite = composite.set('year', year)
  return composite

def prep_s2(img):
    # Mask out flagged clouds
    img = maskS2clouds(img)
    # Rename bands
    img = img.addBands(img.normalizedDifference(['B8', 'B4']))
    old_names = list(s2_band_dict.keys())
    new_names = list(s2_band_dict.values())
    img = img.select(old_names, new_names)
     # Add ndvi

    return img

def add_ndvi(img, keys, values, platform):
    print(platform)
    # Add ndvi
    #band_names=
    newer_names = keys.copy()
    newest_names = values.copy()
    newest_names.replace('nir', 'nir'+platform)
    newest_names.replace('red', 'red'+platform)
    img = img.addBands(img.normalizedDifference(['nir', 'red']))
    # Rename ndvi    
    newer_names.append('nd')
    newest_names.append('ndvi'+platform)
    newest_names.replace('nir', 'nir'+platform)
    newest_names.replace('red', 'red'+platform)
    
#    newer_names = list(['nd']) + list(band_names)
#    newer_names.extend(band_names)
#    newest_names = list(['ndvi']) + list(band_names)
#    newest_names.extend(band_names)
    print(newest_names)
    img = img.select(newer_names, newest_names)
    return img
# =============================================================================
# Load study data
# =============================================================================
key_csv = 'C:/Users/Eggen/Dropbox/HCSproject/data/strata_key.csv'
key_df = pd.read_csv(key_csv)
from_vals = list(key_df['project_code'].astype(float).values)
to_vals = list(key_df['code_simpl'].astype(float).values)

#sites = ['app_jambi', 'app_oki', 'app_kaltim', 'app_kalbar',
#         'app_muba', 
#         'app_riau',
#         'crgl_stal', 'gar_pgm', 'nbpol_ob', 'wlmr_calaro']
sites = ['gar_pgm']
  
feature_dict = {}
for site in sites:
    strata_img = ee.Image(hcs_db.rasters[site])
    geometry = strata_img.geometry()
    feature = ee.Feature(geometry)
    feature_dict[site] = feature
fc = ee.FeatureCollection(list(feature_dict.values()))
all_study_area = fc.geometry().bounds()
all_json_coords = all_study_area.getInfo()['coordinates']

# =============================================================================
# Prep landsat data
# =============================================================================
ic = ee.ImageCollection('LANDSAT/LC08/C01/T2_SR')
ic = ic.filterDate(date_start, date_end)
ic = ic.filterMetadata(name = 'WRS_ROW', operator = 'less_than', value = 120)
ic = ic.filterBounds(all_study_area)
ic_masked = ic.map(prep_ls8)
clean_l8_img = ee.Image(ic_masked.qualityMosaic('ndvi_l8'))
#print(clean_l8_img.bandNames().getInfo())

# =============================================================================
# Prep SAR data
# =============================================================================
#radarCollectionByYear = ee.ImageCollection(ee.List.sequence(2014,2018,1).map(prep_sar))
sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
sentinel1 = sentinel1.filterDate(date_start, date_end)
sentinel1 = sentinel1.filter(ee.Filter.eq('instrumentMode', 'IW'))
sentinel1 = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
sentinel1 = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
sentinel1 = sentinel1.filterBounds(all_study_area)
sentinel1 = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

radar_composite = ee.Image(prep_sar(sentinel1))
# =============================================================================
# Prep Sentinel-2 data
# =============================================================================
sentinel2 = ee.ImageCollection('COPERNICUS/S2')
sentinel2 =sentinel2.filterDate(date_start, date_end)
# Pre-filter to get less cloudy granules.
sentinel2 =sentinel2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))
sentinel2 =sentinel2.filterBounds(all_study_area)
sentinel2_masked=sentinel2.map(prep_s2)
#clean_s2_img=ee.Image(sentinel2_masked.median())
clean_s2_img=sentinel2_masked.qualityMosaic('ndvi_s2')

# =============================================================================
# Create site-level images for classification with reclassed strata and landsat data
# =============================================================================
bands = list(l8_band_dict.values()) + list(['ndvi_l8'])
bands.extend(list(['remapped']))
bands.extend(list(s1_band_dict.values()))
bands.extend(list(s2_band_dict.values()))

print(bands)
img_dict=dict.fromkeys(sites,0)
for site in sites:
    strata_img = ee.Image(hcs_db.rasters[site])
    strata_img = strata_img.remap(from_vals, to_vals, 4)
    geometry = strata_img.geometry()
    coords = geometry.coordinates()
    json_coords = coords.getInfo()
    strata_img = strata_img.int()
    class_img = clean_l8_img.addBands(strata_img).addBands(radar_composite).addBands(clean_s2_img)
    img_dict[site]=ee.Image(class_img).select(bands)
#    export = ee.batch.Export.image.toAsset(class_img, scale = 30, region = json_coords,
#                                          assetId = out_path + site + str(year) +'_input_wRadar_andS2',
#                                           maxPixels = 1e13)
#    export.start()

### Note: Current version of code using ned Landsat 8 surface reflectance product 
### is providng way worse results than old version that relied on deprecated
### Landsat 8 products. Need to update code to create a useable product. 
    
# =============================================================================
# Jackknife classification
# =============================================================================

img_dict = {site: ee.Image(out_path + site + str(year) + '_input_wRadar_andS2',).select(bands) \
           for site in sites} ## Note - using old landsat files because new exports are a mess
for site in sites:
    img_dict[site] = img_dict[site].set({'site': site})
out_point_dict = {}
for test_site in sites:
    train_sites = [site for site in sites if site != test_site]
    train_imgs = {site: img_dict[site] for site in train_sites}
    train_classifier = genClassifierMultisite(train_imgs, bands, 10000)
    testAccuracy = validate(train_classifier, img_dict[test_site], 5000)
    out_point = ee.Feature(ee.Geometry.Point(0, 0))
    out_dict = {str(n) + str(m): testAccuracy.array().get([n,m]) for n in range(4) for m in range(4)}
    out_dict.update({'test_site': test_site,
                     'kappa': testAccuracy.kappa(),
                     'acc': testAccuracy.accuracy(),
                     'p_acc': testAccuracy.producersAccuracy(),
                     'c_acc': testAccuracy.consumersAccuracy()})
    out_point = out_point.set(out_dict)
    out_point_dict[test_site] = out_point
#out_acc = out_point_dict['app_jambi'].getInfo()

# =============================================================================
# Presentation prep - temporary
# =============================================================================
#test_site = 'app_riau'
#train_sites = ['app_jambi']
#train_imgs = {site: img_dict[site].cast(ee.Dictionary({'remapped': "int8"})) for site in train_sites}
#train_classifier = genClassifierMultisite(train_imgs, bands, 5000)
#testAccuracy = validate(train_classifier, img_dict[test_site], 20000)
#out_point = ee.Feature(ee.Geometry.Point(0, 0))
#out_dict = {str(n) + str(m): testAccuracy.array().get([n,m]) for n in range(4) for m in range(4)}
#out_dict.update({'test_site': test_site,
#                 'kappa': testAccuracy.kappa(),
#                 'acc': testAccuracy.accuracy(),
#                 'p_acc': testAccuracy.producersAccuracy(),
#                 'c_acc': testAccuracy.consumersAccuracy()})
#out_point = out_point.set(out_dict)
#out_acc_same = out_point.getInfo()
#print(out_acc_same)
#
#train_sites = ['app_kalbar']
#train_imgs = {site: img_dict[site].cast(ee.Dictionary({'remapped': "int8"})) for site in train_sites}
#train_classifier = genClassifierMultisite(train_imgs, bands, 5000)
#testAccuracy = validate(train_classifier, img_dict[test_site], 20000)
#out_point = ee.Feature(ee.Geometry.Point(0, 0))
#out_dict = {str(n) + str(m): testAccuracy.array().get([n,m]) for n in range(4) for m in range(4)}
#out_dict.update({'test_site': test_site,
#                 'kappa': testAccuracy.kappa(),
#                 'acc': testAccuracy.accuracy(),
#                 'p_acc': testAccuracy.producersAccuracy(),
#                 'c_acc': testAccuracy.consumersAccuracy()})
#out_point = out_point.set(out_dict)
#out_acc_one = out_point.getInfo()
#print(out_acc_one)
#
#train_sites = ['app_jambi', 'app_oki', 'crgl_stal']
#train_imgs = {site: img_dict[site].cast(ee.Dictionary({'remapped': "int8"})) for site in train_sites}
#train_classifier = genClassifierMultisite(train_imgs, bands, 5000)
#testAccuracy = validate(train_classifier, img_dict[test_site], 20000)
#out_point = ee.Feature(ee.Geometry.Point(0, 0))
#out_dict = {str(n) + str(m): testAccuracy.array().get([n,m]) for n in range(4) for m in range(4)}
#out_dict.update({'test_site': test_site,
#                 'kappa': testAccuracy.kappa(),
#                 'acc': testAccuracy.accuracy(),
#                 'p_acc': testAccuracy.producersAccuracy(),
#                 'c_acc': testAccuracy.consumersAccuracy()})
#out_point = out_point.set(out_dict)
#out_acc_sumat = out_point.getInfo()
#print(out_acc_sumat)
#
#train_sites = ['app_kaltim', 'app_kalbar', 'app_jambi', 'app_oki', 'crgl_stal'
#              #  , 'app_muba'
#               ]
#train_imgs = {site: img_dict[site].cast(ee.Dictionary({'remapped': "int8"})) for site in train_sites}
#train_classifier = genClassifierMultisite(train_imgs, bands, 5000)
#testAccuracy = validate(train_classifier, img_dict[test_site], 20000)
#out_point = ee.Feature(ee.Geometry.Point(0, 0))
#out_dict = {str(n) + str(m): testAccuracy.array().get([n,m]) for n in range(4) for m in range(4)}
#out_dict.update({'test_site': test_site,
#                 'kappa': testAccuracy.kappa(),
#                 'acc': testAccuracy.accuracy(),
#                 'p_acc': testAccuracy.producersAccuracy(),
#                 'c_acc': testAccuracy.consumersAccuracy()})
#out_point = out_point.set(out_dict)
#out_acc_indo = out_point.getInfo()
#print(out_acc_indo)
#
#train_sites = ['app_kaltim', 'app_kalbar', 'app_jambi', 'app_oki', 'crgl_stal'
#             #  , 'app_muba'
#               ]
#train_imgs = {site: img_dict[site].cast(ee.Dictionary({'remapped': "int8"})) for site in train_sites}
#train_classifier = genClassifierMultisite(train_imgs, bands, 5000)
#testAccuracy = validate(train_classifier, img_dict[test_site], 20000)
#out_point = ee.Feature(ee.Geometry.Point(0, 0))
#out_dict = {str(n) + str(m): testAccuracy.array().get([n,m]) for n in range(4) for m in range(4)}
#out_dict.update({'test_site': test_site,
#                 'kappa': testAccuracy.kappa(),
#                 'acc': testAccuracy.accuracy(),
#                 'p_acc': testAccuracy.producersAccuracy(),
#                 'c_acc': testAccuracy.consumersAccuracy()})
#out_point = out_point.set(out_dict)
#out_acc_world = out_point.getInfo()
#print(out_acc_world)

import matplotlib.pyplot as plt
import pandas as pd
df = pd.Series({'Same plot': 94,
                   'South Sumatra plot': 64,
                   'All Sumatra plots': 70,
                   'All Indonesia and PNG plots': 72,
                   'Nigeria plot': 20}) 
## =============================================================================
## Compare to results if you use same site for classifciation and testing
## =============================================================================
#for test_site in sites:
#    train_sites = [site for site in sites if site == test_site]
#    train_imgs = {site: img_dict[site] for site in train_sites}
#    train_classifier = genClassifierMultisite(train_imgs, bands, 1000)
#    testAccuracy = validate(train_classifier, img_dict[test_site], 5000)
#    out_point = ee.Feature(ee.Geometry.Point(0, 0))
#    out_dict = {str(n) + str(m): testAccuracy.array().get([n,m]) for n in range(4) for m in range(4)}
#    out_dict.update({'test_site': test_site,
#                     'kappa': testAccuracy.kappa(),
#                     'acc': testAccuracy.accuracy(),
#                     'p_acc': testAccuracy.producersAccuracy(),
#                     'c_acc': testAccuracy.consumersAccuracy()})
#    out_point = out_point.set(out_dict)
#    out_point_dict[test_site] = out_point
#out_point_dict['app_jambi'].getInfo()

# =============================================================================
# Export accuracy metrics
# =============================================================================
#out_point_dict = gee_tools.dict_to_eedict(out_point_dict)
#acc_fc = ee.FeatureCollection(out_point_dict.values())
#export = ee.batch.Export.table.toDrive(acc_fc, folder = 'hcs', description = 'jackknife_acc',
#                                       fileNamePrefix = 'jackknife1e6_rf', fileFormat = 'csv')
#export.start()

# =============================================================================
# Export classified images
# =============================================================================
classifier = genClassifierMultisite(img_dict, bands, 3000)
for site in sites:
    to_class = img_dict[site]
    mask = to_class.select('remapped').mask()
    to_class = to_class.updateMask(mask)
    classed_img = to_class.classify(classifier)
    site_json_coords = feature_dict[site].geometry().bounds().getInfo()['coordinates']
    print(site_json_coords)
    export = ee.batch.Export.image.toAsset(classed_img, description = site + '_ClassW_radar_s2', scale = 30, region = site_json_coords,
                                       assetId = out_path + site + str(year) + '_ClassW_radar_s2',
                                       maxPixels = 1e13)
    export.start()
print(out_path + site + str(year) + '_ClassW_radar_s2')
