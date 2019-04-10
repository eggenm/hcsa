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
# Define functions
# =============================================================================
def genClassifierMultisite(img_dict, bands, n_site = 2000):
    training = ee.FeatureCollection([])
    for site, img in img_dict.items():
        bounds = img.geometry().bounds()
        site_training = img.sample(numPixels = n_site, seed = 0, scale = 30, 
                                   region = bounds)
        training = training.merge(site_training)
    classifier = ee.Classifier.randomForest(30).train(training, 'remapped', inputProperties = bands)
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
        self.class_img = ee.Image('users/rheilmayr/indonesia/' + class_site + '_toClass')
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

band_dict =  {'B1': 'ublue',
              'B2': 'blue',
              'B3': 'green',
              'B4': 'red',
              'B5': 'nir',
              'B6': 'swir1',
              'B7': 'swir2',
              'B10': 'tir1',
              'B11': 'tir2',
              'sr_aerosol': 'sr_aerosol'} 

def prep_ls8(img):
    """
    Used to map the initialized class onto an imagecollection. Will mask out clouds,
    add an ndvi band, and rename all bands.
    """
    # Mask out flagged clouds
    img = maskCloudsLandsat8(img)
    
    # Rename bands
    old_names = list(band_dict.keys())
    new_names = list(band_dict.values())
    img = img.select(old_names, new_names)

    # Add ndvi
    img = img.addBands(img.normalizedDifference(['nir', 'red']))
    
    # Rename ndvi
    newer_names = new_names.copy()
    newest_names = new_names.copy()
    newer_names.append('nd')
    newest_names.append('ndvi')
    img = img.select(newer_names, newest_names)

    return img


# =============================================================================
# Load study data
# =============================================================================
key_csv = 'D:/cloud/dropbox/rspo/hcsProject/data/strata_key.csv'
key_df = pd.read_csv(key_csv)
from_vals = list(key_df['project_code'].astype(float).values)
to_vals = list(key_df['code_simpl'].astype(float).values)

sites = ['app_jambi', 'app_oki', 'app_kaltim', 'app_kalbar', 'app_muba', 'app_riau',
         'crgl_stal', 'gar_pgm', 'nbpol_ob', 'wlmr_calaro']
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
ic = ic.filterDate('2010-01-01', '2016-06-01')
ic = ic.filterMetadata(name = 'WRS_ROW', operator = 'less_than', value = 120)
ic = ic.filterBounds(all_study_area)
ic_masked = ic.map(prep_ls8)
clean_img = ic_masked.qualityMosaic('ndvi')

# =============================================================================
# Create site-level images for classification with reclassed strata and landsat data
# =============================================================================
for site, strata_img in hcs_db.rasters.items():
    strata_img = ee.Image(strata_img)
    strata_img = strata_img.remap(from_vals, to_vals, 4)
    geometry = strata_img.geometry()
    coords = geometry.coordinates()
    json_coords = coords.getInfo()
    strata_img = strata_img.float()
    class_img = clean_img.addBands(strata_img)
    export = ee.batch.Export.image.toAsset(class_img, scale = 30, region = json_coords,
                                           assetId = 'users/rheilmayr/hcs_out/' + site + '_toClass',
                                           maxPixels = 1e13)
    export.start()

### Note: Current version of code using ned Landsat 8 surface reflectance product 
### is providng way worse results than old version that relied on deprecated
### Landsat 8 products. Need to update code to create a useable product. 
    
# =============================================================================
# Jackknife classification
# =============================================================================
bands = ['swir1', 'nir', 'red', 'pan', 'swir2', 'blue',
         'green', 'tir1', 'tir2', 'ndvi', 'remapped']
img_dict = {site: ee.Image('users/rheilmayr/indonesia/' + site + '_toClass').select(bands) \
            for site in sites} ## Note - using old landsat files because new exports are a mess
for site in sites:
    img_dict[site] = img_dict[site].set({'site': site})
    
out_point_dict = {}
for test_site in sites:
    train_sites = [site for site in sites if site != test_site]
    train_imgs = {site: img_dict[site] for site in train_sites}
    train_classifier = genClassifierMultisite(train_imgs, bands, 50000)
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
out_point_dict['app_jambi'].getInfo()

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
out_point_dict = gee_tools.dict_to_eedict(out_point_dict)
acc_fc = ee.FeatureCollection(out_point_dict.values())
export = ee.batch.Export.table.toDrive(acc_fc, folder = 'hcs', description = 'jackknife_acc',
                                       fileNamePrefix = 'jackknife1e6_rf', fileFormat = 'csv')
export.start()

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
    export = ee.batch.Export.image.toAsset(classed_img, description = site + '_class', scale = 30, region = site_json_coords,
                                       assetId = 'users/rheilmayr/indonesia/' + site + '_class_30',
                                       maxPixels = 1e13)
    export.start()