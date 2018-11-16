"""
This file contains downsampling funtions to be used in the Cardiac
Segmentation Hackathon Challenge at the UTSW Hack-Med event on
Nov 9-10, 2018. The contributors to this file include:
Liangwei (Luke) Ge

"""

import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os
from CardiacSegmentation_UtilitiesV2 import cPreprocess

BASE = "/project/hackathon/hackers09/hack084/Data/"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("dir", "mr_train_resized", "subdir that holds images")
tf.app.flags.DEFINE_float("rate", 0.1, "scaling rate")

DATA_PATH = BASE + FLAGS.dir
OUT_DIR = DATA_PATH + '_' + str(FLAGS.rate)

print("data path: ", DATA_PATH)
print("output path: ", OUT_DIR)

os.mkdir(OUT_DIR)

for root, dirs, files in os.walk(DATA_PATH):
   for file in files:
      is_label = "label" in file
      print("Processing file", file, is_label)
      
      # Open original file:
      print("opening", file)
      cPreprocessor=cPreprocess()
      img = cPreprocessor.fFetchRawDataFile(DATA_PATH +'/' + file)      

      # reference image:
      print("scaling")
      size = np.array(img.GetSize())
      new_size = np.int_(size * FLAGS.rate)
      print(new_size, new_size[0], new_size[1], new_size[2])

      ref_img = sitk.ResampleImageFilter()
      ref_img.SetSize(new_size)
      ref_img.SetDefaultPixelValue(img.GetPixelIDValue())
      ref_img.SetOutputOrigin(img.GetOrigin())
      ref_img.SetOutputDirection(img.GetDirection())
      ref_img.SetOutputSpacing([sz*spc/nsz for nsz, sz, spc in zip(new_size, img.GetSize(), img.GetSpacing())])

      ref_img.SetTransform(sitk.Transform())

      if is_label:
         ref_img.SetInterpolator(sitk.sitkNearestNeighbor)
      else:
         ref_img.SetInterpolator(sitk.sitkBSpline)
      NIIResized = ref_img.Execute(img)

      # save
      print("saving")
      sitk.WriteImage(NIIResized, OUT_DIR + '/' + file)

