"""
This file contains utility funtions to be used in the Cardiac
Segmentation Hackathon Challenge at the UTSW Hack-Med event on
Nov 9-10, 2018. The contributors to this file include:
Cooper Mellema
Paul Acosta

"""

import numpy as np
import math
from keras import backend as K
import pandas as pd
import os
import sys
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt



def fLogDiceloss(aPredictedVolumes, aActualVolumes):
    """
    This function returns the log of the dice score between two sets
    This function was originally implemented as a means of determining
    the accuracy of an automated segmentation deep learning algorithm
    in correctly classifying 3d Volumes. The Dice score is defined as:

            2|X Intersect Y|
    ______________________________
    |Cardinality X|+|Cardinality Y|    where X and Y are the labelled volumes

    :param aPredictedVolumes: a 3d array of the predicted volume labels
    :param aActualVolumes: a 3d array with the true volume labels
    :return: the log of the dice score
    """
    # May or may not need the tensor->array conversion, depending on how
    # output is formatted. No tests done on grayed out code
    # aPredictedVolumes = K.clip(aPredictedVolumes, K.epsilon(), 1-K.epsilon())

    flDiceLoss = ((2 * np.abs(np.intersect1d(aPredictedVolumes, aActualVolumes).size))
                   /(aPredictedVolumes.size + aActualVolumes.size)
                  )

    flLogDiceLoss = np.log(flDiceLoss)

    return flLogDiceLoss

class cPreprocess(object):
    """ This class contains all the methods for preprocessing the Cardiac Segmentation Data
    The general function types contained herein are as follows:
    -Functions to fetch Raw Data
    -Functions to fetch Training/Test Data
    -Functions to normalize Training and Test Data
    -Functions to reformat Training and Test Data
    """

    def __init__(self):
        self.RawDataLocation = '/project/hackathon/hackers09/shared/Data'
        #self.RawDataLocation = '/project/bioinformatics/DLLab/shared/Collab-Aashoo/WholeHeartSegmentation'
        self.ProcessedDataLocation = '/project/hackathon/hackers09/shared/Data'
        #self.ProcessedDataLocation = '/project/bioinformatics/DLLab/shared/Collab-Aashoo/WholeHeartSegmentation'
        self.TrainDataLocation = os.path.join(self.RawDataLocation, 'CoregTestV2All')
        self.TestDataLocation = os.path.join(self.ProcessedDataLocation, 'CoregTestV2All')
        self.Dimension = 3 # 3 dimensional image

        # Initialize the reference image that other images will be resampled based on
        self.ReferenceImageParams = {'origin': np.zeros(self.Dimension), # sets the origin of the image to [0,0,0]
                                     #'direction': np.identity(self.Dimension).flatten(), # sets direction to [1,1,1] (arbitrary)
                                     'size': [288]*self.Dimension, # downsample and/or upsample to 288x288x288
                                     'spacing': np.ones(self.Dimension) # set spacing to 1mm (arbitrary selection)
                                    }
        self.ReferenceImage = sitk.Image(self.ReferenceImageParams['size'], 2)
        # self.ReferenceImage.SetOrigin(self.ReferenceImageParams['origin'])
        self.ReferenceImage.SetSpacing(self.ReferenceImageParams['spacing'])
        # self.ReferenceImage.SetDirection(self.ReferenceImageParams['direction'])

    def fFetchRawDataFile(self, sPath):
        """Fetches a .nii file

        (currently redundant with sitk.ReadImage, but will be changed later)

        :param sPath: path to the .nii file
        :return: the .nii file in the sitk object form
        """
        NIIFile = sitk.ReadImage(sPath)
        return NIIFile

    def fResizeImage(self, NIIFile, is_label):
        """ Resizes a .nii file to parameters set in self
            Performs cropping and reslicing

        :param NIIFile: .nii file to be set at origin, spacing, direction
               is_label: boolean that indicates if an image is a label
        :return:
        """
        # cResampler.SetReferenceImage(self.ReferenceImage)
        # NIIResampled=cResampler.execute(NIIFile)
        # NIIResampled=sitk.Resample(NIIFile, self.ReferenceImage)

        NIIResampled = sitk.ResampleImageFilter()
        NIIResampled.SetOutputOrigin(NIIFile.GetOrigin())
        # NIIResampled.SetInterpolator(sitk.sitkNearestNeighbor)
        # print("Original")
        # print(NIIFile.GetSize())
        NIIResampled.SetSize(NIIFile.GetSize())
        NIIResampled.SetOutputSpacing(np.ones(3))
        NIIResampled.SetTransform(sitk.Transform())
        NIIResampled.SetDefaultPixelValue(NIIFile.GetPixelIDValue())

        if is_label:
            NIIResampled.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            NIIResampled.SetInterpolator(sitk.sitkBSpline)

        NIIResized = NIIResampled.Execute(NIIFile)
        # print("Resized:")
        # print(NIIResized.GetSize())
        return NIIResized


    def fPadImage(self, NIIFile):
        """
        Pads a .nii file to the parameters set in self

        :param NIIFile: .nii file to be set at origin, spacing, direction
        :return
        """
        NIIPadded = sitk.MirrorPad(NIIFile, self.ReferenceImage)
        return NIIPadded

    def fNIIFileToNormArray(self, NIIFile, flStd=1, flMean=0):
        """ Returns normalized array from the .nii file

        array is normalized by subtracting the mean and dividing by the std,
        if nothing is passed, just returns the array

        :param NIIFile: Take in a .nii file
        :return: returns an array from the .nii file
        """
        aDerivedImg = sitk.GetArrayFromImage(NIIFile)
        aDerivedImg = aDerivedImg - flMean
        aDerivedImg = np.divide(aDerivedImg, flStd)
        return aDerivedImg

    def fFetchTrainingData(self, sNIIFileName, **Args):
        """ Fetches Training data in raw form, normallizes it, and samples down to the same size
        Uses a pipeline of this form:
        Raw file -> fFetchRawDataFile -> imported raw nii file ->
        fResizeImage -> resize and Resampled nii file ->
        fNIIFileToNormArray -> normalized array

        :param sNIIFileName: the file name of the .nii file
        :return: array of normalized, resized data
        """
        NIIFile = self.fFetchRawDataFile((os.path.join(self.TrainDataLocation, sNIIFileName)))
        bLabel = 'label' in sNIIFileName
        NIIFileResized = self.fResizeImage(NIIFile, is_label=bLabel)
        aDerivedImg = self.fNIIFileToNormArray(NIIFileResized, **Args)
        return aDerivedImg

    def fFetchTestData(self):
        pdTestData=pd.DataFrame
        return pdTestData

    def fSaveITK(self, sNIIFileName, sOutDir):
        """
        Saves a new NII file after processing
        :param sNIIFileName: string file name of the .nii file being loaded
        :param sOutDir: string of directory to save file to
        :return:
        """

        NIIFile = self.fFetchRawDataFile((os.path.join(self.TrainDataLocation, sNIIFileName)))
        # print(NIIFile.GetSize())
        bLabel = 'label' in sNIIFileName
        NIIFileResized = self.fResizeImage(NIIFile, is_label = bLabel)
        print(NIIFileResized.GetSize())
        sitk.WriteImage(NIIFileResized, os.path.join(sOutDir,sNIIFileName))

def fCoregister(NIIFile1, NIIFile2, NIIFile2Label, bSegmentation=False):
    """
    adapted from: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/60_Registration_Introduction.html
    Takes 2 .nii files and linearly coregisters them
    TO NIIFILE1 AS REFERENCE
    :param NIIFile1: first .nii file
    :param NIIFile2: second .nii file
    :param bSegmentation: set to true if file is segmentation file, does Nearest Neighbor
        interpolation rather than linear interpolation
    :return: the transform to coregister the .nii files
    """
    # first, initialize an aligining transform
    cAlign = sitk.CenteredTransformInitializer(NIIFile1, NIIFile2,
                                               sitk.Euler3DTransform(),
                                               sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # initialize the registration method
    cRegistration = sitk.ImageRegistrationMethod()

    cRegistration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    cRegistration.SetMetricSamplingStrategy(cRegistration.RANDOM)
    cRegistration.SetMetricSamplingPercentage(0.01)
    cRegistration.SetInterpolator(sitk.sitkLinear)

    cRegistration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6,
                                   convergenceWindowSize=10)
    cRegistration.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    cRegistration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    cRegistration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    cRegistration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    cRegistration.SetInitialTransform(cAlign, inPlace=False)

    cTransform = cRegistration.Execute(sitk.Cast(NIIFile1, sitk.sitkFloat32),
                                                  sitk.Cast(NIIFile2, sitk.sitkFloat32))

    # if the image is a segmentation image, resample using knn, rather than linear interpolation
    NIIFile2CoregToNIIFile1 = sitk.Resample(NIIFile2, NIIFile1, cTransform,
                                                sitk.sitkLinear, 0.0, NIIFile1.GetPixelID())

    NIIFile2LabelCoregToNIIFile1 = sitk.Resample(NIIFile2Label, NIIFile1, cTransform,
                                                sitk.sitkNearestNeighbor, 0.0, NIIFile1.GetPixelID())

    return NIIFile2CoregToNIIFile1, NIIFile2LabelCoregToNIIFile1

class cSliceNDice(object):
    """ This class contains all the methods for augmenting and slicing the image
    The general function types contained herein are as follows:
    -Functions to take subsections of the data
    -functions to spatially jitter the data
    -functions to rotate the data
    """
    def __init__(self, NIIFile):
        self.NIIFile=NIIFile

    def fJitterv2(self):
        x_translation, y_translation, z_translation = (50, 0, 0)

        affine = sitk.AffineTransform(3)
        affine.SetTranslation((x_translation, y_translation, z_translation))
        resampled = sitk.Resample(self.NIIFile, affine)
        return resampled


    def fJitter(self,  aDirection = 'any'):
        """ This function translates a function using a gaussian
        process defined by flSigma, (std of the 3D gaussian)
        :param:flSigma: the std of a gaussian used to move the image
        :param:aDirection: a vector of the aDirection to translate the image
            default: 'any' means that the aDirection is chosen randomly
        :return: NIIFile, translated
        """

        flSigma = 50
        if aDirection == 'any':
            aDirection = np.array([np.random.normal(0, 1) for i in range(3)])

        # Normalize the direction vector
        flNorm = float(np.linalg.norm(aDirection, ord=1))
        if flNorm == 0:
            raise ValueError("'direction' vector has length 0"
                             "\ndivide by 0 error when normalizing direction vector")
        else:
            aDirection = np.array(aDirection)
            aDirection = (aDirection/flNorm)

        # Generate the step size based on the flSigma parameter
        flStep = abs(np.random.normal(0, flSigma))

        # Initialize the translation class
        cTranslator = sitk.TranslationTransform(3)
        cTranslator.SetOffset((aDirection[0]*flStep, aDirection[1]*flStep, aDirection[2]*flStep))

        NIIFile1 = self.NIIFile
        # Translate the NIIFile
        #NIIFileTranslated = sitk.Resample(self.NIIFile, File1, cTranslator)
        NIIFileTranslated = sitk.Resample(self.NIIFile, NIIFile1, cTranslator,
                      sitk.sitkNearestNeighbor, 0.0, NIIFile1.GetPixelID())
        return NIIFileTranslated

    def fPatch(self, aCenter = 'any', iSize = 64, aLimits=None):
        """ Cuts a small sub-patch out of the larger image for data augmentation
        :param center: the center of the subsampled region
        :param size: the size of the subsampled patch
        :return: a new NII file of the subsampled region
        """
        # Set the point where the patch will be (by default, doesn't go over the edges
        # of the image)
        flNIIWidth=self.NIIFile.GetSize()[0]

        if aCenter=='any':
            flXRange = np.random.uniform(iSize/2, flNIIWidth-iSize/2)
            flYRange = np.random.uniform(iSize/2, flNIIWidth-iSize/2)
            flZRange = np.random.uniform(iSize/2, flNIIWidth-iSize/2)
            aCenter=[flXRange, flYRange, flZRange]


        # Initialize the crop class and size of the cube
        # cCropper=sitk.CropImageFilter()
        if aLimits is None: # if the limits are given, use those, otherwise make a cube around the center
            aLimits=[iSize/2, iSize/2, iSize/2]
        lsLowerBound=[int(iC - iL) for iC, iL in zip(aCenter, aLimits)]
        lsUpperBound=[int(iC + iL) for iC, iL in zip(aCenter, aLimits)]
        # cCropper.SetLowerBoundaryCropSize(lsLowerBound)
        # cCropper.SetUpperBoundaryCropSize(lsUpperBound)

        # Create the patch
        NIIPatch = self.NIIFile[lsLowerBound[0]:lsUpperBound[0],
                                lsLowerBound[1]:lsUpperBound[1],
                                lsLowerBound[2]:lsUpperBound[2]
                               ]

        return NIIPatch

    def fWarpIDv2(self):
        x_scale, y_scale, z_scale = 3.0, 0.7, 0.5



        affine = sitk.AffineTransform(3)
        affine.Scale((x_scale))
        resampled = sitk.Resample(self.NIIFile, affine)

        return resampled


    def fWarp1D(self):
        """Warps a .nii file by flMaxScale percent up or down
        :param max_scale: the percent up or down an image dimension will be scaled
        :param bIsotropic: if true, the image will be scaled isotropically (all dimensions
            scaled the same amount), else each dimension will be separately scaled up or down
        :return: rescaled .nii file
        """
        # Generate the scaling factor

        flMaxScale = 0.05
        bIsotropic = False
        if bIsotropic:
            flScaleFactor = np.random.uniform(-flMaxScale, flMaxScale)
        else:
            aScaleFactor = [np.random.uniform(-flMaxScale, flMaxScale),
                            np.random.uniform(-flMaxScale, flMaxScale),
                            np.random.uniform(-flMaxScale, flMaxScale)
                            ]

        # Initialize the transformer
        cTransform=sitk.AffineTransform(3)
        if bIsotropic:
            aWarp=np.zeros((3,3,3))
            aWarp[0,0,0]=1+flScaleFactor
            aWarp[1,1,1]=1+flScaleFactor
            aWarp[2,2,2]=1+flScaleFactor
        else:
            aWarp=np.zeros((3,3,3))
            aWarp[0,0,0]=1+aScaleFactor[0]
            aWarp[1,1,1]=1+aScaleFactor[1]
            aWarp[2,2,2]=1+aScaleFactor[2]

        cTransform.SetMatrix(aWarp)

        # Do the resampling
        NIIWarped = sitk.Resample(self.NIIFile, cTransform)

        return NIIWarped


    def random_three_vector(self):

        """
        Generates a random 3D unit vector (direction) with a uniform spherical distribution
        Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
        :return:
        """
        phi = np.random.uniform(0,np.pi*2)
        costheta = np.random.uniform(-1,1)

        theta = np.arccos( costheta )
        x = np.sin( theta) * np.cos( phi )
        y = np.sin( theta) * np.sin( phi )
        z = np.cos( theta )
        return (x,y,z)

    def fRotatev2(self):
        degrees = 20


        affine = sitk.AffineTransform(3)
        radians = np.pi * degrees / 180.
        affine.Rotate(1, 2, 30.0, pre= False)
        resampled = sitk.Resample(self.NIIFile, affine)
        return resampled





    def fRotate(self):
        """Rotates a .nii file by flMax Theta and flMaxPhi
        :param max_scale: the percent up or down an image dimension will be scaled
        :param bIsotropic: if true, the image will be scaled isotropically (all dimensions
            scaled the same amount), else each dimension will be separately scaled up or down
        :return: rescaled .nii file
        """
        # Change Theta and Phi to radians
        #flMaxTheta = random.uniform(0,15)
        flMaxTheta = .001
        flMaxTheta=flMaxTheta*np.pi/180


        # Initialize the transformer
        cTransform=sitk.AffineTransform(3)

        tUnitVector = self.random_three_vector()

        units = [0, 1]
        l = tUnitVector[0]
        m = tUnitVector[1]
        n = tUnitVector[2]
        #aUnitVector = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        #l = aUnitVector[0]/math.sqrt(math.pow(aUnitVector[0], 2) + math.pow(aUnitVector[1], 2) + math.pow(aUnitVector[2], 2))
        #m = aUnitVector[1]/math.sqrt(math.pow(aUnitVector[0], 2) + math.pow(aUnitVector[1], 2) + math.pow(aUnitVector[2], 2))
        #n = aUnitVector[2]/math.sqrt(math.pow(aUnitVector[0], 2) + math.pow(aUnitVector[1], 2) + math.pow(aUnitVector[2], 2))

        aRotationMatrix = np.zeros(shape =(3,3))
        aRotationMatrix[0][0] = l*l*(1-math.cos(flMaxTheta) + math.cos(flMaxTheta))
        aRotationMatrix[0][1]  = m*l*(1-math.cos(flMaxTheta)) - n*math.sin(flMaxTheta)
        aRotationMatrix[0][2] = n*l*(1-math.cos(flMaxTheta)) + m*math.cos(flMaxTheta)
        aRotationMatrix[1][0] = l*m*(1-math.cos(flMaxTheta)) + n*math.sin(flMaxTheta)
        aRotationMatrix[1][1] = m*m*(1-math.cos(flMaxTheta)) + math.cos(flMaxTheta)
        aRotationMatrix[1][2] = n*m*(1-math.cos(flMaxTheta)) - l*math.sin(flMaxTheta)
        aRotationMatrix[2][0] = l*n*(1-math.cos(flMaxTheta)) - m*math.sin(flMaxTheta)
        aRotationMatrix[2][1] = m*n*(1-math.cos(flMaxTheta)) - l*math.sin(flMaxTheta)
        aRotationMatrix[2][2] = n*n*(1- math.cos(flMaxTheta)) + math.cos(flMaxTheta)

        cTransform.SetMatrix(aRotationMatrix.ravel())

        NIIRotated = sitk.Resample(self.NIIFile, cTransform)
      #  NIIWarpedLabel=sitk.Resample(NIIFileLabel, cTransform)


        return NIIRotated



    def fShear(self, flMaxShear, bIsotropic=True):
            return self

##############What follows is an example of how to use the preprocesser##################

Preprocesser=cPreprocess()

# # Convert all files to normalized arrays arrays after they have been preprocessed
# for Root, Dirs, Files in os.walk(Preprocesser.TrainDataLocation):
#     Files.sort()
#     aUnNormalizedAll = np.zeros(np.append(len(Files), Preprocesser.ReferenceImageParams['size'))
#     for iFile, File in enumerate(Files):
#         aUnNormalizedAll[iFile, :, :, :] = Preprocesser.fFetchTrainingData(File)
#
# std = np.std(aUnNormalizedAll)
# mean = np.mean(aUnNormalizedAll)
#
# for Root, Dirs, Files in os.walk(Preprocesser.TrainDataLocation):
#     Files.sort()
#     aNormalizedAll = np.zeros(np.append(len(Files), Preprocesser.ReferenceImageParams['size']))
#     for iFile, File in enumerate(Files):
#         aNormalizedAll[iFile, :, :, :] = Preprocesser.fFetchTrainingData(File, flStd=std, flMean=mean)

# Create a folder with resized data saved as new .nii files
# for Root, Dirs, Files in os.walk(Preprocesser.TrainDataLocation):
#     Files.sort()
#     for iFile, File in enumerate(Files):
#         outDir = '/project/bioinformatics/DLLab/shared/Collab-Aashoo/WholeHeartSegmentation/mr_train_resized'
#         Preprocesser.fSaveITK(File, outDir)

# do rough coregistration on all files
for Root, Dirs, Files in os.walk(Preprocesser.TrainDataLocation):
    Files.sort()
    for iFile, File in enumerate(Files):
        # load only the image files, load the labels separately
        if not 'label' in File:
                # coregister to the above file
                LabelFile=File[:-12]+'label.nii.gz'
                NIIFile = Preprocesser.fFetchRawDataFile(os.path.join(Preprocesser.TrainDataLocation, File))
#                NIIFileLabel = Preprocesser.fFetchRawDataFile(os.path.join(Preprocesser.TrainDataLocation, LabelFile))
                #NIIFile = cSlicer.fPatch(aCenter=[296, 260, 80], iSize=64, aLimits=[128, 128, 64])
                cSlicer = cSliceNDice(NIIFile)
                #NIIFile = cSlicer.fWarpIDv2()
                NIIFile = cSlicer.fRotate()
                #NIIFile = cSlicer.fJitterv2()
                #NIIFile = cSlicer.fJitter('any')
                #NIIFileLabel = cSlicer.fPatch(aCenter=[296, 260, 80], iSize=64, aLimits=[128, 128, 64])
                print('File number ' + str(iFile) + ' and its label are cropped')
                #NIICoreg, NIICoregLabel = fCoregister(NIIFile1, NIIFile, NIIFileLabel)
                print('File number ' + str(iFile) + ' and its label are coregistered')
                sitk.WriteImage(NIIFile, os.path.join('/project/hackathon/hackers09/shared/Data/Vishal', (File[:-7] + 'WarpTest.nii.gz')))
                #sitk.WriteImage(NIICoreg, os.path.join(
                 #   '/project/bioinformatics/DLLab/shared/Collab-Aashoo/'
                  #  'WholeHeartSegmentation/CoregTestAll', (File[:-7] + 'CoregTest.nii.gz')))
              #  sitk.WriteImage(NIIFileLabel, os.path.join('/project/hackathon/hackers09/shared/Data/Vishal',
                                                       #(File[:-12] + 'labelCoregTest2.nii.gz')))
               # sitk.WriteImage(NIICoregLabel, os.path.join(
                #    '/project/bioinformatics/DLLab/shared/Collab-Aashoo/'
                 #   'WholeHeartSegmentation/CoregTestAll', (File[:-12] + 'labelCoregTest.nii.gz')))

