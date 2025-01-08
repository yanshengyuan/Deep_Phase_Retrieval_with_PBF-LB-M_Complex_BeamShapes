# -*- coding: utf-8 -*-

"""
One liner

Followed by detailed description separated by a single line. 
"""

# Import required modules
from LightPipes import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import math
import random
import configparser
import numpy as np
from UserFunctions.UserFunctions import SmoothStep
from UserFunctions.UserFunctions import SmoothCircAperture
import time
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

# Define paths and filenames for input
inputPathStr="./Input_Data/"
outputPathStr = "./Output_Data/"
configFileStr="Config_AI_Data_Generator-train.dat"

# Open data generator config file
config = configparser.ConfigParser()
checkFile = config.read(inputPathStr+configFileStr)

# Define initial field
wavelength = config["field_initialization"].getfloat("wavelength")
gridSize = config["field_initialization"].getfloat("gridSize")
gridPixelnumber = config["field_initialization"].getint("gridPixelnumber")
beamDiameter = config["gaussian_beam"].getfloat("beamDiameter")
beamWaist = beamDiameter/2

lightField = Begin(gridSize,wavelength,gridPixelnumber)
lightField = GaussBeam(lightField, beamWaist, n = 0, m = 0, x_shift = 0, y_shift=0, tx=0, ty=0, doughnut=False, LG=True)

# Prepare and apply CGH phase mask to field
cghFilename = config["cgh_data"]["cghFilename"]
cghBackgroundValue = config["cgh_data"].getint("cghBackgroundValue") 
cghGreyValues = config["cgh_data"].getint("cghGreyValues")
cghSize = config["cgh_data"].getfloat("cghSize")
cghPixelNumber = config["cgh_data"].getint("cghPixelNumber")

cghImageData = mpimg.imread(inputPathStr + cghFilename) 
cghPhaseData = 2*np.pi*(np.asarray(cghImageData[:,100:700])-cghBackgroundValue/cghGreyValues)

cghField=Begin(cghSize,wavelength,cghPixelNumber)
cghField=MultPhase(cghField,cghPhaseData)
cghField=Interpol(cghField, gridSize, gridPixelnumber, x_shift=0.0, y_shift=0.0, angle=0.0, magnif=1.0)

lightField=MultPhase(lightField,Phase(cghField))

# Prepare calculation of Zernike coefficients      
zernikeMaxOrder = config["zernike_coefficients"].getint("zernikeMaxOrder")
zernikeAmplitude = config["zernike_coefficients"].getfloat("zernikeAmplitude")
zernikeRadius = config["zernike_coefficients"].getfloat("zernikeRadius")
nollMin = config["zernike_coefficients"].getint("nollMin")


nollMax = np.sum(range(1,zernikeMaxOrder + 1))  # Maximum Noll index
nollRange=range(nollMin,nollMax+1)
zernikeCoeff=np.zeros(nollMax)

# Prepare main control loop
randomSeed = config["run_control"].getint("randomSeed") 
runMax = config["run_control"].getint("runMax")
random.seed(randomSeed)

#Prepare field focusing 
beamMagnification = config["field_focussing"].getfloat("beamMagnification")
focalLength = config["field_focussing"].getfloat("focalLength") / beamMagnification
focalReduction = config["field_focussing"].getfloat("focalReduction")
                   
f1=focalLength*focalReduction
f2=f1*focalLength/(f1-focalLength)
frac=focalLength/f1
newSize=frac*gridSize
newExtent=[-newSize/2/mm,newSize/2/mm,-newSize/2/mm,newSize/2/mm]

#Prepare field aperture
apertureRadius = config["field_aperture"].getfloat("apertureRadius")
apertureSmoothWidth = config["field_aperture"].getfloat("apertureSmoothWidth")

# Prepare propagation of field to caustic planes
focWaist = wavelength/np.pi*focalLength/beamWaist  # Focal Gaussian beam waist
zR = np.pi*focWaist**2/wavelength # Rayleigh range focused Gaussian beam

causticPlanes = []
causticPlanes.append(("01-pre6", config["caustic_planes"].getfloat("preprefocPlane") ))
causticPlanes.append(("02-pre5", config["caustic_planes"].getfloat("prefocPlane") ))
causticPlanes.append(("03-pre4", config["caustic_planes"].getfloat("pstprefocPlane") ))
causticPlanes.append(("04-foc", config["caustic_planes"].getfloat("focPlane")) )
causticPlanes.append(("05-pst4", config["caustic_planes"].getfloat("prepostfocPlane") ))
causticPlanes.append(("06-pst5", config["caustic_planes"].getfloat("postfocPlane") ))
causticPlanes.append(("07-pst6", config["caustic_planes"].getfloat("pstpostfocPlane") ))  

# Prepare intensity output
outputSize = config["data_output"].getfloat("outputSize")
outputPixelnumber = config["data_output"].getint("outputPixelnumber")

# Initialize and start main control loop
for runCount in range(0,runMax):
    # Define paths and filenames for output
    runName = f"run{runCount:04}"
    outFileName = runName + "_intensity_"
    
    # Generate and apply Zernike phase distortion      
    distField = lightField
    zernikeField = lightField
    
    for countNoll in nollRange:        
        (nz,mz) = noll_to_zern(countNoll)
        zernikeCoeff[countNoll-1] = random.uniform(-1,1)*zernikeAmplitude/(nz+1)
        zernikeField = Zernike(zernikeField,nz,mz,zernikeRadius,zernikeCoeff[countNoll-1],units='rad')
    distField = zernikeField

    # Prefocus field and apply aperture
    distField = Lens(f1,0,0,distField);
    distField = SmoothCircAperture(distField, apertureRadius, apertureSmoothWidth)
    
    # Propagate field to caustic planes
    causticFields=[]
    outputFields=[]
    
    for causticStr,causticPos in causticPlanes:
        cField=LensFresnel(distField,f2,focalLength+causticPos*zR)
        cField=Convert(cField)
        causticFields.append(cField)
        outputField=Interpol(cField,outputSize,outputPixelnumber)        
        outputFields.append(outputField)
        
        # Prepare and save caustic intensities as .png files
        outFileStr= outputPathStr + outFileName + causticStr + ".png"
        mpimg.imsave(outFileStr,-Intensity(outputField),cmap='Greys')
        
    # Prepare and save Zernike coefficients intensities as .npz files
    zernikeFileStr = outputPathStr + runName + "_zernikeCoeff" + ".npy"
    np.save(zernikeFileStr, zernikeCoeff)
    print(runCount)
# End of main control loop
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))