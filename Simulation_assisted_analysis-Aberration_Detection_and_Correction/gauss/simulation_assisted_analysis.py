#Compute, collect, compare, and analyze ARE, MSE, MAE, and WaveFrontError.
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
import os
import pandas as pd
from PIL import Image

filename="mlp"

def RMSE(wave1, wave2):
    mse=0.0
    for x in range(len(wave1)):
        for y in range(len(wave1[x])):
            if(wave1[x][y]>=wave2[x][y]):
                mse += (wave1[x][y]-wave2[x][y])**2
            if(wave1[x][y]<wave2[x][y]):
                mse += (wave2[x][y]-wave1[x][y])**2
    mse=mse/((len(wave1))**2)
    rmse=math.sqrt(mse)
    return rmse

# Define paths and filenames for input
inputPathStr="./Input_Data/"
outputPathStr_corr = "./CorrErr/" + filename + "/"
outputPathStr_recons = "./ReconsErr/" + filename + "/"
configFileStr="Config_AI_Data_Generator.dat"

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
beamshape = os.path.basename(current_script_dir)
inputpath='../'+beamshape+'_testset/'
inputlist=sorted(os.listdir(inputpath))

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
gaussian_mask = Intensity(lightField)
mask_avg = np.mean(gaussian_mask)
gaussian_mask=gaussian_mask/mask_avg

# Prepare and apply CGH phase mask to field
cghFilename = config["cgh_data"]["cghFilename"]
cghBackgroundValue = config["cgh_data"].getint("cghBackgroundValue") 
cghGreyValues = config["cgh_data"].getint("cghGreyValues")
cghSize = config["cgh_data"].getfloat("cghSize")
cghPixelNumber = config["cgh_data"].getint("cghPixelNumber")

cghImageData = mpimg.imread(inputPathStr + cghFilename) 
cghPhaseData = 2*np.pi*(np.asarray(cghImageData[:,:,0])-cghBackgroundValue/cghGreyValues)

cghField=Begin(cghSize,wavelength,cghPixelNumber)
cghField=MultPhase(cghField,cghPhaseData)
cghField=Interpol(cghField, gridSize, gridPixelnumber, x_shift=0.0, y_shift=0.0, angle=0.0, magnif=1.0)

#lightField=MultPhase(lightField,Phase(cghField))
masked_CGHapplied_unw=Phase(lightField, unwrap=True)*gaussian_mask
masked_CGHapplied_w=Phase(lightField, unwrap=False)*gaussian_mask
#mpimg.imsave(phaseimg_path+"2-Masked-CGH_PhaseMask_applied.png",masked_CGHapplied,cmap='Greys')

# Prepare calculation of Zernike coefficients      
zernikeMaxOrder = config["zernike_coefficients"].getint("zernikeMaxOrder")
zernikeAmplitude = config["zernike_coefficients"].getfloat("zernikeAmplitude")
zernikeRadius = config["zernike_coefficients"].getfloat("zernikeRadius")
nollMin = config["zernike_coefficients"].getint("nollMin")


nollMax = np.sum(range(1,zernikeMaxOrder + 1))  # Maximum Noll index
nollRange=range(nollMin,nollMax+1)
zernikeCoeff=np.zeros(nollMax)

# Prepare main control loop
runMax = config["run_control"].getint("runMax")

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
causticPlanes.append(("-01-pre-", config["caustic_planes"].getfloat("prefocPlane") ))
causticPlanes.append(("-02-foc-", config["caustic_planes"].getfloat("focPlane")) )
causticPlanes.append(("-03-pst-", config["caustic_planes"].getfloat("postfocPlane") )) 

# Prepare intensity output
outputSize = config["data_output"].getfloat("outputSize")
outputPixelnumber = config["data_output"].getint("outputPixelnumber")

distField = lightField
distField = Lens(f1,0,0,distField)
distField = SmoothCircAperture(distField, apertureRadius, apertureSmoothWidth)
#aberration-free intensity
GTintensity=[]
for causticStr,causticPos in causticPlanes:
    cField=LensFresnel(distField,f2,focalLength+causticPos*zR)
    cField=Convert(cField)
    outputField=Interpol(cField,outputSize,outputPixelnumber)
    intensity_gt=Intensity(outputField)
    outFileStr= outputPathStr_corr + "gt/" + causticStr + ".png"
    mpimg.imsave(outFileStr,-intensity_gt,cmap='Greys')
    gt_img = Image.open(outFileStr)
    if gt_img.mode != 'L':
        gt_img = gt_img.convert('L')
    intensity_gt=np.array(gt_img)
    GTintensity.append(intensity_gt)

pred=np.load(filename+".npy")
gt=np.load("gt.npy")
Metrics=[]
# Initialize and start main control loop
for runCount in range(0,runMax):
    print(runCount)
    runName = f"run{runCount:04}"
    outFileName = runName + "_intensity_"
    
    # MAE(z)
    zernikeField = lightField
    zernikeField_pred = lightField
    
    zernikecoefficients = pred[runCount]
    zGT = gt[runCount]

    mae = 0.0
    for x in range(len(zGT)):
        mae += abs(zernikecoefficients[x]-zGT[x])
    mae = mae/len(zGT)
    
    #Apply aberrations
    for countNoll in nollRange: 
        if countNoll>3:
            (nz,mz) = noll_to_zern(countNoll)
            zernikeCoeff[countNoll-1] = zGT[countNoll-4]
            zernikeField = Zernike(zernikeField,nz,mz,zernikeRadius,zernikeCoeff[countNoll-1],units='rad')
        else:
            (nz,mz) = noll_to_zern(countNoll)
            zernikeCoeff[countNoll-1] = 0
            zernikeField = Zernike(zernikeField,nz,mz,zernikeRadius,zernikeCoeff[countNoll-1],units='rad')

    #Correct aberrations
    for countNoll in nollRange: 
        if countNoll>3:
            (nz,mz) = noll_to_zern(countNoll)
            zernikeCoeff[countNoll-1] = -zernikecoefficients[countNoll-4]
            zernikeField = Zernike(zernikeField,nz,mz,zernikeRadius,zernikeCoeff[countNoll-1],units='rad')
        else:
            (nz,mz) = noll_to_zern(countNoll)
            zernikeCoeff[countNoll-1] = 0
            zernikeField = Zernike(zernikeField,nz,mz,zernikeRadius,zernikeCoeff[countNoll-1],units='rad')
    distField = zernikeField
    
    #Wavefront Error
    masked_corrected_wavefront_unw=Phase(distField, unwrap=True)*gaussian_mask
    masked_corrected_wavefront_w=Phase(distField, unwrap=False)*gaussian_mask
    wavefront_error_unw=RMSE(masked_CGHapplied_unw, masked_corrected_wavefront_unw)
    wavefront_error_w=RMSE(masked_CGHapplied_w, masked_corrected_wavefront_w)      
    
    #Correction Error
    distField_pred = zernikeField
    # Prefocus field and apply aperture
    distField_pred = Lens(f1,0,0,distField_pred)
    distField_pred = SmoothCircAperture(distField_pred, apertureRadius, apertureSmoothWidth)

    PRDintensity=[]
    for causticStr,causticPos in causticPlanes:
        cField_pred=LensFresnel(distField_pred,f2,focalLength+causticPos*zR)
        cField_pred=Convert(cField_pred)
        outputField_pred=Interpol(cField_pred,outputSize,outputPixelnumber)
        intensity_pred=Intensity(outputField_pred)
        outFileStr= outputPathStr_corr + outFileName + causticStr + ".png"
        mpimg.imsave(outFileStr,-intensity_pred,cmap='Greys')
        pred_img = Image.open(outFileStr)
        if pred_img.mode != 'L':
            pred_img = pred_img.convert('L')
        intensity_pred=np.array(pred_img)
        PRDintensity.append(intensity_pred)
    rmse=[]
    for i in range(len(GTintensity)):
        rmse.append(RMSE(GTintensity[i], PRDintensity[i]))
    corr_error=0.0
    for i in range(len(rmse)):
        corr_error+=rmse[i]
    corr_error=corr_error/len(rmse)
    
    #Reconstruct the aberration
    for countNoll in nollRange: 
        if countNoll>3:
            (nz,mz) = noll_to_zern(countNoll)
            zernikeCoeff[countNoll-1] = zernikecoefficients[countNoll-4]
            zernikeField_pred = Zernike(zernikeField_pred,nz,mz,zernikeRadius,zernikeCoeff[countNoll-1],units='rad')
        else:
            (nz,mz) = noll_to_zern(countNoll)
            zernikeCoeff[countNoll-1] = 0
            zernikeField_pred = Zernike(zernikeField_pred,nz,mz,zernikeRadius,zernikeCoeff[countNoll-1],units='rad')
    
    #Reconstruction Error
    distField_pred = zernikeField_pred
    # Prefocus field and apply aperture
    distField_pred = Lens(f1,0,0,distField_pred)
    distField_pred = SmoothCircAperture(distField_pred, apertureRadius, apertureSmoothWidth)

    RECONintensity=[]
    for causticStr,causticPos in causticPlanes:
        cField_pred=LensFresnel(distField_pred,f2,focalLength+causticPos*zR)
        cField_pred=Convert(cField_pred)
        outputField_pred=Interpol(cField_pred,outputSize,outputPixelnumber)
        intensity_pred=Intensity(outputField_pred)
        outFileStr= outputPathStr_recons + outFileName + causticStr + ".png"
        mpimg.imsave(outFileStr,-intensity_pred,cmap='Greys')
        pred_img = Image.open(outFileStr)
        if pred_img.mode != 'L':
            pred_img = pred_img.convert('L')
        intensity_pred=np.array(pred_img)
        RECONintensity.append(intensity_pred)
    
    INPintensity=[]
    for i in range(3):
        inp_img_path=inputpath+inputlist[runCount*3+i]
        inp_img = Image.open(inp_img_path)
        if inp_img.mode != 'L':
            inp_img = inp_img.convert('L')
        intensity_inp=np.array(inp_img)
        INPintensity.append(intensity_inp)
    
    rmse=[]
    for i in range(len(INPintensity)):
        rmse.append(RMSE(INPintensity[i], RECONintensity[i]))
    recons_error=0.0
    for i in range(len(rmse)):
        recons_error+=rmse[i]
    recons_error=recons_error/len(rmse)
    
    Metrics.append([mae, wavefront_error_unw, wavefront_error_w, corr_error, recons_error])
    print(mae)
    print(wavefront_error_unw)
    print(wavefront_error_w)
    print(corr_error)
    print(recons_error)

AvgMAE=0.0
for i in range(len(Metrics)):
    AvgMAE+=Metrics[i][0]
AvgMAE=AvgMAE/len(Metrics)

AvgWavefrontErr_unw=0.0
for i in range(len(Metrics)):
    AvgWavefrontErr_unw+=Metrics[i][1]
AvgWavefrontErr_unw=AvgWavefrontErr_unw/len(Metrics)

AvgWavefrontErr_w=0.0
for i in range(len(Metrics)):
    AvgWavefrontErr_w+=Metrics[i][2]
AvgWavefrontErr_w=AvgWavefrontErr_w/len(Metrics)

AvgCorrErr=0.0
for i in range(len(Metrics)):
    AvgCorrErr+=Metrics[i][3]
AvgCorrErr=AvgCorrErr/len(Metrics)

AvgReconsErr=0.0
for i in range(len(Metrics)):
    AvgReconsErr+=Metrics[i][4]
AvgReconsErr=AvgReconsErr/len(Metrics)

print("MAE(Zernike Coefficients):%f"%(AvgMAE))
print("WaveFront Error(Unwrapped Phase):%f"%(AvgWavefrontErr_unw))
print("WaveFront Error(Wrapped Phase):%f"%(AvgWavefrontErr_w))
print("Correction Error(Intensity):%f"%(AvgCorrErr))
print("Reconstruction Error(Intensity):%f"%(AvgReconsErr))

Metrics_np = np.array(Metrics)
np.save(filename+"_Metrics.npy",Metrics_np)

Metrics_pd=pd.DataFrame(Metrics_np[:,:], columns=['MAE','WavefrontErr_unw','WavefrontErr_w','CorrErr','ReconsErr'])
pd.plotting.scatter_matrix(Metrics_pd, alpha=0.15, figsize=None, ax=None, grid=False,
                           diagonal='hist', marker='.', density_kwds=None, hist_kwds=None,
                           range_padding=0.0)
plt.savefig(filename+"Scatter_Matrix.png",dpi=1000, bbox_inches='tight')