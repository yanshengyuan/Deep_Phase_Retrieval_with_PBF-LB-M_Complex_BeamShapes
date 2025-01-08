import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})

bench=np.load('chair_phasenet_Metrics.npy')
bench=np.delete(bench, 2, 1)
tear=np.load('tear_vit_Metrics.npy')
tear=np.delete(tear, 2, 1)
rec=np.load('rec_vit_Metrics.npy')
rec=np.delete(rec, 2, 1)
hat=np.load('hat_vit_Metrics.npy')
hat=np.delete(hat, 2, 1)
ring=np.load('ring_vit_Metrics.npy')
ring=np.delete(ring, 2, 1)
gauss=np.load('gauss_vit_Metrics.npy')
gauss=np.delete(gauss, 2, 1)

gauss_resnet=np.load('gauss_resnet_Metrics.npy')
gauss_resnet=np.delete(gauss_resnet, 2, 1)

bench=pd.DataFrame(bench, columns=['MAE(Z) [rad]', 'Wavefront error [$\lambda$]', 'Correction error [a.u.]', 'Reconstruction error [a.u.]'])
tear=pd.DataFrame(tear, columns=['MAE(Z) [rad]', 'Wavefront error [$\lambda$]', 'Correction error [a.u.]', 'Reconstruction error [a.u.]'])
rec=pd.DataFrame(rec, columns=['MAE(Z) [rad]', 'Wavefront error [$\lambda$]', 'Correction error [a.u.]', 'Reconstruction error [a.u.]'])
hat=pd.DataFrame(hat, columns=['MAE(Z) [rad]', 'Wavefront error [$\lambda$]', 'Correction error [a.u.]', 'Reconstruction error [a.u.]'])
ring=pd.DataFrame(ring, columns=['MAE(Z) [rad]', 'Wavefront error [$\lambda$]', 'Correction error [a.u.]', 'Reconstruction error [a.u.]'])
gauss=pd.DataFrame(gauss_resnet, columns=['MAE(Z) [rad]', 'Wavefront error [$\lambda$]', 'Correction error [a.u.]', 'Reconstruction error [a.u.]'])

bench['Wavefront error [$\lambda$]'] = bench['Wavefront error [$\lambda$]'] / (2 * np.pi)
tear['Wavefront error [$\lambda$]'] = tear['Wavefront error [$\lambda$]'] / (2 * np.pi)
rec['Wavefront error [$\lambda$]'] = rec['Wavefront error [$\lambda$]'] / (2 * np.pi)
hat['Wavefront error [$\lambda$]'] = hat['Wavefront error [$\lambda$]'] / (2 * np.pi)
ring['Wavefront error [$\lambda$]'] = ring['Wavefront error [$\lambda$]'] / (2 * np.pi)
gauss['Wavefront error [$\lambda$]'] = gauss['Wavefront error [$\lambda$]'] / (2 * np.pi)

transparency=0.5
plt.figure(figsize=(14,4), dpi=400)
plt.figure(1)

'''
ax1=plt.subplot(1,2,1)
sns.kdeplot(data=bench, x="MAE(Z) [rad]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='Chair-PhaseNet')
sns.kdeplot(data=tear, x="MAE(Z) [rad]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='Tear-ViT-Base')
sns.kdeplot(data=rec, x="MAE(Z) [rad]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='RecTophat-ViT-Base')
sns.kdeplot(data=hat, x="MAE(Z) [rad]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='Tophat-ViT-Base')
sns.kdeplot(data=ring, x="MAE(Z) [rad]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='Ring-ViT-Base')
sns.kdeplot(data=gauss, x="MAE(Z) [rad]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='Gaussian-ResNet18')
ax1.legend()
#mae_distr.figure.savefig('mae.png',dpi=600)
#plt.close()

ax2=plt.subplot(1,2,2)
sns.kdeplot(data=bench, x="Reconstruction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='Chair-PhaseNet')
sns.kdeplot(data=tear, x="Reconstruction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='Tear-ViT-Base')
sns.kdeplot(data=rec, x="Reconstruction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='RecTophat-ViT-Base')
sns.kdeplot(data=hat, x="Reconstruction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='Tophat-ViT-Base')
sns.kdeplot(data=ring, x="Reconstruction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='Ring-ViT-Base')
sns.kdeplot(data=gauss, x="Reconstruction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='Gaussian-ResNet18')
ax2.legend()
#wavefrontErr_distr.figure.savefig('wavefrontErr.png',dpi=600)
#plt.close()
plt.savefig('ErrDistr_detection.png', bbox_inches='tight')
'''

#'''
ax1=plt.subplot(1,2,1)
sns.kdeplot(data=bench, x="Wavefront error [$\lambda$]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='Chair-PhaseNet')
sns.kdeplot(data=tear, x="Wavefront error [$\lambda$]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='Tear-ViT-Base')
sns.kdeplot(data=rec, x="Wavefront error [$\lambda$]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='RecTophat-ViT-Base')
sns.kdeplot(data=hat, x="Wavefront error [$\lambda$]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='Tophat-ViT-Base')
sns.kdeplot(data=ring, x="Wavefront error [$\lambda$]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='Ring-ViT-Base')
sns.kdeplot(data=gauss, x="Wavefront error [$\lambda$]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax1, label='Gaussian-ViT-Base')
ax1.legend()

ax2=plt.subplot(1,2,2)
sns.kdeplot(data=bench, x="Correction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='Chair-PhaseNet')
sns.kdeplot(data=tear, x="Correction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='Tear-ViT-Base')
sns.kdeplot(data=rec, x="Correction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='RecTophat-ViT-Base')
sns.kdeplot(data=hat, x="Correction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='Tophat-ViT-Base')
sns.kdeplot(data=ring, x="Correction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='Ring-ViT-Base')
sns.kdeplot(data=gauss, x="Correction error [a.u.]", multiple='stack', linewidth=0, 
            palette="crest", alpha=transparency, log_scale=True, ax=ax2, label='Gaussian-ViT-Base')
ax2.legend()
#reconsErr_distr.figure.savefig('reconsErr.png',dpi=600)
#plt.close()
plt.savefig('ErrDistr_correction.png', bbox_inches='tight')
#'''