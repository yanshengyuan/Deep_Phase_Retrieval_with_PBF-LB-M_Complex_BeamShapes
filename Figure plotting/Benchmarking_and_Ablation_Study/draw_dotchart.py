import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 13})

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    n_bars = len(data)

    bar_width = total_width / n_bars

    bars = []

    for i, (name, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        bars.append(bar[0])

    if legend:
        ax.legend(bars, data.keys())

numplane=pd.read_excel('ablation_numplanes.xlsx')
causticdistance=pd.read_excel('ablation_causticdistance.xlsx')
prefocpst=pd.read_excel('ablation_prefocpst.xlsx')
#sym=pd.read_excel('ablation_sym&asym.xlsx')

benchmarking=pd.read_excel('benchmarking.xlsx')
orderz=pd.read_excel('ablation_orderZernike.xlsx')

#'''
#Zernike max order analysis starts
resnet_bench=list(orderz.iloc[0][:3])
resnet_gaussian=list(orderz.iloc[1][:3])
resnet_tear=list(orderz.iloc[2][:3])
resnet_rec=list(orderz.iloc[3][:3])
resnet_hat=list(orderz.iloc[4][:3])
resnet_ring=list(orderz.iloc[5][:3])
mlp_bench=list(orderz.iloc[0][3:6])
mlp_gaussian=list(orderz.iloc[1][3:6])
mlp_tear=list(orderz.iloc[2][3:6])
mlp_rec=list(orderz.iloc[3][3:6])
mlp_hat=list(orderz.iloc[4][3:6])
mlp_ring=list(orderz.iloc[5][3:6])
vit_bench=list(orderz.iloc[0][6:9])
vit_gaussian=list(orderz.iloc[1][6:9])
vit_tear=list(orderz.iloc[2][6:9])
vit_rec=list(orderz.iloc[3][6:9])
vit_hat=list(orderz.iloc[4][6:9])
vit_ring=list(orderz.iloc[5][6:9])

plt.figure(figsize=(18,5), dpi=400)
plt.figure(1)

ind=np.arange(3)

ax1=plt.subplot(1,3,1)
data1={
       'Chair':resnet_bench,
       'Tear':resnet_tear,
       'RecTopHat':resnet_rec,
       'TopHat':resnet_hat,
       'Ring':resnet_ring,
       'Gaussian':resnet_gaussian
       }
bar_plot(ax1, data1)
ax1.set_ylabel('MAE(Z) [rad]')
ax1.set_xticks(ind)
ax1.set_xlabel('Max Noll index')
ax1.set_xticklabels( ('10', '15', '21') )
ax1.set_title('ResNet18', fontsize=20)

ax2=plt.subplot(1,3,2)
data2={
       'Chair':mlp_bench,
       'Tear':mlp_tear,
       'RecTopHat':mlp_rec,
       'TopHat':mlp_hat,
       'Ring':mlp_ring,
       'Gaussian':mlp_gaussian
       }
bar_plot(ax2, data2)
ax2.set_ylabel('MAE(Z) [rad]')
ax2.set_xticks(ind)
ax2.set_xlabel('Max Noll index')
ax2.set_xticklabels( ('10', '15', '21') )
ax2.set_title('MLP12', fontsize=20)

ax3=plt.subplot(1,3,3)
data3={
       'Chair':vit_bench,
       'Tear':vit_tear,
       'RecTopHat':vit_rec,
       'TopHat':vit_hat,
       'Ring':vit_ring,
       'Gaussian':vit_gaussian
       }
bar_plot(ax3, data3)
ax3.set_ylabel('MAE(Z) [rad]')
ax3.set_xticks(ind)
ax3.set_xlabel('Max Noll index')
ax3.set_xticklabels( ('10', '15', '21') )
ax3.set_title('ViT-Base', fontsize=20)

plt.savefig('orderzs_dotchart.png', bbox_inches='tight')
#Zernike max order analysis finishes
#'''

'''
#model scaling analysis starts
model_scaling=pd.read_excel('ablation_modelscaling_val.xlsx')
params_resnet=[4.9, 11.2, 21.3, 23.5, 42.5]
params_mlp=[75.6, 86.3, 102, 105, 109]
params_vit=[5.5, 21.6, 85.7, 303, 630]
dep_resnet=[10, 18, 34, 50, 101]
dep_mlp=[3, 6, 12, 18, 24]
dep_vit=[14, 26, 34]
resnet_bench=model_scaling.iloc[0][:5]
resnet_gaussian=model_scaling.iloc[1][:5]
resnet_tear=model_scaling.iloc[2][:5]
resnet_rec=model_scaling.iloc[3][:5]
resnet_hat=model_scaling.iloc[4][:5]
resnet_ring=model_scaling.iloc[5][:5]
mlp_bench=model_scaling.iloc[0][5:10]
mlp_gaussian=model_scaling.iloc[1][5:10]
mlp_tear=model_scaling.iloc[2][5:10]
mlp_rec=model_scaling.iloc[3][5:10]
mlp_hat=model_scaling.iloc[4][5:10]
mlp_ring=model_scaling.iloc[5][5:10]
vit_bench=model_scaling.iloc[0][12:15]
vit_gaussian=model_scaling.iloc[1][12:15]
vit_tear=model_scaling.iloc[2][12:15]
vit_rec=model_scaling.iloc[3][12:15]
vit_hat=model_scaling.iloc[4][12:15]
vit_ring=model_scaling.iloc[5][12:15]

plt.figure(figsize=(15,10), dpi=400)
plt.figure(1)
ax1=plt.subplot(2,3,1)
ax1.stem(dep_resnet, resnet_bench, label='Chair', markerfmt='o', linefmt='--')
ax1.stem(dep_resnet, resnet_tear, label='Tear', markerfmt='^', linefmt='--')
ax1.stem(dep_resnet, resnet_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax1.stem(dep_resnet, resnet_hat, label='Tophat', markerfmt='*', linefmt='--')
ax1.stem(dep_resnet, resnet_ring, label='Ring', markerfmt='v', linefmt='--')
ax1.stem(dep_resnet, resnet_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax1.set_xlabel('Depth')
ax1.set_ylabel('MAE(Z) [rad]')
ax1.legend()
ax1.set_title('ResNet models', fontsize=20)
plt.yscale('log')

ax2=plt.subplot(2,3,2)
ax2.stem(dep_mlp, mlp_bench, label='Chair', markerfmt='o', linefmt='--')
ax2.stem(dep_mlp, mlp_tear, label='Tear', markerfmt='^', linefmt='--')
ax2.stem(dep_mlp, mlp_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax2.stem(dep_mlp, mlp_hat, label='Tophat', markerfmt='*', linefmt='--')
ax2.stem(dep_mlp, mlp_ring, label='Ring', markerfmt='v', linefmt='--')
ax2.stem(dep_mlp, mlp_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax2.set_xlabel('Depth')
ax2.set_ylabel('MAE(Z) [rad]')
ax2.legend()
ax2.set_title('MLP models', fontsize=20)
plt.yscale('log')

ax3=plt.subplot(2,3,3)
ax3.stem(dep_vit, vit_bench, label='Chair', markerfmt='o', linefmt='--')
ax3.stem(dep_vit, vit_tear, label='Tear', markerfmt='^', linefmt='--')
ax3.stem(dep_vit, vit_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax3.stem(dep_vit, vit_hat, label='Tophat', markerfmt='*', linefmt='--')
ax3.stem(dep_vit, vit_ring, label='Ring', markerfmt='v', linefmt='--')
ax3.stem(dep_vit, vit_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax3.set_xlabel('Depth')
ax3.set_ylabel('MAE(Z) [rad]')
ax3.legend()
ax3.set_title('ViT models', fontsize=20)
plt.yscale('log')

resnet_bench=model_scaling.iloc[0][:5]
resnet_gaussian=model_scaling.iloc[1][:5]
resnet_tear=model_scaling.iloc[2][:5]
resnet_rec=model_scaling.iloc[3][:5]
resnet_hat=model_scaling.iloc[4][:5]
resnet_ring=model_scaling.iloc[5][:5]
mlp_bench=model_scaling.iloc[0][5:10]
mlp_gaussian=model_scaling.iloc[1][5:10]
mlp_tear=model_scaling.iloc[2][5:10]
mlp_rec=model_scaling.iloc[3][5:10]
mlp_hat=model_scaling.iloc[4][5:10]
mlp_ring=model_scaling.iloc[5][5:10]
vit_bench=model_scaling.iloc[0][10:15]
vit_gaussian=model_scaling.iloc[1][10:15]
vit_tear=model_scaling.iloc[2][10:15]
vit_rec=model_scaling.iloc[3][10:15]
vit_hat=model_scaling.iloc[4][10:15]
vit_ring=model_scaling.iloc[5][10:15]

ax4=plt.subplot(2,3,4)
ax4.stem(params_resnet, resnet_bench, label='Chair', markerfmt='o', linefmt='--')
ax4.stem(params_resnet, resnet_tear, label='Tear', markerfmt='^', linefmt='--')
ax4.stem(params_resnet, resnet_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax4.stem(params_resnet, resnet_hat, label='Tophat', markerfmt='*', linefmt='--')
ax4.stem(params_resnet, resnet_ring, label='Ring', markerfmt='v', linefmt='--')
ax4.stem(params_resnet, resnet_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax4.set_xlabel('Parameters [M]')
ax4.set_ylabel('MAE(Z) [rad]')
ax4.legend()
plt.yscale('log')

ax5=plt.subplot(2,3,5)
ax5.stem(params_mlp, mlp_bench, label='Bench', markerfmt='o', linefmt='--')
ax5.stem(params_mlp, mlp_tear, label='Tear', markerfmt='^', linefmt='--')
ax5.stem(params_mlp, mlp_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax5.stem(params_mlp, mlp_hat, label='Tophat', markerfmt='*', linefmt='--')
ax5.stem(params_mlp, mlp_ring, label='Ring', markerfmt='v', linefmt='--')
ax5.stem(params_mlp, mlp_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax5.set_xlabel('Parameters [M]')
ax5.set_ylabel('MAE(Z) [rad]')
ax5.legend()
plt.yscale('log')

ax6=plt.subplot(2,3,6)
ax6.stem(params_vit, vit_bench, label='Bench', markerfmt='o', linefmt='--')
ax6.stem(params_vit, vit_tear, label='Tear', markerfmt='^', linefmt='--')
ax6.stem(params_vit, vit_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax6.stem(params_vit, vit_hat, label='Tophat', markerfmt='*', linefmt='--')
ax6.stem(params_vit, vit_ring, label='Ring', markerfmt='v', linefmt='--')
ax6.stem(params_vit, vit_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax6.set_xlabel('Parameters [M]')
ax6.set_ylabel('MAE(Z) [rad]')
ax6.legend()
plt.yscale('log')

plt.savefig('model_scaling_dotchart.png', bbox_inches='tight')
plt.show()
#model scaling analysis finishes
'''

'''
#numplanes analysis starts
resnet_bench=numplane.iloc[0][:4]
resnet_gaussian=numplane.iloc[1][:4]
resnet_tear=numplane.iloc[2][:4]
resnet_rec=numplane.iloc[3][:4]
resnet_hat=numplane.iloc[4][:4]
resnet_ring=numplane.iloc[5][:4]
mlp_bench=numplane.iloc[0][4:8]
mlp_gaussian=numplane.iloc[1][4:8]
mlp_tear=numplane.iloc[2][4:8]
mlp_rec=numplane.iloc[3][4:8]
mlp_hat=numplane.iloc[4][4:8]
mlp_ring=numplane.iloc[5][4:8]
vit_bench=numplane.iloc[0][8:12]
vit_gaussian=numplane.iloc[1][8:12]
vit_tear=numplane.iloc[2][8:12]
vit_rec=numplane.iloc[3][8:12]
vit_hat=numplane.iloc[4][8:12]
vit_ring=numplane.iloc[5][8:12]

num_planes=[1, 3, 5, 7]
plt.figure(figsize=(15,5), dpi=400)
plt.figure(1)

ax1=plt.subplot(1,3,1)
ax1.stem(num_planes, resnet_bench, label='Chair', markerfmt='o', linefmt='--')
ax1.stem(num_planes, resnet_tear, label='Tear', markerfmt='^', linefmt='--')
ax1.stem(num_planes, resnet_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax1.stem(num_planes, resnet_hat, label='Tophat', markerfmt='*', linefmt='--')
ax1.stem(num_planes, resnet_ring, label='Ring', markerfmt='v', linefmt='--')
ax1.stem(num_planes, resnet_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax1.set_xlabel('Number of input caustic planes')
ax1.set_ylabel('MAE(Z) [rad]')
ax1.legend()
ax1.set_title('ResNet18', fontsize=20)
plt.yscale('log')

ax2=plt.subplot(1,3,2)
ax2.stem(num_planes, mlp_bench, label='Chair', markerfmt='o', linefmt='--')
ax2.stem(num_planes, mlp_tear, label='Tear', markerfmt='^', linefmt='--')
ax2.stem(num_planes, mlp_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax2.stem(num_planes, mlp_hat, label='Tophat', markerfmt='*', linefmt='--')
ax2.stem(num_planes, mlp_ring, label='Ring', markerfmt='v', linefmt='--')
ax2.stem(num_planes, mlp_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax2.set_xlabel('Number of input caustic planes')
ax2.set_ylabel('MAE(Z) [rad]')
ax2.legend()
ax2.set_title('MLP12', fontsize=20)
plt.yscale('log')

ax3=plt.subplot(1,3,3)
ax3.stem(num_planes, vit_bench, label='Chair', markerfmt='o', linefmt='--')
ax3.stem(num_planes, vit_tear, label='Tear', markerfmt='^', linefmt='--')
ax3.stem(num_planes, vit_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax3.stem(num_planes, vit_hat, label='Tophat', markerfmt='*', linefmt='--')
ax3.stem(num_planes, vit_ring, label='Ring', markerfmt='v', linefmt='--')
ax3.stem(num_planes, vit_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax3.set_xlabel('Number of input caustic planes')
ax3.set_ylabel('MAE(Z) [rad]')
ax3.legend()
ax3.set_title('ViT-Base', fontsize=20)
plt.yscale('log')

plt.savefig('numplanes_dotchart.png', bbox_inches='tight')
plt.show()
#numplane analysis finishes
'''

'''
#caustic distance analysis starts
resnet_bench=causticdistance.iloc[0][:3]
resnet_gaussian=causticdistance.iloc[1][:3]
resnet_tear=causticdistance.iloc[2][:3]
resnet_rec=causticdistance.iloc[3][:3]
resnet_hat=causticdistance.iloc[4][:3]
resnet_ring=causticdistance.iloc[5][:3]
mlp_bench=causticdistance.iloc[0][3:6]
mlp_gaussian=causticdistance.iloc[1][3:6]
mlp_tear=causticdistance.iloc[2][3:6]
mlp_rec=causticdistance.iloc[3][3:6]
mlp_hat=causticdistance.iloc[4][3:6]
mlp_ring=causticdistance.iloc[5][3:6]
vit_bench=causticdistance.iloc[0][6:9]
vit_gaussian=causticdistance.iloc[1][6:9]
vit_tear=causticdistance.iloc[2][6:9]
vit_rec=causticdistance.iloc[3][6:9]
vit_hat=causticdistance.iloc[4][6:9]
vit_ring=causticdistance.iloc[5][6:9]

caustic_distances=[13.9, 17.4, 20.9]
plt.figure(figsize=(16.6,5), dpi=400)
plt.figure(1)

ax1=plt.subplot(1,3,1)
ax1.stem(caustic_distances, resnet_bench, label='Chair', markerfmt='o', linefmt='--')
ax1.stem(caustic_distances, resnet_tear, label='Tear', markerfmt='^', linefmt='--')
ax1.stem(caustic_distances, resnet_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax1.stem(caustic_distances, resnet_hat, label='Tophat', markerfmt='*', linefmt='--')
ax1.stem(caustic_distances, resnet_ring, label='Ring', markerfmt='v', linefmt='--')
ax1.stem(caustic_distances, resnet_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax1.set_xlabel('Distance of pre/post Fourier plane from Fourier plane (mm)')
ax1.set_ylabel('MAE(Z) [rad]')
ax1.legend()
ax1.set_title('ResNet18', fontsize=20)
ax1.yaxis.set_label_coords(-0.02, 0.7)
plt.yscale('log')

ax2=plt.subplot(1,3,2)
ax2.stem(caustic_distances, mlp_bench, label='Chair', markerfmt='o', linefmt='--')
ax2.stem(caustic_distances, mlp_tear, label='Tear', markerfmt='^', linefmt='--')
ax2.stem(caustic_distances, mlp_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax2.stem(caustic_distances, mlp_hat, label='Tophat', markerfmt='*', linefmt='--')
ax2.stem(caustic_distances, mlp_ring, label='Ring', markerfmt='v', linefmt='--')
ax2.stem(caustic_distances, mlp_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax2.set_xlabel('Distance of pre/post Fourier plane from Fourier plane (mm)')
ax2.set_ylabel('MAE(Z) [rad]')
ax2.legend()
ax2.set_title('MLP12', fontsize=20)
ax2.yaxis.set_label_coords(-0.02, 0.85)
plt.yscale('log')

ax3=plt.subplot(1,3,3)
ax3.stem(caustic_distances, vit_bench, label='Chair', markerfmt='o', linefmt='--')
ax3.stem(caustic_distances, vit_tear, label='Tear', markerfmt='^', linefmt='--')
ax3.stem(caustic_distances, vit_rec, label='RecTophat', markerfmt='x', linefmt='--')
ax3.stem(caustic_distances, vit_hat, label='Tophat', markerfmt='*', linefmt='--')
ax3.stem(caustic_distances, vit_ring, label='Ring', markerfmt='v', linefmt='--')
ax3.stem(caustic_distances, vit_gaussian, label='Gaussian', markerfmt='D', linefmt='--')
ax3.set_xlabel('Distance of pre/post Fourier plane from Fourier plane (mm)')
ax3.set_ylabel('MAE(Z) [rad]')
ax3.legend()
ax3.set_title('ViT-Base', fontsize=20)
plt.yscale('log')
pos = ax3.get_position()
new_pos = [pos.x0 + 0.01, pos.y0, pos.width, pos.height]  # Move it to the right by 0.05 units
ax3.set_position(new_pos)

plt.savefig('causticdistances_dotchart.png', bbox_inches='tight')
#caustic distance analysis finishes
'''

'''
#prefocpst analysis starts
resnet_bench=list(prefocpst.iloc[0][:4])
resnet_gaussian=list(prefocpst.iloc[1][:4])
resnet_tear=list(prefocpst.iloc[2][:4])
resnet_rec=list(prefocpst.iloc[3][:4])
resnet_hat=list(prefocpst.iloc[4][:4])
resnet_ring=list(prefocpst.iloc[5][:4])
mlp_bench=list(prefocpst.iloc[0][4:8])
mlp_gaussian=list(prefocpst.iloc[1][4:8])
mlp_tear=list(prefocpst.iloc[2][4:8])
mlp_rec=list(prefocpst.iloc[3][4:8])
mlp_hat=list(prefocpst.iloc[4][4:8])
mlp_ring=list(prefocpst.iloc[5][4:8])
vit_bench=list(prefocpst.iloc[0][8:12])
vit_gaussian=list(prefocpst.iloc[1][8:12])
vit_tear=list(prefocpst.iloc[2][8:12])
vit_rec=list(prefocpst.iloc[3][8:12])
vit_hat=list(prefocpst.iloc[4][8:12])
vit_ring=list(prefocpst.iloc[5][8:12])

plt.figure(figsize=(18,5), dpi=400)
plt.figure(1)

ind=np.arange(4)

ax1=plt.subplot(1,3,1)
data1={
       'Chair':resnet_bench,
       'Tear':resnet_tear,
       'RecTopHat':resnet_rec,
       'TopHat':resnet_hat,
       'Ring':resnet_ring,
       'Gaussian':resnet_gaussian
       }
bar_plot(ax1, data1)
ax1.set_ylabel('MAE(Z) [rad]')
ax1.set_xticks(ind)
ax1.set_xticklabels( ('pre-F', 'pre-F+F', 'post-F+F', 'post-F') )
ax1.set_title('ResNet18', fontsize=20)

ax2=plt.subplot(1,3,2)
data2={
       'Chair':mlp_bench,
       'Tear':mlp_tear,
       'RecTopHat':mlp_rec,
       'TopHat':mlp_hat,
       'Ring':mlp_ring,
       'Gaussian':mlp_gaussian
       }
bar_plot(ax2, data2)
ax2.set_ylabel('MAE(Z) [rad]')
ax2.set_xticks(ind)
ax2.set_xticklabels( ('pre-F', 'pre-F+F', 'post-F+F', 'post-F') )
ax2.set_title('MLP12', fontsize=20)

ax3=plt.subplot(1,3,3)
data3={
       'Chair':vit_bench,
       'Tear':vit_tear,
       'RecTopHat':vit_rec,
       'TopHat':vit_hat,
       'Ring':vit_ring,
       'Gaussian':vit_gaussian
       }
bar_plot(ax3, data3)
ax3.set_ylabel('MAE(Z) [rad]')
ax3.set_xticks(ind)
ax3.set_xticklabels( ('pre-F', 'pre-F+F', 'post-F+F', 'post-F') )
ax3.set_title('ViT-Base', fontsize=20)

plt.savefig('prefocpsts_dotchart.png', bbox_inches='tight')
#prefocpst analysis finishes
'''

'''
                #sym analysis starts
                resnet_bench=list(sym.iloc[0][:3])
                resnet_gaussian=list(sym.iloc[1][:3])
                resnet_tear=list(sym.iloc[2][:3])
                resnet_rec=list(sym.iloc[3][:3])
                resnet_hat=list(sym.iloc[4][:3])
                resnet_ring=list(sym.iloc[5][:3])
                mlp_bench=list(sym.iloc[0][3:6])
                mlp_gaussian=list(sym.iloc[1][3:6])
                mlp_tear=list(sym.iloc[2][3:6])
                mlp_rec=list(sym.iloc[3][3:6])
                mlp_hat=list(sym.iloc[4][3:6])
                mlp_ring=list(sym.iloc[5][3:6])
                vit_bench=list(sym.iloc[0][6:9])
                vit_gaussian=list(sym.iloc[1][6:9])
                vit_tear=list(sym.iloc[2][6:9])
                vit_rec=list(sym.iloc[3][6:9])
                vit_hat=list(sym.iloc[4][6:9])
                vit_ring=list(sym.iloc[5][6:9])
                
                plt.figure(figsize=(18,5), dpi=400)
                plt.figure(1)
                
                ind=np.arange(3)
                
                ax1=plt.subplot(1,3,1)
                data1={
                       'Chair':resnet_bench,
                       'Gaussian':resnet_gaussian,
                       'Tear':resnet_tear,
                       'RecTopHat':resnet_rec,
                       'TopHat':resnet_hat,
                       'Ring':resnet_ring
                       }
                bar_plot(ax1, data1)
                ax1.set_ylabel('MAE(Z) [rad]')
                ax1.set_xticks(ind)
                ax1.set_xticklabels( ('asym-prefoc', 'sym', 'asym-postfoc') )
                ax1.set_title('ResNet18', fontsize=20)
                plt.ylim(0, 0.013)
                
                ax2=plt.subplot(1,3,2)
                data2={
                       'Chair':mlp_bench,
                       'Gaussian':mlp_gaussian,
                       'Tear':mlp_tear,
                       'RecTopHat':mlp_rec,
                       'TopHat':mlp_hat,
                       'Ring':mlp_ring
                       }
                bar_plot(ax2, data2)
                ax2.set_ylabel('MAE(Z) [rad]')
                ax2.set_xticks(ind)
                ax2.set_xticklabels( ('asym-prefoc', 'sym', 'asym-postfoc') )
                ax2.set_title('MLP12', fontsize=20)
                plt.ylim(0, 0.013)
                
                ax3=plt.subplot(1,3,3)
                data3={
                       'Chair':vit_bench,
                       'Gaussian':vit_gaussian,
                       'Tear':vit_tear,
                       'RecTopHat':vit_rec,
                       'TopHat':vit_hat,
                       'Ring':vit_ring
                       }
                bar_plot(ax3, data3)
                ax3.set_ylabel('MAE(Z) [rad]')
                ax3.set_xticks(ind)
                ax3.set_xticklabels( ('asym-prefoc', 'sym', 'asym-postfoc') )
                ax3.set_title('ViT-Base', fontsize=20)
                plt.ylim(0, 0.013)
                
                plt.savefig('syms_dotchart.png', bbox_inches='tight')
                #sym analysis finishes
'''

'''
#benchmarking metrics comparison starts
colors = ['r','b','g','y','b','p']
fig=plt.figure(figsize=(14, 7), dpi=400)
smnet_mae=[]
mockl_mae=[]
mlp_mae=[]
resnet_mae=[]
phasenet_mae=[]
vit_mae=[]
smnet_recons=[]
mockl_recons=[]
mlp_recons=[]
resnet_recons=[]
phasenet_recons=[]
vit_recons=[]
smnet_wave=[]
mockl_wave=[]
mlp_wave=[]
resnet_wave=[]
phasenet_wave=[]
vit_wave=[]
smnet_corr=[]
mockl_corr=[]
mlp_corr=[]
resnet_corr=[]
phasenet_corr=[]
vit_corr=[]
for i in range(6):
    resnet_mae.append(benchmarking.iloc[i*6+0][0])
    mlp_mae.append(benchmarking.iloc[i*6+1][0])
    vit_mae.append(benchmarking.iloc[i*6+2][0])
    phasenet_mae.append(benchmarking.iloc[i*6+3][0])
    mockl_mae.append(benchmarking.iloc[i*6+4][0])
    smnet_mae.append(benchmarking.iloc[i*6+5][0])
    resnet_recons.append(benchmarking.iloc[i*6+0][1])
    mlp_recons.append(benchmarking.iloc[i*6+1][1])
    vit_recons.append(benchmarking.iloc[i*6+2][1])
    phasenet_recons.append(benchmarking.iloc[i*6+3][1])
    mockl_recons.append(benchmarking.iloc[i*6+4][1])
    smnet_recons.append(benchmarking.iloc[i*6+5][1])
    resnet_wave.append(benchmarking.iloc[i*6+0][2])
    mlp_wave.append(benchmarking.iloc[i*6+1][2])
    vit_wave.append(benchmarking.iloc[i*6+2][2])
    phasenet_wave.append(benchmarking.iloc[i*6+3][2])
    mockl_wave.append(benchmarking.iloc[i*6+4][2])
    smnet_wave.append(benchmarking.iloc[i*6+5][2])
    resnet_corr.append(benchmarking.iloc[i*6+0][3])
    mlp_corr.append(benchmarking.iloc[i*6+1][3])
    vit_corr.append(benchmarking.iloc[i*6+2][3])
    phasenet_corr.append(benchmarking.iloc[i*6+3][3])
    mockl_corr.append(benchmarking.iloc[i*6+4][3])
    smnet_corr.append(benchmarking.iloc[i*6+5][3])

result_mae=[
        resnet_mae,
        mlp_mae,
        vit_mae,
        phasenet_mae,
        mockl_mae,
        smnet_mae
        ]
result_recons=[
        resnet_recons,
        mlp_recons,
        vit_recons,
        phasenet_recons,
        mockl_recons,
        smnet_recons
        ]
result_wave=[
        resnet_wave,
        mlp_wave,
        vit_wave,
        phasenet_wave,
        mockl_wave,
        smnet_wave
        ]
result_corr=[
        resnet_corr,
        mlp_corr,
        vit_corr,
        phasenet_corr,
        mockl_corr,
        smnet_corr
        ]
#'''

'''
ax1=plt.subplot(121, projection='3d')
result_mae = np.array(result_mae)
ax1.set_xlabel('Beam shapes', labelpad=13)
ax1.set_ylabel('Models', labelpad=13)
ax1.set_zlabel('MAE(Z) [rad]')
xlabels = np.array(['Chair', 'Tear', 'RecTophat', 'Tophat', 'Ring', 'Gaussian'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['ResNet18','MLP12','ViT-Base','PhaseNet','Leonhard et al.','smNet'])
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos=result_mae
zpos = zpos.ravel()

dx=0.335
dy=0.335
dz=zpos

ax1.w_xaxis.set_ticks(xpos + dx/2)
ax1.w_xaxis.set_ticklabels(xlabels, ha='right')

ax1.w_yaxis.set_ticks(ypos + dy/2)
ax1.w_yaxis.set_ticklabels(ylabels, va='bottom')
ax1.tick_params(axis='y', which='major', pad=0)
ax1.tick_params(axis='x', which='major', pad=-5)
ax1.view_init(elev=26, azim=-52)
for tick in ax1.get_yticklabels():
    tick.set_horizontalalignment('left')

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.rainbow(values)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
ax1.set_title('a. MAE(Z)', fontsize=20, fontweight='bold')

ax2=plt.subplot(122, projection='3d')
result_recons = np.array(result_recons)
ax2.set_xlabel('Beam shapes', labelpad=13)
ax2.set_ylabel('Models', labelpad=13)
ax2.set_zlabel('Reconstruction error [a.u.]')
xlabels = np.array(['Chair', 'Tear', 'RecTophat', 'Tophat', 'Ring', 'Gaussian'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['ResNet18','MLP12','ViT-Base','PhaseNet','Leonhard et al.','smNet'])
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos=result_recons
zpos = zpos.ravel()

dx=0.335
dy=0.335
dz=zpos

ax2.w_xaxis.set_ticks(xpos + dx/2)
ax2.w_xaxis.set_ticklabels(xlabels, ha='right')

ax2.w_yaxis.set_ticks(ypos + dy/2)
ax2.w_yaxis.set_ticklabels(ylabels, va='bottom')
ax2.tick_params(axis='y', which='major', pad=0)
ax2.tick_params(axis='x', which='major', pad=-5)
ax2.view_init(elev=26, azim=-52)
for tick in ax2.get_yticklabels():
    tick.set_horizontalalignment('left')

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.rainbow(values)
ax2.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
ax2.set_title('b. Reconstruction error', fontsize=20, fontweight='bold')

plt.savefig('detection_benchmarking.png', bbox_inches='tight')
'''

'''
ax1=plt.subplot(121, projection='3d')
result_wave = np.array(result_wave)
ax1.set_xlabel('Beam shapes', labelpad=13)
ax1.set_ylabel('Models', labelpad=13)
ax1.set_zlabel('Wavefront error [$\lambda$]')
xlabels = np.array(['Chair', 'Tear', 'RecTophat', 'Tophat', 'Ring', 'Gaussian'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['ResNet18','MLP12','ViT-Base','PhaseNet','Leonhard et al.','smNet'])
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos=result_wave
zpos = zpos.ravel()

dx=0.335
dy=0.335
dz=zpos

ax1.w_xaxis.set_ticks(xpos + dx/2)
ax1.w_xaxis.set_ticklabels(xlabels, ha='right')

ax1.w_yaxis.set_ticks(ypos + dy/2)
ax1.w_yaxis.set_ticklabels(ylabels, va='bottom')
ax1.tick_params(axis='y', which='major', pad=0)
ax1.tick_params(axis='x', which='major', pad=-5)
ax1.view_init(elev=26, azim=-52)
for tick in ax1.get_yticklabels():
    tick.set_horizontalalignment('left')

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.rainbow(values)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
ax1.set_title('a. Wavefront error', fontsize=20, fontweight='bold')

ax2=plt.subplot(122, projection='3d')
result_corr = np.array(result_corr)
ax2.set_xlabel('Beam shapes', labelpad=13)
ax2.set_ylabel('Models', labelpad=13)
ax2.set_zlabel('Correction error [a.u.]')
xlabels = np.array(['Chair', 'Tear', 'RecTophat', 'Tophat', 'Ring', 'Gaussian'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['ResNet18','MLP12','ViT-Base','PhaseNet','Leonhard et al.','smNet'])
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos=result_corr
zpos = zpos.ravel()

dx=0.335
dy=0.335
dz=zpos

ax2.w_xaxis.set_ticks(xpos + dx/2)
ax2.w_xaxis.set_ticklabels(xlabels, ha='right')

ax2.w_yaxis.set_ticks(ypos + dy/2)
ax2.w_yaxis.set_ticklabels(ylabels, va='bottom')
ax2.tick_params(axis='y', which='major', pad=0)
ax2.tick_params(axis='x', which='major', pad=-5)
ax2.view_init(elev=26, azim=-52)
for tick in ax2.get_yticklabels():
    tick.set_horizontalalignment('left')

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.rainbow(values)
ax2.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
ax2.set_title('b. Correction error', fontsize=20, fontweight='bold')

plt.savefig('correction_benchmarking.png', bbox_inches='tight')
'''
#benchmarking metrics comparison finishes