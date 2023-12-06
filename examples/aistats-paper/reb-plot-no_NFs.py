import matplotlib.pyplot as plt
import numpy as np

plt.rc('text',usetex=True)
plt.rc('font', family='serif')

# import npz results# reb-6d-80k-gCorr-kstar-simple_align.npz
results = np.load("results_backup/6d-40k-simple_align-3rep-gCorr-CELLINA.npz")
results_nf = np.load("results_backup/reb-norm_flows.npz")

from matplotlib.cm import get_cmap

# Import color maps
set1 = get_cmap('Set1').colors
set2 = get_cmap('Set2').colors
set3 = get_cmap('Set3').colors
dark2 = get_cmap('Dark2').colors
tab10 = get_cmap('tab10').colors
tab20 = get_cmap('tab20').colors
tab20b = get_cmap('tab20b').colors
tab20c = get_cmap('tab20c').colors

# # Combine to define custom color maps
# custom = tab20c[1::4][:4]+tab20b[13:14]+tab20b[0:1]
# custom2 = dark2[0:1]+tab20[5:6]+set2[5:6]+set3[3:4]+tab20b[14:15]+set1[1:2]+dark2[7:8]
# custom3 = dark2[0:1]+tab20[5:6]+set2[5:6]+set3[3:4]+tab20b[17:18]+set1[1:2]+dark2[7:8]
# custom4 = dark2[0:1]+set2[5:6]+set3[3:4]+tab20b[17:18]+set1[1:2]+tab20b[5:6]+dark2[7:8]
# custom5 = dark2[0:1]+dark2[5:6]+set3[3:4]+tab20b[17:18]+set1[1:2]+tab20b[5:6]+dark2[7:8]
# custom6 = dark2[0:1]+dark2[5:6]+set3[3:4]+tab20b[17:18]+set1[1:2]+tab20b[5:6]+tab20b[12:13]
# custom7 = set1[1:2]+dark2[5:6]+dark2[0:1]+set3[3:4]+tab20b[17:18]+tab20b[5:6]+tab20b[12:13]
# custom8 = set1[1:2]+dark2[5:6]+dark2[0:1]+set3[3:4]+tab20b[17:18]+tab20[19:20]+tab20b[12:13]
custom9 = tab10[0:1]+dark2[5:6]+dark2[0:1]+set3[3:4]+tab20b[17:18]+tab20[18:19]+tab20b[12:13]

mycmap = {}
mycmap['BMTI'] = custom9[0]
mycmap['PAk'] = custom9[1]
mycmap['kNN'] = custom9[2]
mycmap['GKDE'] = custom9[3]
mycmap['aKDE'] = custom9[4]
mycmap['GMM'] = custom9[5]
mycmap['kstarNN'] = custom9[6]
mycmap['NF'] = "0.65"

# # Test colours
# plt.figure()
# plt.plot(np.linspace(-1,1,100), c=mycmap['BMTI'])
# plt.plot(np.linspace(-1,1,100)+1,c=mycmap['PAk'])
# plt.plot(np.linspace(-1,1,100)+2,c=mycmap['kNN'])
# plt.plot(np.linspace(-1,1,100)+3,c=mycmap['GKDE'])
# plt.plot(np.linspace(-1,1,100)+4,c=mycmap['aKDE'])
# plt.plot(np.linspace(-1,1,100)+5,c=mycmap['GMM'])
# plt.plot(np.linspace(-1,1,100)+6,c=mycmap['kstarNN'])
# plt.plot(np.linspace(-1,1,100)+7,c=mycmap['NF'])
# plt.show()




#titlestring="6d-80k-gCorr"
titlestring="6d potential"
savestring="time_scaling-6d-40k"

def plot_times(results,noNFs=False):
    plt.figure()
    
    r = results['MAE_kNN_Abr'].shape[0]
    print("r is ", r)
    xs = results['Nsample']

    m = np.mean(results['time_BMTI'],axis=0)
    s = np.std(results['time_BMTI'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, "-o", lw=3, label=r"\textbf{BMTI}", color=mycmap['BMTI'])
    plt.fill_between(xs, m-s, m+s, alpha=0.5, color=mycmap['BMTI'])

    m = np.mean(results['time_PAk'],axis=0)
    s = np.std(results['time_PAk'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, "-o", lw=3, label=r"\textbf{PAk}", color=mycmap['PAk'])
    plt.fill_between(xs, m-s, m+s, alpha=0.5, color=mycmap['PAk'])

    m = np.mean(results['time_kNN_Abr'],axis=0)
    s = np.std(results['time_kNN_Abr'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, "-o", lw=3, label=r"\textbf{kNN}", color=mycmap['kNN'])
    plt.fill_between(xs, m-s, m+s, alpha=0.5, color=mycmap['kNN'])

    m = np.mean(results['time_GKDE_Scott'],axis=0)
    s = np.std(results['time_GKDE_Scott'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, "-o", lw=3, label=r"\textbf{GKDE}", color=mycmap['GKDE'])
    plt.fill_between(xs, m-s, m+s, alpha=0.5, color=mycmap['GKDE'])

    m = np.mean(results['time_awkde'],axis=0)
    s = np.std(results['time_awkde'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, "-o", lw=3, label=r"\textbf{aKDE}", color=mycmap['aKDE'])
    plt.fill_between(xs, m-s, m+s, alpha=0.5, color=mycmap['aKDE'])

    m = np.mean(results['time_GMM'],axis=0)
    s = np.std(results['time_GMM'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, "-o", lw=3, label=r"\textbf{GMM}", color=mycmap['GMM'])
    plt.fill_between(xs, m-s, m+s, alpha=0.5, color=mycmap['GMM'])

    if noNFs is False:
        m = np.mean(results_nf['time_NF'],axis=0)[:-1]
        s = np.std(results_nf['time_NF'],axis=0)[:-1]/np.sqrt(r)
        plt.plot(results_nf['Nsample'][:-1], m, "-o", lw=3, label=r"\textbf{NF}", color=mycmap['NF'])
        plt.fill_between(results_nf['Nsample'][:-1], m-s, m+s, alpha=0.5, color=mycmap['NF'])

    plt.xscale("log")
    plt.yscale("log")
    #plt.ylim(0.1, 1.32);
    plt.xlim(120, 50000);
    plt.legend(loc=(0.08,0.37),fontsize=16)
    plt.xlabel(r"\textbf{Sample size  -  N}",fontsize = 16)
    plt.ylabel(r"\textbf{Time (s)}",fontsize = 16)
    plt.yticks( fontsize = 16)
    plt.xticks( fontsize = 16)
    plt.title(r"\textbf{Time scaling - 6d potential}", fontsize=18)
    plt.tight_layout()
    plt.savefig("plots/reb-times-{}.png".format(savestring),dpi=300)
    plt.show()


def plot_MAEs(results,noNFs=False):
    plt.figure()
    xs = results['Nsample']
    r = results['MAE_kNN_Abr'].shape[0]

    m = np.mean(results['MAE_kNN_Abr'],axis=0)
    s = np.std(results['MAE_kNN_Abr'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kNN-Abr")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MAE_kNN_Zhao'],axis=0)
    s = np.std(results['MAE_kNN_Zhao'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kNN-Zhao")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MAE_kstarNN'],axis=0)
    s = np.std(results['MAE_kstarNN'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kstarNN")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MAE_GKDE_Sil'],axis=0)
    s = np.std(results['MAE_GKDE_Sil'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="GKDE-Sil")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MAE_PAk'],axis=0)
    s = np.std(results['MAE_PAk'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="PAk")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MAE_BMTI'],axis=0)
    s = np.std(results['MAE_BMTI'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="BMTI")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MAE_GMM'],axis=0)
    s = np.std(results['MAE_GMM'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="GMM")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MAE_GKDE_Scott'],axis=0)
    s = np.std(results['MAE_GKDE_Scott'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="GKDE-Scott")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    if noNFs is False:
        xs_nf = results_nf['Nsample']
        m = np.mean(results_nf['MAE_NF'],axis=0)
        s = np.std(results_nf['MAE_NF'],axis=0)/np.sqrt(r)
        plt.plot(xs_nf, m, label="NF")
        plt.fill_between(xs_nf, m-s, m+s, alpha=0.5)

    plt.xscale("log")
    plt.legend()
    plt.xlabel("Nsample")
    plt.ylabel("MAE")
    plt.title(titlestring)
    plt.savefig("plots/reb-MAE-{}.png".format(savestring))
    plt.show()


def plot_MSEs(results,noNFs=False):
    plt.figure()
    xs = results['Nsample']
    r = results['MAE_kNN_Abr'].shape[0]

    m = np.mean(results['MSE_kNN_Abr'],axis=0)
    s = np.std(results['MSE_kNN_Abr'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kNN-Abr")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MSE_kNN_Zhao'],axis=0)
    s = np.std(results['MSE_kNN_Zhao'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kNN-Zhao")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MSE_kstarNN'],axis=0)
    s = np.std(results['MSE_kstarNN'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kstarNN")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MSE_GKDE_Sil'],axis=0)
    s = np.std(results['MSE_GKDE_Sil'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="GKDE-Sil")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MSE_PAk'],axis=0)
    s = np.std(results['MSE_PAk'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="PAk")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['MSE_BMTI'],axis=0)
    s = np.std(results['MSE_BMTI'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="BMTI")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)

    m = np.mean(results['MSE_GMM'],axis=0)
    s = np.std(results['MSE_GMM'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="GMM")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    
    if noNFs is False:
        xs_nf = results_nf['Nsample']    
        m = np.mean(results_nf['MSE_NF'],axis=0)
        s = np.std(results_nf['MSE_NF'],axis=0)/np.sqrt(r)
        plt.plot(xs_nf, m, label="NF")
        plt.fill_between(xs_nf, m-s, m+s, alpha=0.5)

    plt.xscale("log")
    plt.legend()
    plt.xlabel("Nsample")
    plt.ylabel("MSE")
    plt.title(titlestring)
    plt.savefig("plots/reb-MSE-{}.png".format(savestring))
    plt.show()


def plot_nf_MAE(results_nf):
    plt.figure()
    xs_nf = results_nf['Nsample']
    r = results_nf['MAE_NF'].shape[0]
    m = np.mean(results_nf['MAE_NF'],axis=0)
    s = np.std(results_nf['MAE_NF'],axis=0)/np.sqrt(r)
    plt.plot(xs_nf, m, label="NF")
    plt.fill_between(xs_nf, m-s, m+s, alpha=0.5)
    m = np.mean(results_nf['MAE_NF_noreg'],axis=0)
    s = np.std(results_nf['MAE_NF_noreg'],axis=0)/np.sqrt(r)
    plt.plot(xs_nf, m, label="NF_noreg")
    plt.fill_between(xs_nf, m-s, m+s, alpha=0.5)

    plt.xscale("log")
    plt.legend()
    plt.xlabel("Nsample")
    plt.ylabel("MAE")
    plt.title(titlestring)
    plt.savefig("plots/reb-nf-MAE-{}.png".format(savestring))
    plt.show()


def plot_KL(results):
    plt.figure()
    xs = results['Nsample']
    r = results['MAE_kNN_Abr'].shape[0]

    m = np.mean(results['KLD_kNN_Abr'],axis=0)
    s = np.std(results['KLD_kNN_Abr'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kNN-Abr")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['KLD_kNN_Zhao'],axis=0)
    s = np.std(results['KLD_kNN_Zhao'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kNN-Zhao")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['KLD_kstarNN'],axis=0)
    s = np.std(results['KLD_kstarNN'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kstarNN")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['KLD_GKDE_Sil'],axis=0)
    s = np.std(results['KLD_GKDE_Sil'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="GKDE-Sil")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['KLD_PAk'],axis=0)
    s = np.std(results['KLD_PAk'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="PAk")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['KLD_BMTI'],axis=0)
    s = np.std(results['KLD_BMTI'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="BMTI")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['KLD_GMM'],axis=0)
    s = np.std(results['KLD_GMM'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="GMM")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    
    m = np.mean(results['KLD_GKDE_Scott'],axis=0)
    s = np.std(results['KLD_GKDE_Scott'],axis=0)/np.sqrt(r)
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Nsample")
    plt.ylabel("KL")
    
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Nsample")
    plt.ylabel("KLD")
    plt.title(titlestring)
    plt.savefig("plots/reb-KLD-{}.png".format(savestring))
    plt.show()


#plot_nf_MAE(results_nf)

plot_times(results,noNFs=False)

#plot_KL(results)

plot_MAEs(results,noNFs=False)

plot_MSEs(results,noNFs=False)

