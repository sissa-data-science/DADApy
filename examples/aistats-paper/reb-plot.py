import matplotlib.pyplot as plt
import numpy as np

# import npz results# reb-6d-80k-gCorr-kstar-simple_align.npz
results = np.load("results/data.npz")
results_nf = np.load("results/reb-norm_flows_1.npz")

def plot_times(results):
    plt.figure()
    xs = results['Nsample']
    r = results['MAE_kNN_Abr'].shape[0]
    print("r is ", r)
    m = np.mean(results['time_kNN_Abr'],axis=0)
    s = np.std(results['time_kNN_Abr'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kNN-Abr")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    m = np.mean(results['time_kNN_Zhao'],axis=0)
    s = np.std(results['time_kNN_Zhao'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kNN-Zhao")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    m = np.mean(results['time_kstarNN'],axis=0)
    s = np.std(results['time_kstarNN'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kstarNN")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    m = np.mean(results['time_GKDE_Sil'],axis=0)
    s = np.std(results['time_GKDE_Sil'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="GKDE-Sil")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    m = np.mean(results['time_PAk'],axis=0)
    s = np.std(results['time_PAk'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="PAk")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    m = np.mean(results['time_BMTI'],axis=0)
    s = np.std(results['time_BMTI'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="BMTI")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    m = np.mean(results['time_GMM'],axis=0)
    s = np.std(results['time_GMM'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="GMM")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    m = np.mean(results['time_GKDE_Scott'],axis=0)
    s = np.std(results['time_GKDE_Scott'],axis=0)/np.sqrt(r)
    plt.plot(xs, m, label="kde")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    xs_nf = results_nf['Nsample']
    m = np.mean(results_nf['time_NF'],axis=0)
    s = np.std(results_nf['time_NF'],axis=0)/np.sqrt(r)
    plt.plot(xs_nf, m, label="NF")
    plt.fill_between(xs_nf, m-s, m+s, alpha=0.5)

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Nsample")
    plt.ylabel("time (s)")
    plt.title("6d-80k-gCorr")
    plt.savefig("plots/reb-times-6d-80k-gCorr.png")
    plt.show()


def plot_MAEs(results):
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
    plt.plot(xs, m, label="kde")
    plt.fill_between(xs, m-s, m+s, alpha=0.5)
    xs_nf = results_nf['Nsample']
    m = np.mean(results_nf['MAE_NF'],axis=0)
    s = np.std(results_nf['MAE_NF'],axis=0)/np.sqrt(r)
    plt.plot(xs_nf, m, label="NF")
    plt.fill_between(xs_nf, m-s, m+s, alpha=0.5)


    plt.xscale("log")
    plt.legend()
    plt.xlabel("Nsample")
    plt.ylabel("MAE")
    plt.title("6d-80k-gCorr")
    plt.savefig("plots/reb-MAE-6d-80k-gCorr.png")
    plt.show()


def plot_MSEs(results):
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
    xs_nf = results_nf['Nsample']
    m = np.mean(results_nf['MSE_NF'],axis=0)
    s = np.std(results_nf['MSE_NF'],axis=0)/np.sqrt(r)
    plt.plot(xs_nf, m, label="NF")
    plt.fill_between(xs_nf, m-s, m+s, alpha=0.5)


    plt.xscale("log")
    plt.legend()
    plt.xlabel("Nsample")
    plt.ylabel("MSE")
    plt.title("6d-80k-gCorr")
    plt.savefig("plots/reb-MSE-6d-80k-gCorr.png")
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
    plt.title("6d-80k-gCorr")
    plt.savefig("plots/reb-nf-MAE-6d-80k-gCorr.png")
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
    
    plt.show()


#plot_nf_MAE(results_nf)

plot_times(results)

plot_KL(results)

plot_MAEs(results)

plot_MSEs(results)

