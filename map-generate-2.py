import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import ellipj
from scipy.special import ellipk


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
# plt.rcParams['font.size'] = 18
plt.rcParams['font.size'] = 27
plt.rcParams['lines.linewidth'] = 1.7
plt.rc('legend',fontsize=12, title_fontsize=14)
plt.rcParams['figure.figsize'] = (10, 7)  # default is (6.4, 4.8)


def save_xls(mydict, xls_path):
    with pd.ExcelWriter(xls_path) as writer:
        for n, df in mydict.items():
            pd.DataFrame(df).to_excel(writer, n, index=False)


def mathieu(t, x, A, Q):  # periodic function for one dimension
    _, cn, _, _ = ellipj(t, 0.5)
    P = A + Q * cn**2
    return [x[1], -P * x[0]]


def find_gamma(A, Q):  # finding mu for one dimensionQ

    t = np.linspace(0, tmax, 1000)

    soln = solve_ivp(mathieu, [0, tmax], [1, 0], t_eval=t,
                     args=(A, Q))
    h = soln.y[0][-1]
    g = soln.y[1][-1]

    B = np.array([[h, (h/g)*h - g**-1], [g, h]])
    if A==0 and Q==0:
        B = np.array([[1, 0], [0, 1]])

    return np.log(np.max(np.abs(np.linalg.eig(B)[0])))


tmax = 4 * ellipk(0.5)

# # for PBHs
# m = 7
# k = np.logspace(-4, 2, 2**m)
# qs = [4.9e6]


N = 100
# for reproducing Fig. 4 in that '97 paper
As = np.linspace(0, 10, N)  # put in a range for A=kappa^2 here
Qs = np.linspace(0, 100, N+1)  # range for q=g^2/lam
# and then might need to change some variable names below

# ## finding growth rates
# mus = np.zeros((np.size(Qs), np.size(As)))

# sheet = {}
# sheet.update({'Qs': Qs})
# sheet.update({'As': As})

# for i, q in enumerate(Qs):
#     for j, A in enumerate(As):
#         mus[i, j] = find_gamma(A, q)
# sheet.update({'gs': mus})

# save_xls(sheet, 'g2/g2-map.xlsx')



## plotting growth rates
mus = np.squeeze(pd.read_excel('g2/g2-map.xlsx', sheet_name='gs').values)
As = np.squeeze(pd.read_excel('g2/g2-map.xlsx', sheet_name='As').values)
Qs = np.squeeze(pd.read_excel('g2/g2-map.xlsx', sheet_name='Qs').values)

plt.clf()
plot = plt.pcolormesh(Qs, As, mus.T / tmax, cmap='inferno', shading='auto', alpha=1)
plot.set_edgecolor('face')
clb = plt.colorbar(plot)
clb.ax.set_title(r'$\mathcal{C}$')
plt.ylabel(r'$A$')
plt.xlabel(r'$q$')
plt.tight_layout()
plt.savefig('g2/g2-map.png')