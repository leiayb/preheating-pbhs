import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import ellipj
from scipy.special import ellipk
from multiprocessing import Pool


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 1.7
plt.rc('legend',fontsize=12, title_fontsize=14)
plt.rcParams['figure.figsize'] = (10, 7)  # default is (6.4, 4.8)


def save_xls(mydict, xls_path):
    with pd.ExcelWriter(xls_path) as writer:
        for n, df in mydict.items():
            pd.DataFrame(df).to_excel(writer, n, index=False)


# def mathieu(t, x, A, Q):  # periodic function for one dimension
#     P = A + 2*Q * np.cos(2*t)
#     return [x[1], -P * x[0]]

def mathieu(t, x, A, Q):  # periodic function for one dimension
    _, cn, _, _ = ellipj(t, 0.5)
    P = A + Q * cn**2
    return [x[1], -P * x[0]]


def find_mu(inp):  # finding mu for one dimensionQ
    k, Q = inp

    t = np.linspace(0, tmax, 1000)

    soln = solve_ivp(mathieu, [0, tmax], [1, 0], t_eval=t,
                     args=(k**2, Q))
    h = soln.y[0][-1]
    g = soln.y[1][-1]

    B = np.array([[h, (h/g)*h - g**-1], [g, h]])
    if k==0 and Q==0:
        B = np.array([[1, 0], [0, 1]])

    return np.log(np.max(np.abs(np.linalg.eig(B)[0])))


# tmax = np.pi
tmax = 4 * ellipk(0.5)
# print(tmax)
# exit()

m = 7
k = np.logspace(-3, 4, 2**m)
# qs = [4.9e6]
# qs = [1e2]
# qs = np.logspace(2, 7)
qs = [1e7]

if __name__=='__main__':

    mus = np.zeros((np.size(qs), np.size(k)))

    sheet = {}
    sheet.update({'qs': qs})
    sheet.update({'k': k})

    # for i, q in enumerate(qs):
    #     for j, kk in enumerate(k):
    #         mus[i, j] = find_mu(kk**2, q)

    x, y = np.meshgrid(k, qs)
    inputs = np.stack([x.ravel(), y.ravel()]).T
    pool = Pool(processes=4)
    result = list(pool.imap(find_mu, inputs))
    mus = np.reshape(result, (np.size(qs), np.size(k)))
    sheet.update({'gs': mus})

    save_xls(sheet, 'qe7/qe7.xlsx')