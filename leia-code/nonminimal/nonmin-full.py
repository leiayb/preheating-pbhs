import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.optimize import fmin
from scipy.optimize import root
from multiprocessing import Pool
from scipy.special import spherical_jn
import natural_units as nu


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['font.size'] = 27
# plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 1.7
# plt.rc('legend',fontsize=12, title_fontsize=14)
plt.rc('legend',fontsize=18, title_fontsize=24)
plt.rcParams['figure.figsize'] = (10, 7)  # default is (6.4, 4.8)



def V(phi, xi):  # units of Mpl, same as phi
    Vtilde = phi**4 / 4
    return Vtilde / (1 + xi * phi**2)**2


def d2V(phi, xi):
    return 3 * phi**2 * (1 - xi * phi**2) / (1 + xi * phi**2)**4


def dtdf(f, t, Phi, xi):
    Gll = (1 + xi * f**2 * (1 + 6*xi)) / (1 + xi * f**2)**2
    return np.sqrt(Gll) / (np.sqrt(2) * np.sqrt(V(Phi, xi) - V(f, xi)))


def field(gk, k, r, T, meff2=1):  # most important function, IFTing resonance bands
    ok = np.sqrt(meff2 + k**2)  # mode frequency omega_k
    vk = np.exp(gk * T) / np.sqrt(ok) / np.sqrt(2)  # mode function v_k

    nk = ok * vk**2 - 0.5  # -1/2 to regularize
    rhok = ok * nk
    r_j0 = simpson(spherical_jn(0, np.outer(k, r)) * k**2 * rhok * np.sqrt(2/np.pi) * k, np.log(k))

    plt.clf()
    plt.loglog(k, rhok, 'r')
    plt.ylabel(r'$\rho_{\chi k}/m_{\rm LVS}$')
    plt.xlabel(r'$k/m_{\rm LVS}$')
    plt.tight_layout() 
    plt.savefig(dirname+'rhok.png')

    return rhok, r_j0


def fvar(gk, meff2, k, T):
    ok = np.sqrt(meff2 + k**2)  # mode frequency omega_k
    vk = np.exp(gk * T) / np.sqrt(ok) / np.sqrt(2)  # mode function v_k
    vkr = vk - 1 / (np.sqrt(ok) * np.sqrt(2))  # regularized so that vacuum gives zero
    return simpson(4*np.pi * k**2 * vkr**2, k)


def Mt(dr, r):  # deltaM with tophat window window
    return cumulative_trapezoid(4*np.pi*r**2 * dr, r, initial=0)


def init_worker(interp_func, q, xi, xic):
    global _shared_state
    _shared_state['interpolation'] = interp_func
    _shared_state['q'] = q
    _shared_state['xi'] = xi
    _shared_state['xic'] = xic


def save_xls(mydict, xls_path):
    with pd.ExcelWriter(xls_path) as writer:
        for n, df in mydict.items():
            pd.DataFrame(df).to_excel(writer, n, index=False)


#FIXME
def mathieu(T, x, k):  # x = [x, dx/dt], returns dx/dt = [dx/dt, ddx/dtt]
    phi = _shared_state['interpolation'](T)
    q = _shared_state['q']
    xi = _shared_state['xi']
    xic = _shared_state['xic']
    drive = phi**2 * (q * (1 + xi*phi**2) - xic * phi**2) / (1 + xi*phi**2)**3
    Q = (k**2 + drive) * x[0]  # * tmax**2 for rescaling if wanted

    return [x[1], -Q]


def wrapper(inp):
    tmax, k = inp

    t = np.linspace(0, tmax, 1000)

    soln = solve_ivp(mathieu, [0, tmax], [1, 0], t_eval=t,
                     args=(k,))
    h = soln.y[0][-1]
    g = soln.y[1][-1]

    B = np.array([[h, (h/g)*h - g**-1], [g, h]])
    if k==0 and Phi==0:
        B = np.array([[1, 0], [0, 1]])

    return np.real(np.log(np.max(np.abs(np.linalg.eig(B)[0]))))


_shared_state = {}



if __name__=='__main__':

    Amax = 1.5
    N = 100
    m = 7
    ks = np.logspace(-3, 1, 2**m)
    Phis = [0.1, 1]
    mep = 1e-5

    lam = 1e-2
    xi = 1
    xic = 1

    folder = 'nonmin-figs'
    dirname = folder+'/'
    filename = "sheets/lam{}-xi{}-xi{}.xlsx".format(int(np.log10(lam)), xi, xic)

    sheet = {}
    sheet.update({'Phis': Phis})
    sheet.update({'ks': ks})
    meff2s = []
    tmaxs = []
    mus = np.zeros((np.size(ks), np.size(Phis)))

    ##### main run
    mapgen = True  #FIXME run again after derivatives correct
    if mapgen:
        for j, Phi in enumerate(Phis):

            print(Phi)

            tsol1 = solve_ivp(dtdf, [0, Phi], [0],
                            t_eval=np.linspace(0, Phi, int(1/mep)),
                            args=(Phi, xi))
            tq1 = tsol1.y[0]
            
            tmax = 4 * tq1[-1]
            y1 = tsol1.t

            t = np.concatenate((tq1[-1]-tq1[::-1], tq1[-1]+tq1[1:], 3*tq1[-1]-tq1[::-1][1:], 3*tq1[-1]+tq1[1:]))
            P = np.concatenate((y1[::-1], -y1[1:], -y1[::-1][1:], y1[1:]))

            # interpolation = interp1d(t, P, kind='linear', fill_value='extrapolate')
            interpolation = CubicSpline(t, P, bc_type='periodic', extrapolate='periodic')  # awesome that it has periodic option

            meff2s.append(d2V(np.average(interpolation(t)), xi))
            tmaxs.append(tmax)

            # ### plot potential and phiz

            # plt.clf()
            # plt.plot(np.concatenate((t, t+tmax)), np.concatenate((interpolation(t), interpolation(t))), 'r')
            # plt.xlabel(r'$\sqrt{\lambda{M_\mathrm{pl} t$')
            # plt.ylabel(r'$\varphi/M_\mathrm{pl}$')
            # plt.tight_layout()
            # plt.savefig(dirname+'phiz-{}.png'.format(Phi))

            # if Phi==Phis[-1]:
            #     plt.clf()
            #     plt.plot(np.linspace(-2*Phi, 2*Phi), V(np.linspace(-2*Phi, 2*Phi), lam, xi), 'r')
            #     plt.xlabel(r'$\varphi/M_\mathrm{pl}$')
            #     plt.ylabel(r'$V/M_\mathrm{pl}^4$')
            #     plt.title(r'$\lambda_\phi={},\ \xi_\phi={}$'.format(lam, xi))
            #     plt.tight_layout()
            #     plt.savefig(dirname+'potn.png')

        ####### map-making:
            x, y = np.meshgrid(tmax, ks)
            inputs = np.stack([x.ravel(), y.ravel()]).T
            # pool = Pool(processes=4)
            pool = Pool(processes=4, initializer=init_worker, initargs=(interpolation, lam**-0.5, xi, xic))
            # wrapper_partial = partial(wrapper, interp_func=interpolation)
            # result = list(pool.imap(wrapper_partial, inputs))
            result = list(pool.imap(wrapper, inputs))  # without partial
            mus[:, j] = result.copy()

        sheet.update({'mus': mus})
        sheet.update({'meff2s': meff2s})
        sheet.update({'tmaxs': tmaxs})

    save_xls(sheet, filename)

    print('mapgen done')


    ### plotting
    plotmap = False
    if plotmap:
        for divH in [True, False]:
            
            mus = np.squeeze(pd.read_excel(filename, sheet_name='mus').values)

            if divH:
                HTs = np.sqrt(V(Phis) / 3) * tmaxs
                mus = mus / HTs

            ks = np.squeeze(pd.read_excel(filename+'.xlsx', sheet_name='ks').values)
            Phis = np.squeeze(pd.read_excel(filename+'.xlsx', sheet_name='Phis').values)

            # map plot
            plt.clf()
            phmin = fmin(V, 0)[0]
            plot = plt.pcolormesh(ks, Phis, mus.T, cmap='inferno', shading='auto', alpha=1,
                                    # vmin=0, vmax=5
                                    )
            plot.set_edgecolor('face')
            clb = plt.colorbar(plot)
            if divH:
                clb.ax.set_title(r'$\gamma/H$')
            else:
                clb.ax.set_title(r'$\gamma T$')
            # plt.yticks([0, 0.25, 0.5, 0.75])
            plt.xlabel(r'$k/\sqrt{\lambda}M_\mathrm{pl}$')
            plt.ylabel(r'$\Phi/M_\mathrm{pl}$')
            plt.tight_layout()
            if divH:
                plt.savefig(dirname+'map-H.png')
            else:
                plt.savefig(dirname+'map.png')
            print('plotmap done')


    ## plotting enclosed masses
    plotmass = False
    if plotmass:
        plt.clf()
        mus = np.squeeze(pd.read_excel(filename, sheet_name='mus').values)
        ks = np.squeeze(pd.read_excel(filename, sheet_name='ks').values)
        rs = 1 / ks[::-1]  # radius coordinate values; high k means higher resolution (lower min r) and vice versa
        Phis = np.squeeze(pd.read_excel(filename, sheet_name='Phis').values)

        for i, Phi in enumerate(Phis):  #FIXME omitting Hubble radius until I have better rhor

            H = np.sqrt(5/3) * V(Phi, xi) 
        
            lMpl = 0

            # optionally, convert axes to non-natural units
            rp = rs.copy()  # r for plotting, in case we change up units here:
            massmult = 1
            cm = False
            gram = False
            if cm:
                rp *= 10**lMpl / nu.Mpl / nu.m * 100
            if gram:
                massmult = 10**(-lMpl) * nu.Mpl / nu.g

            T = 0
            while varc < V(Phi) / meff2s[i]:
                T += 1  # step BEFORE recalculating varchi
                varc = fvar(mus[:, i], meff2s[i], ks, T)
            T -= 1
            print('T =', T)
            _, rj0 = field(mus[:, i], ks, rs, T)
            plt.clf()
            plt.loglog(rs, rj0, 'r')
            plt.ylabel(r'$\rho_\chi(r)/m_{\rm LVS}^4$')
            plt.xlabel(r'$m_{\rm LVS} r$')
            plt.tight_layout()
            plt.savefig(dirname+'rhos.png') 
            print(max(rj0 / (V(Phi))))  # want some >1 for oscillons to form: 
            #FIXME get <<1 here, doesn't agree with oscillons paper?

            if i==0:
                plt.loglog(rp, Mt(rj0, rs) * massmult, label=r'$M_\mathrm{enclosed}$ of $\phi$')
                # plot black hole mass vs radius to compare to phi mass:
                plt.loglog(rp, 10**(2*lMpl) * rs/2 * massmult, ':b', label=r'$M_\mathrm{black\ hole}$')
                if cm:  # plot Hubble radius
                    plt.plot([10**lMpl / nu.Mpl / nu.m * 100/H, 10**lMpl / nu.Mpl / nu.m * 100/H], plt.ylim(), 'k')
                else:
                    plt.plot([1/H, 1/H], plt.ylim(), 'k', label=r'$Hubble radii')
            else:
                plt.loglog(rp, Mt(rj0, rs) * massmult)
                if cm:  # plot Hubble radius
                    plt.plot([10**lMpl / nu.Mpl / nu.m * 100/H, 10**lMpl / nu.Mpl / nu.m * 100/H], plt.ylim(), 'k', label=r'Hubble radii')
                else:
                    plt.plot([1/H, 1/H], plt.ylim(), 'k')

        if gram:
            plt.ylabel(r'mass [g]')
        else:
            plt.ylabel(r'$M/m_{\rm LVS}$')
        if cm:
            plt.xlabel(r'radius [cm]')
        else:
            plt.xlabel(r'$m_{\rm LVS} r$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(dirname+'enclosed-mass.png')


        
    print('done')