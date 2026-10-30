import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy import fft
from scipy.integrate import simpson
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import quad
import natural_units as nu
from scipy.special import ellipj
from scipy.special import ellipk
from scipy.special import ellipe
from scipy.special import spherical_jn


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['font.size'] = 27
# plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 1.7
# plt.rc('legend',fontsize=12, title_fontsize=14)
plt.rc('legend',fontsize=24)#, title_fontsize=14)
plt.rcParams['figure.figsize'] = (10, 7)  # default is (6.4, 4.8)


cn2 = (-0.5 * ellipk(0.5) + ellipe(0.5)) / 0.5 / ellipk(0.5)  # average value of cn^2 over period
def meff2(q):  # could put 3 in place of q for meff2 of phi
    return q * cn2


def Wg(y, R):  # Gaussian window function
    return np.exp(-y**2 / R**2)


#FIXME check if this is implemented correctly, R vs r etc
def Mg(dr, r):  # deltaM with Gaussian window
    return np.array([0]+[simpson(4*np.pi*r[:i+1]**2 * dr[:i+1] * Wg(r[:i+1], r[i+1]), r[:i+1]) for i in range(len(r)-1)])


def Mt(dr, r):  # deltaM with tophat window window
    return cumulative_trapezoid(4*np.pi*r**2 * dr, r, initial=0)


def field(ck, k, r, N):  # most important function, IFTing resonance bands
    ok = np.sqrt(meff2(q) + k**2)  # mode frequency omega_k
    vk = np.exp(ck * N) / np.sqrt(ok) / np.sqrt(2)  # mode function v_k
    fk = (vk - 1 / np.sqrt(ok) / np.sqrt(2))* k**1.5

    f_j0 = simpson(spherical_jn(0, np.outer(k, r)) * k**2 * fk * np.sqrt(2/np.pi) * k, np.log(k))

    nk = ok * vk**2 - 0.5  # -1/2 to regularize
    rhok = ok * nk
    r_j0 = simpson(spherical_jn(0, np.outer(k, r)) * k**2 * rhok * np.sqrt(2/np.pi) * k, np.log(k))

    # plotting Fourier-space spectra of field number and energy densities
    plt.clf()
    plt.loglog(k, nk, 'r')
    plt.ylabel(r'$n_k$')
    plt.xlabel(r'$k/\sqrt{\lambda}\Phi$')
    plt.tight_layout() 
    plt.savefig(dirname+'nk.png')

    plt.clf()
    plt.loglog(k, rhok, 'r')
    plt.ylabel(r'$\rho_{\chi k}/\sqrt{\lambda}\Phi$')
    plt.xlabel(r'$k/\sqrt{\lambda}\Phi$')
    plt.tight_layout() 
    plt.savefig(dirname+'rhok.png')

    return r_j0, f_j0


def fvar(ck, q, k, N):
    ok = np.sqrt(meff2(q) + k**2)  # mode frequency omega_k
    vk = np.exp(ck * N) / np.sqrt(ok) / np.sqrt(2)  # mode function v_k
    vkr = vk - 1 / (np.sqrt(ok) * np.sqrt(2))  # regularized so that vacuum gives zero
    return simpson(4*np.pi * k**2 * vkr**2, k) / (2*np.pi)**3

dirname = 'qe7/'
filename = dirname+'qe7'

k = np.squeeze(pd.read_excel(filename+'.xlsx', sheet_name='k').values)
r = 1 / k[::-1]  # radius coordinate values; high k means higher resolution (lower min r) and vice versa

q = np.squeeze(pd.read_excel(filename+'.xlsx', sheet_name='qs').values)
ck = np.squeeze(pd.read_excel(filename+'.xlsx', sheet_name='gs').values)

## find N
lam = 1e-14
g2 = q * lam
varc = 0
N = 0
dN = 1
digits = 2
while dN > 10**(-digits-1):
    while varc < cn2 / g2:
        N += dN  # step before recalculating new varchi
        varc = fvar(ck, q, k, N)
    N -= dN  # undo overshoot from last step
    varc = fvar(ck, q, k, N)
    dN /= 10  # increase precision
print('N =',round(N, digits))


# ### nnl scan to pick q's
# filename = 'mus'

# g2s = np.linspace(-12, -7)
# qs = np.squeeze(pd.read_excel(filename+'.xlsx', sheet_name='qs').values)
# ck = np.squeeze(pd.read_excel(filename+'.xlsx', sheet_name='gs').values)
# vars = np.zeros((np.size(qs), np.size(g2s)))

# for i, log10g2 in enumerate(g2s):
#     for j, q in enumerate(qs):

#         # log10g2 = -7  # choose g^2 by choosing its log10
#         # N = (np.log10(cn2) - log10g2) / np.log10(fvar(ck[j, :], q, k, 1))
#         vars[j, i] = np.log10(fvar(ck[j, :], q, k, 1))


# plt.clf()
# plot = plt.pcolormesh(np.log10(qs), g2s, vars.T, cmap='inferno', shading='auto', alpha=1,
#                     #   vmin=0, vmax=15
#                      )
# plot.set_edgecolor('face')
# clb = plt.colorbar(plot)
# plt.ylabel(r'$\log_{10}(g^2)$')
# plt.xlabel(r'$\log_{10}q$')
# plt.tight_layout()
# plt.savefig(dirname+'var-map.png')

# plt.clf()
# plot = plt.pcolormesh(np.log10(qs)[np.where(np.log10(qs)>3)], g2s, vars[np.where(np.log10(qs)>3)][:].T,
#                       cmap='inferno', shading='auto', alpha=1,
#                     #   vmin=0, vmax=15
#                      )
# plot.set_edgecolor('face')
# clb = plt.colorbar(plot)
# plt.xlim([2-0.05, 7+0.05])
# plt.ylabel(r'$\log_{10}(g^2)$')
# plt.xlabel(r'$\log_{10}q$')
# plt.tight_layout()
# plt.savefig(dirname+'var-map3.png')

# plt.clf()
# plot = plt.pcolormesh(np.log10(qs)[np.where(np.log10(qs)>5)], g2s, vars[np.where(np.log10(qs)>5)][:].T,
#                       cmap='inferno', shading='auto', alpha=1,
#                     #   vmin=0, vmax=15
#                      )
# plot.set_edgecolor('face')
# clb = plt.colorbar(plot)
# plt.xlim([2-0.05, 7+0.05])
# plt.ylabel(r'$\log_{10}(g^2)$')
# plt.xlabel(r'$\log_{10}q$')
# plt.tight_layout()
# plt.savefig(dirname+'var-map5.png')

# exit()


# set up field spectra
fk_vac = (meff2(q) + k**2)**-0.25 / k**1.5 / np.sqrt(2)  # field modes for vacuum
fk_res = fk_vac.copy()
fk_res = fk_res * np.exp(ck * N)  # field modes that have been resonantly amplified after one period

# plot the field spectra fk
plt.clf()
plt.loglog(k, fk_vac, 'k', label='vacuum')
plt.loglog(k, fk_res, 'r', label='resonated')
plt.legend()
plt.ylabel(r'$\lambda\Phi^2|\chi_k|$')
plt.xlabel(r'$k/\sqrt{\lambda}\Phi$')
plt.tight_layout() 
plt.savefig(dirname+'spectra.png')

# find the IFTs of the field modes to get their spatial profile
r_vac, _ = field(0, k, r, N)
r_res, f_res = field(ck, k, r, N)
r_res[r>r[np.argmin(r_res)]] = 10*r_res[np.argmin(r_res)] # negative number to fall away in mass calc
# above 10 is somewhat arbitrary factor to get enclosed mass to fall off at zero-crossing of rho

# # plot field profiles
# plt.clf()
# # plt.loglog(r, f_vac, 'k', label='vacuum')
# plt.loglog(r, f_res, 'r', label='resonated')
# plt.ylabel(r'$\chi(r)/\sqrt{\lambda}\Phi$')
# plt.xlabel(r'$\sqrt{\lambda}\Phi r$')
# # plt.legend()
# plt.tight_layout()
# plt.savefig(dirname+'fields.png')

# # exit()


Phi = 6**.25
H = Phi / np.sqrt(12)

# plot energy profiles
plt.clf()
# plt.loglog(r, r_vac, 'k', label='vacuum')
plt.loglog(r, r_res, 'r')
# plt.semilogx(r, ri_res, '--', label=r'IFT')
plt.ylabel(r'$\rho_\chi(r)/\lambda^2\Phi^4$')
plt.xlabel(r'$\sqrt{\lambda}\Phi r$')
plt.plot([1/H, 1/H], plt.ylim(), 'k', label=r'Hubble radius with $\Phi/M_\mathrm{{pl}}={}$'.format({round(Phi, 2)}))
# plt.plot([1.032, 1.032], plt.ylim())
# plt.legend()
plt.tight_layout()
plt.savefig(dirname+'rhos.png') 


### rest of code is finding mass enclosed within r and comparing to black hole mass with radius r

plt.clf()
lMpl = -np.log10(Phi) + 7  # assuming lambda=10^-14

# optionally, convert axes to non-natural units
rp = r.copy()  # r for plotting, in case we change up units here:
massmult = 1
cm = False
gram = False
if cm:
    rp *= 10**lMpl / nu.Mpl / nu.m * 100
if gram:
    massmult = 10**(-lMpl) * nu.Mpl / nu.g
if lMpl==0:
    string = r'$\sqrt{\lambda}\Phi/M_\mathrm{{pl}}=1$'
else:
    string = r'$\sqrt{{\lambda}}\Phi/M_\mathrm{{pl}}=10^{}$'.format({-round(lMpl, 2)})

# plot enclosed mass
plt.loglog(rp, Mt(r_res, r) * massmult, 'r', label=r'$M_\chi^\mathrm{encl}$')
gauss = False  # if we only want to look at tophat, we can set this to False (turns out not to make huge difference)
if gauss:
    plt.plot(rp, Mg(r_res, r) * massmult, '--r', alpha=0.7)

# plot Hubble radii
if cm:
    plt.plot([10**lMpl / nu.Mpl / nu.m * 100/H, 10**lMpl / nu.Mpl / nu.m * 100/H], plt.ylim(),
             'k', label=r'$R_H$ with $\Phi/M_\mathrm{{pl}}={}$'.format({round(Phi, 2)}))
else:
    plt.plot([1/H, 1/H], plt.ylim(), 'k', label=r'Hubble radius with $\Phi/M_\mathrm{{pl}}={}$'.format({round(Phi, 2)}))
            
# plot black hole mass vs radius to compare to phi mass:
plt.loglog(rp, 10**(2*lMpl) * r/2 * massmult, ':b', label=r'$M_\mathrm{{BH}}$ with $\sqrt{{\lambda}}\Phi/M_{{\rm pl}}=10^{}$'.format({-round(lMpl, 2)}))

if gram:
    plt.ylabel(r'mass [g] with '+string)
else:
    plt.ylabel(r'$M/\sqrt{\lambda}\Phi$')
if cm:
    plt.xlabel(r'radius [cm] with '+string)
else:
    plt.xlabel(r'$\sqrt{\lambda}\Phi r$')
plt.legend()
plt.tight_layout()
plt.savefig(dirname+'masses.png')


