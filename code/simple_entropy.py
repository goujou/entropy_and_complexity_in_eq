"""Simple entropy examples plots."""


import matplotlib.pyplot as plt
import numpy as np


def xlogx(x, base=None):
    """Make sure ``0*log(0)=0``."""
    if x == 0:
        return 0
    if base == 2:
        return x * np.log2(x)
    if base is None:
        return x * np.log(x)


if __name__ == "__main__":
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(21, 6), ncols=3)

    # Bernoulli plot
    x1 = np.linspace(0, 1, 101)
    y1 = [-xlogx(x, 2) - xlogx(1 - x, 2) for x in x1]
    ax1.plot(x1, y1, c="black", ls="-", lw=4)

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel(r"$p$")
    ax1.set_ylabel(r"$\mathrm{\mathbb{H}}$ (bits)")
    ax1.axvline(0.5, c="black", lw=4, alpha=0.2)
    ax1.text(-0.05, 1.05, "(a)", transform=ax1.transAxes, size=20, weight="bold")

    for item in [ax1.title, ax1.xaxis.label, ax1.yaxis.label]:
        item.set_fontsize(20)

    for item in ax1.get_xticklabels() + ax1.get_yticklabels():
        item.set_fontsize(18)

    # exponential plot
    x2 = np.linspace(0, 5, 501)
    y2 = 1 - np.log(x2)
    ax2.plot(x2, y2, c="black", ls="-", lw=4)
    ax2.axhline(0, c="black")

    ax2.set_xlim([0, 5])
    ax2.set_ylim([y2[-1], y2[1]])
    ax2.set_xlabel(r"$\lambda$ (yr)")
    ax2.set_ylabel(r"$\mathrm{\mathbb{H}}$ (nats)")
    ax2.text(-0.05, 1.05, "(b)", transform=ax2.transAxes, size=20, weight="bold")

    for item in [ax2.title, ax2.xaxis.label, ax2.yaxis.label]:
        item.set_fontsize(20)

    for item in ax2.get_xticklabels() + ax2.get_yticklabels():
        item.set_fontsize(18)

    # Poisson plot
    x3 = np.linspace(0, 5, 501)
    y3 = [x - xlogx(x) for x in x3]
    ax3.plot(x3, y3, c="black", ls="-", lw=4)
    ax3.axhline(0, c="black")

    ax3.set_xlim([0, 5])
    #    ax3.set_ylim([y3[-1], y3[1]])
    ax3.set_xlabel(r"$\lambda$ (yr)")
    ax3.set_ylabel(r"$\theta$ (nats/yr)")
    ax3.axvline(1, c="black", lw=4, alpha=0.2)
    ax3.text(-0.05, 1.05, "(c)", transform=ax3.transAxes, size=20, weight="bold")

    for item in [ax3.title, ax3.xaxis.label, ax3.yaxis.label]:
        item.set_fontsize(20)

    for item in ax3.get_xticklabels() + ax3.get_yticklabels():
        item.set_fontsize(18)

    fig.tight_layout()
    fig.savefig("simple_entropy_py.png")
    fig.savefig("simple_entropy_py.pdf")
    plt.close(fig)
