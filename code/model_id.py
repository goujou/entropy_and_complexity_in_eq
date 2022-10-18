"""Model identification simulation."""

import matplotlib.pyplot as plt
# http://github.com/MPIBGC-TEE
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from scipy.optimize import minimize
from sympy import Matrix, log, simplify, symbols


def keep_constr(x):
    """Ensure that constraints are kept."""
    xi_1 = 3
    xi_2 = 5
    xi_3 = 4

    B_12 = x[0]
    B_21 = x[1]
    z_1 = x[2]
    z_2 = x[3]

    constr_1 = abs(xi_1 - (B_12 + z_2))
    constr_2 = abs(xi_2 - (B_21 + z_1 + B_12 + z_2))
    constr_3 = abs(xi_3 - (z_1 * B_12 + z_1 * z_2 + B_21 * z_2))

    res = constr_1 + constr_2 + constr_3
    print("constr", res)
    return res


def theta_func(u, B):
    """Entropy rat per unit time.

    Using this customized function for the sake of stability.
    """
    z_1 = -B_11 - B_21
    z_2 = -B_22 - B_12

    # load model (LAPM)
    m = LinearAutonomousPoolModel(u, B, force_numerical=True)

    # mean transit time
    ET = m.T_expected_value

    # steady state
    x = m.xss

    def xminuslogx(x):
        """Make sure ``0*(1-log(0)) = 0``."""
        if x == 0:
            return 0
        return x * (1 - log(x))

    H0 = x[0] * (xminuslogx(B[1, 0]) + xminuslogx(z_1))
    H1 = x[1] * (xminuslogx(B[0, 1]) + xminuslogx(z_2))

    theta = (H0 + H1) / ET
    return theta


def f(x):
    """Function to minimize, negative entropy rate per unit time."""
    B_12 = x[0]
    B_21 = x[1]
    z_1 = x[2]
    z_2 = x[3]

    B_11 = -(B_21 + z_1)
    B_22 = -(B_12 + z_2)

    B = Matrix([[B_11, B_12], [B_21, B_22]])
    u = Matrix(2, 1, [1, 0])

    # negative entropy rate
    theta = theta_func(u, B).evalf()
    print(B, theta.evalf())

    # add values to history only if constraints are not too
    # severley broken
    constr = keep_constr(x)
    if constr <= 1e-02:
        entropy_values.append(theta)
        constr_values.append(constr)
    return -theta


if __name__ == "__main__":
    # tilde B
    B_11, B_12, B_21, B_22, z_1, z_2 = symbols("B_11 B_12 B_21 B_22 z_1 z_2")

    B_11 = -2
    B_12 = 2
    B_21 = 1
    B_22 = -3

    B = Matrix([[B_11, B_12], [B_21, B_22]])
    u = Matrix(2, 1, [1, 0])
    model_tilde = LinearAutonomousPoolModel(u, B)
    theta_tilde = model_tilde.entropy_rate

    print("tilde B =", B)
    print("tilde xss =", model_tilde.xss)
    print("tilde ET =", model_tilde.T_expected_value)
    print("tilde entropy rate:", simplify(theta_tilde), "=", theta_tilde.evalf())
    print("\n----------------------\n")
    # input()

    # optimization
    x0 = [3, 0, 1, 1]  # B_12, B_21, z_1, z_2
    bnds = [(0, None)] * 4
    constr = {"type": "eq", "fun": keep_constr}

    entropy_values = []
    x_values = []
    constr_values = []
    # add entropy values until until minimization terminates
    min_res = minimize(f, x0, bounds=bnds, constraints=constr, tol=1e-6)
    print(min_res)
    if min_res.success:
        # make plots
        steps = range(len(entropy_values))

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(
            steps,
            entropy_values,
            c="black",
            label=r"$\theta(M)$ during optimization",
            lw=4,
        )
        # ax.plot(steps, constr_values)
        ax.plot(
            steps,
            [theta_tilde.evalf()] * len(steps),
            c="black",
            ls="--",
            label=r"$\theta(\widetilde{M})$",
            lw=4,
        )

        ax.set_xlim([steps[0], steps[-1]])
        ax.set_ylim([1.75, 1.95])
        ax.set_ylabel(r"$\theta$ (nats/yr)", fontsize=20)
        ax.set_xlabel("number of iterations", fontsize=20)

        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(18)

        ax.legend(fontsize=20, loc=4)

        # save plots
        # fig.savefig('../figs/optimization.pdf')
        fig.savefig("optimization.png")
        fig.savefig("optimization.pdf")
        plt.close(fig)
    else:
        print("Minimzation procedure was not successful.")
