import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sympy import symbols, Matrix, simplify, log

from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel

def keep_constr(x):
    xi_1 = 3
    xi_2 = 5
    xi_3 = 4

    B_12 = x[0]
    B_21 = x[1]
    z_1 = x[2]
    z_2 = x[3]

    constr_1 = abs(xi_1 - (B_12 + z_2))
    constr_2 = abs(xi_2 - (B_21 + z_1 + B_12 + z_2))
    constr_3 = abs(xi_3 - (z_1*B_12 + z_1*z_2 + B_21*z_2))

    res = (constr_1 + constr_2 + constr_3) * 1
    print('constr', res)
    return res

def theta(u,B):
    z_1 = -B_11 - B_21
    z_2 = -B_22 - B_12

    m = LinearAutonomousPoolModel(u, B, force_numerical=True)

    ET = m.T_expected_value
    x = m.xss

    def x1minuslogx(x):
        if x == 0: return 0
        return x*(1-log(x))

    H0 = x[0] * (x1minuslogx(B[1,0]) + x1minuslogx(z_1))
    H1 = x[1] * (x1minuslogx(B[0,1]) + x1minuslogx(z_2))

    res = (H0+H1)/ET
    return res

def f(x):
    B_12 = x[0]
    B_21 = x[1]
    z_1 = x[2]
    z_2 = x[3]

    B_11 = -(B_21 + z_1)
    B_22 = -(B_12 + z_2)

    B = Matrix([[B_11, B_12], [B_21, B_22]])
    u = Matrix(2,1, [1,0])
    #model = LinearAutonomousPoolModel(u, B, force_numerical=True)

    #res = -model.entropy_rate
    res = - theta(u,B).evalf()
    print(B, -res.evalf())

    constr = keep_constr(x)
    if constr <= 1e-02:
        entropy_values.append(-res)
        constr_values.append(constr)
    return res


if __name__ == '__main__':
    # the result that should be obtained by the optimization
    B_11, B_12, B_21, B_22, z_1, z_2 = symbols('B_11 B_12 B_21 B_22 z_1 z_2')

    B_11 = -2
    B_12 = 2
    B_21 = 1
    B_22 = -3

    B = Matrix([[B_11, B_12], [B_21, B_22]])
    u = Matrix(2,1, [1,0])
    model = LinearAutonomousPoolModel(u, B)

    res = model.entropy_rate
    print('target B =', B)
    print('target xss =', model.xss)
    print('target ET =', model.T_expected_value)
    print('target entropy rate:', simplify(res), '=', res.evalf())
    print('theta =', theta(u,B), '=', theta(u,B).evalf())
    print('\n----------------------\n')
    #input()

    # optimization
    x0 = [3, 0, 1, 1] # B_12, B_21, z_1, z_2
    bnds = [(0,None)] * 4
    constr = {'type': 'eq', 'fun': keep_constr}

    entropy_values = []
    x_values = []
    constr_values = []
    try:
        y = -minimize(f, x0, bounds=bnds, constraints=constr, tol=1e-6)
    except TypeError:
        print('Optimization aborted due to TypeError.')

    steps = range(len(entropy_values))
    
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(1,1,1)

    ax.plot(steps, entropy_values, c="black", label=r'$\theta(M)$ during optimization')
    #ax.plot(steps, constr_values)
    ax.plot(steps, [res.evalf()]*len(steps), c='black', ls="--", label=r'$\theta(\widetilde{M})$')

    ax.set_xlim([steps[0],steps[-1]])
    ax.set_ylim([1.75,1.95])
    ax.set_ylabel(r'$\theta$ (nats/yr)', fontsize=20)
    ax.set_xlabel('number of iterations', fontsize=20)

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    
    ax.legend(fontsize=20, loc=4)

    #fig.savefig('../figs/optimization.pdf')
    fig.savefig('optimization.png')
    fig.savefig('optimization.pdf')
    plt.close(fig)


