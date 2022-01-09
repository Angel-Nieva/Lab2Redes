import matplotlib.pyplot as plt


def graficar(titulo, x, y, xlab, ylab):
    """ Permite graficar una señal """
    plt.figure()
    plt.title(titulo)
    plt.plot(x, y, linewidth = 0.5)
    plt.xlabel(xlab)
    plt.ylabel(ylab)