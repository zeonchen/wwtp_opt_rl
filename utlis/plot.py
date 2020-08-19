import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.pyplot import MultipleLocator


def action_plot(data, values, legend=False, ylabel='', xaxis='Epoch', condition=None):
    assert isinstance(data, pd.DataFrame), 'Data should be a pandas.DataFrame!'
    assert isinstance(values, list), 'Values should be a list!'
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    data['aer'] *= 5.0
    data['slu'] *= 160.0
    palette = sns.color_palette("mako_r", 6)
    sns.set(style="darkgrid")

    f, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1_major_locator = MultipleLocator(20)
    sns.lineplot(data=data, x=xaxis, y=values[0], linewidth=1, hue=condition, ci='sd', palette=palette)
    ax1.yaxis.set_major_locator(ax1_major_locator)
    plt.ylim(0, 160)

    ax1.set_ylabel('Dosage ($kg/d$)')

    ax2 = ax1.twinx()
    ax2_major_locator = MultipleLocator(0.5)
    sns.lineplot(data=data, x=xaxis, y=values[1], linewidth=1, hue=condition, ci='sd', color='g')
    ax2.yaxis.set_major_locator(ax2_major_locator)
    ax2.set_ylabel('Dissolved oxygen ($mg/L$)')
    plt.ylim(0, 5)
    ax2.grid(False)

    if legend:
        ax1.legend(['Dosage'], loc='best')
        ax2.legend(['Dissolved oxygen'], loc='lower right')

    xscale = np.max(np.asarray(data[xaxis])) > 5e3

    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)
    plt.savefig(fname="D:\\rl_wwtp\\parameter.pdf", dpi=300, format="pdf")
    plt.show()


def reward_plot(data, values, ylabel='', xaxis='Epoch', condition=None):
    assert isinstance(data, pd.DataFrame), 'Data should be a pandas.DataFrame!'
    assert isinstance(values, list), 'Values should be a list!'
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    palette = sns.color_palette("mako_r", 6)
    sns.set(style="darkgrid")

    f, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    sns.lineplot(data=data, x=xaxis, y=values[0], linewidth=1,  hue=condition, ci='sd', palette=palette)
    ax1.set_ylabel(ylabel)

    xscale = np.max(np.asarray(data[xaxis])) > 5e3

    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ylim(-2, -0.3)
    plt.tight_layout(pad=0.5)
    plt.savefig(fname="D:\\rl_wwtp\\reward.pdf", dpi=300, format="pdf")
    plt.show()
