import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def make_pointplot(df: pd.DataFrame, x: str, hue: str,
                   ylabels: List[str], titles: List[str], scores: List[str],
                   legend_title: str, title_x: str, xlabel: str):
    """
    function make an column of multiple pointplots divided by x and hue.

    Parameters
    ----------
    df (pd.DataFrame): dataframe with results
    x (str): column to use as x axis
    hue (str): column to use as different cases in plot
    ylabels (list): names of y labels
    titles (list): list of names inserted to different figures titles
    scores (list): name of scores used to plot
    legend_title (str): title of legend, connected with hue
    title_x (str): Title template for plots
    xlabel (str): xlabel, connected with x

    Returns
    -------
        fig (plt.figure): matplotlib object with figure
    """
    # Define figures and axes
    fig, ax = plt.subplots(4, 1, figsize=(10, 15))
    ax = ax.flatten()

    # Plot for each metric
    for axis, ylabel, title, score in zip(ax, ylabels, titles, scores):
        # Make plot
        sns.pointplot(x=x,
                      y=score,
                      hue=hue,
                      data=df,
                      ax=axis)
        # Set titles and labels
        axis.set_title(f'Wykres {title_x} od {title}')
        axis.set_xlabel(f'{xlabel}')
        axis.set_ylabel(ylabel)
        axis.legend(title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    return fig


def make_boxplot(df: pd.DataFrame, x: str, hue: str,
                 ylabels: List[str], titles: List[str], scores: List[str],
                 legend_title: str, title_x: str, xlabel: str):
    """
    function make an column of multiple boxplots divided by x and hue.

    Parameters
    ----------
    df (pd.DataFrame): dataframe with results
    x (str): column to use as x axis
    hue (str): column to use as different cases in plot
    ylabels (list): names of y labels
    titles (list): list of names inserted to different figures titles
    scores (list): name of scores used to plot
    legend_title (str): title of legend, connected with hue
    title_x (str): Title template for plots
    xlabel (str): xlabel, connected with x

    Returns
    -------
        fig (plt.figure): matplotlib object with figure
    """
    # Define figures and axes
    fig, ax = plt.subplots(4, 1, figsize=(10, 15))
    ax = ax.flatten()

    # Plot for each metric
    for axis, ylabel, title, score in zip(ax, ylabels, titles, [f'{score}_mean' for score in scores]):
        # Make plot
        sns.boxplot(x=x,
                    y=score,
                    hue=hue,
                    data=df,
                    ax=axis)
        # Set titles and labels
        axis.set_title(f'Wykres {title_x} od {title}')
        axis.set_xlabel(f'{xlabel}')
        axis.set_ylabel(ylabel)
        axis.legend(title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    return fig


def make_barplot(df: pd.DataFrame, x: str, hue: str,
                 ylabels: List[str], titles: List[str], scores: List[str],
                 title_x: str, xlabel: str):
    """
    function make an column of multiple barplots divided by x and hue.

    Parameters
    ----------
    df (pd.DataFrame): dataframe with results
    x (str): column to use as x axis
    hue (str): column to use as different cases in plot
    ylabels (list): names of y labels
    titles (list): list of names inserted to different figures titles
    scores (list): name of scores used to plot
    title_x (str): Title template for plots
    xlabel (str): xlabel, connected with x

    Returns
    -------
        fig (plt.figure): matplotlib object with figure
    """
    # Define figures and axes
    fig, ax = plt.subplots(4, 1, figsize=(10, 15))
    ax = ax.flatten()

    # Plot for each metric
    for axis, ylabel, title, score in zip(ax, ylabels, titles, scores):
        # Make plot
        sns.barplot(x=x,
                    y=score,
                    hue=hue,
                    data=df,
                    ax=axis)
        # Set titles and labels
        axis.set_title(f'Wykres {title_x} od {title}')
        axis.set_xlabel(f'{xlabel}')
        axis.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    return fig


def make_heatmap(df: pd.DataFrame):
    """
        function makes an annotated heatmap from accuracy of results.

        Parameters
        ----------
        df (pd.DataFrame): dataframe with results

        Returns
        -------
            fig (plt.figure): matplotlib object with figure
        """
    # Define figure
    fig = plt.figure(figsize=(10, 10))

    # Plot heatmap
    sns.heatmap(data=df, vmin=0, vmax=1,
                annot=True, cmap='coolwarm_r', linewidths=.5,
                annot_kws={'fontsize': 13},
                cbar_kws={'label': 'Dokładność'})

    # Specify labels, ticks and titles
    plt.yticks(rotation=0, fontsize=13)
    plt.xticks(fontsize=13)
    plt.title('Wykres dokładności w zależności od parametru beta i SNR.', fontsize=18)
    plt.tight_layout()
    plt.show()
    return fig


def highlight_pos(cell):
    """
    helper function for pd.DataFrame display options. It changes color of text in cell
    to green, red, orange depending on value

    Parameters
    ----------
    cell: value in pd.DataFrame cell

    Returns
    -------
        str : color of text in cell
    """

    # If cell is number and negative then red
    if type(cell) != str and cell < 0:
        return 'color: red'
    # If cell is number and equall 0 then orange
    elif type(cell) != str and cell == 0:
        return 'color: orange'
    # If cell is number and positive then green
    else:
        return 'color: green'
