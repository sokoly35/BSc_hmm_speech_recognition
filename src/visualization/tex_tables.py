import numpy as np
import pandas as pd
from typing import List

def latex_table_generator_comparison(df: pd.DataFrame) -> None:
    """
    function generates and prints latex code of pivot table. In columns should be beta
    params, in rows different SNR.

    Parameters
    ----------
    df (pd.DataFrame): table with results

    Returns
    -------
        None
    """
    # Creating table syntax
    table = r"""\begin{table}[]
\centering
\begin{tabular}{|c|c|c|c|c|c|}
\hline
\rowcolor[HTML]{C0C0C0} """

    # We expect that colmns will be  values of beta
    table += r"\textbf{" + r"$\beta$" + r"} & "
    for i, column in enumerate(df.columns, 1):
        # If it will be last column we need to end it with \hline
        if i == len(df.columns):
            table += r"\textbf{" + str(column) + r"} \\ \hline" + "\n"
        # Adding new column
        else:
            table += r"\textbf{" + str(column) + "} & "
    # Specyfing row color
    table += r"\rowcolor[HTML]{C0C0C0}"
    # Specyfing index col name -- SNR
    table += r"$\mathbf{\mathit{SNR}_{dB}}$ " + r"} " + r" & \multicolumn{5}{c|}{}"
    table += "\\\\ \\hline \n"
    # FIlling table values with our results
    for row in df.iterrows():
        # Adding index of row
        table += r"\cellcolor[HTML]{EFEFEF}" + str(row[0]) + " & "
        for i, num in enumerate(row[1], 1):
            # Case if this is last value from row
            if i == len(row[1]):
                # If value is 0 then it cell will be orange
                if num == 0:
                    table += r"\cellcolor[HTML]{FFFC9E}" + str(np.round(num, 3))
                # If number is positive then cell will be green
                elif num > 0:
                    table += r"\cellcolor[HTML]{9AFF99}" + str(np.round(num, 3))
                # If number is negative then cell will be red
                else:
                    table += r"\cellcolor[HTML]{FFCCC9}" + str(np.round(num, 3))
            else:
                # If value is 0 then it cell will be orange
                if num == 0:
                    table += r"\cellcolor[HTML]{FFFC9E}" + str(np.round(num, 3)) + " & "
                # If number is positive then cell will be green
                elif num > 0:
                    table += r"\cellcolor[HTML]{9AFF99}" + str(np.round(num, 3)) + " & "
                # If number is negative then cell will be red
                else:
                    table += r"\cellcolor[HTML]{FFCCC9}" + str(np.round(num, 3)) + " & "
        table += "\\\\ \\hline \n"
    # ending tabular
    table += r"""\end{tabular}
\caption{}
\label{tab:}
\end{table}"""
    # printing result
    print(table)


def latex_table_generator(df: pd.DataFrame, columns: List[str], index: List[str]) -> None:
    """
    function generates and prints latex code of table. In columns should be different
    types of metrics, in row some parameter.

    Parameters
    ----------
    df (pd.DataFrame): table with results
    columns (list) : names of columns
    index (list) : names of index columns

    Returns
    -------
        None
    """
    # Creating table syntax
    table = r"""\begin{table}[]
\centering
\begin{tabular}{|c|c|c|c|c|c|}
\hline
\rowcolor[HTML]{C0C0C0} """
    # Specyfing index col name
    for id_ in index:
        table += r"\textbf{" + id_ + r"} & "
    # Naming columns
    for i, column in enumerate(columns, 1):
        # Case when this is last column
        if i == len(columns):
            table += r"\textbf{" + column + "}"
        else:
            table += r"\textbf{" + column + "} & "
    table += "\\\\ \\hline \n"
    # Filling table values
    for row in df.iterrows():
        # Adding index of row
        table += r"\cellcolor[HTML]{EFEFEF}" + str(row[0]) + " & "
        for i, num in enumerate(row[1], 1):
            # Case when value is last in the row
            if i == len(row[1]):
                table += str(np.round(num, 3))
            else:
                table += str(np.round(num, 3)) + " & "
        table += "\\\\ \\hline \n"
    # Ending table
    table += r"""\end{tabular}
\caption{}
\label{tab:}
\end{table}"""
    # Printing result
    print(table)


def latex_table_generator_multirow(df: pd.DataFrame, columns: List[str], index: List[str], multi: int) -> None:
    """
    function generates and prints latex code of table. In columns should be different
    types of metrics, in row some parameter.

    Parameters
    ----------
    df (pd.DataFrame): table with results
    columns (list) : names of columns
    index (list) : names of index columns
    multi (int) : number of rows in subsection of second index column

    Returns
    -------
        None
    """
    # Creating table syntax
    table = r"""\begin{table}[]
\centering
\begin{tabular}{|c|c|c|c|c|c|}
\hline
\rowcolor[HTML]{C0C0C0} """
    # Specyfying names of index columns
    for id_ in index:
        table += r"\textbf{" + id_ + r"} & "
    # Specyfing column names
    for i, column in enumerate(columns, 1):
        # Case when column is last in the row
        if i == len(columns):
            table += r"\textbf{" + column + "}"
        else:
            table += r"\textbf{" + column + "} & "
    table += "\\\\ \\hline \n"

    # Filling values of table
    for k, row in enumerate(df.iterrows(), 1):
        # We need to create multirow in first index column
        # this is the case when it is first row of group from the second index column
        if k % multi == 0:
            table += r"\multirow{-" + str(multi) + r"}{*}{\cellcolor[HTML]{EFEFEF}" + str(row[0][0]) + "} & "
        # In other way we make this
        else:
            table += r"\cellcolor[HTML]{EFEFEF}" + " & "
        # Specyfing value of second index col
        table += r"\cellcolor[HTML]{EFEFEF}" + str(row[0][1]) + " & "
        for i, num in enumerate(row[1], 1):
            # Case when value is last in the row
            if i == len(row[1]):
                table += str(np.round(num, 3))
            else:
                table += str(np.round(num, 3)) + " & "
        # different row ending depending of multi value
        if k % multi == 0:
            table += "\\\\ \\hline \n"
        else:
            table += "\\\\ \\cline{2-6} \n"
    # Ending table
    table += r"""\end{tabular}
\caption{}
\label{tab:}
\end{table}"""
    # print result
    print(table)
