# -*- coding: utf-8 -*-
# @Time    : 2021/3/5 17:13
# @Author  : Equator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


# df = pd.read_csv('data_single.csv', names=['square', 'price'])
def single_data(df):
    sns.lmplot('square', 'price', df, fit_reg=True)
    plt.show()
    print(df.head())
    print(df.info())


def multi_data(df):
    df.head()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('square')
    ax.set_ylabel('bedrooms')
    ax.set_zlabel('price')
    ax.scatter3D(df['square'], df['bedrooms'], df['price'], c=df['price'], cmap='Greens')
    plt.show()


# 数据归一化
def normalize(df):
    return df.apply(lambda col: (col - col.mean()) / col.std())


def normalize_feature():
    df = pd.read_csv('data_multi.csv', names=['square', 'bedrooms', 'price'])
    df = normalize(df)
    multi_data(df)


def add_one_col():
    df = pd.read_csv('data_multi.csv', names=['square', 'bedrooms', 'price'])
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    # 根据列合并数据
    df = pd.concat([ones, df], axis=1)
    return df


if __name__ == '__main__':
    df = add_one_col()
    print(df.head())