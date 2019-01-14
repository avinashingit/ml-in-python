# @Author: Avinash Kadimisetty <avinash>
# @Date:   2019-01-14T16:10:32-06:00
# @Email:  kavinash366@gmail.com
# @Project: Machine Learning in Python
# @Filename: oecd_linear_model.py
# @Last modified by:   avinash
# @Last modified time: 2019-01-14T16:18:37-06:00
# @License: MIT License

import sklearn.linear_model
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

oecd_data = pd.read_csv("Datasets/lifesat/oecd_bli_2015.csv", thousands=",")
gdp_data = pd.read_csv("Datasets/lifesat/gdp_per_capita.csv", thousands=",",
                       delimiter="\t", encoding="latin1", na_values="na")
