import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/ece_vs_conf_and_alpha.csv")

res = df.plot(x="confidence", y = "ECE", kind ="line").get_figure()

res.savefig("results/ece_vs_conf_alpha.jpg")


