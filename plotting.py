import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import datetime

def plot_scatterplot(df, col, figsize, ncol):
    
    plt.clf()
    plt.style.use('fivethirtyeight')
    
    g = sns.FacetGrid(df, col=col, hue = 'Attrition', col_wrap=ncol)
    g = (g.map(sns.scatterplot, "Age", "MonthlyIncome").add_legend())
    g.fig.set_size_inches(figsize)

def plot_cohort(df, col1, col2):
    
    plt.clf()
    plt.figure(figsize=(20,8))
    font_opts = {"fontsize": 15, "fontweight": "bold"}

    #####################################################
    #####################################################

    plt.subplot(211)

    Company_Satisfaction = pd.crosstab(df[col2], df[col1])
    Company_Satisfaction = Company_Satisfaction.apply(lambda z: z/z.sum(), axis=1)

    ax = sns.heatmap(
        Company_Satisfaction,
        annot=True, 
        annot_kws={"size": 18},
        linewidths=.5,
        cmap="YlGnBu",
        fmt='.0%', 
        vmax=1,
        vmin=0
    )

    ax.set_title(col1 + " x " + col2 + "\n\nShare Rate\n", **font_opts)
    ax.set_xlabel("")
    ax.set_ylabel(col2+"\n")
    ax.set_xticklabels(labels="")
    plt.yticks(rotation=0)

    #####################################################
    #####################################################

    plt.subplot(212)

    turnover_rate = (
        df.groupby([col1, col2])["Attrition"].sum() / 
        df.groupby([col1, col2])["Attrition"].count()
    ).unstack(level=0)

    ax = sns.heatmap(
        turnover_rate,
        annot=True, 
        annot_kws={"size": 18},
        linewidths=.5,
        cmap="YlGnBu",
        fmt='.0%', 
        vmax=1, 
        vmin=0 
    )

    ax.set_title("Turnover Rate\n", **font_opts)
    ax.set_xlabel("\n"+col1)
    ax.set_ylabel(col2+"\n")
    plt.yticks(rotation=0)


    plt.tight_layout()
    plt.plot();
    
def plot_heatmap(df, col1, col2):
    
    plt.clf()
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(20,12))
    font_opts = {"fontsize": 15, "fontweight": "bold"}

    #####################################################
    #####################################################

    plt.subplot(311)

    Company_Satisfaction = pd.crosstab(df[col1], df[col2])
    Company_Satisfaction = Company_Satisfaction.apply(lambda z: z/z.sum(), axis=1)

    ax = sns.heatmap(
        Company_Satisfaction,
        annot=True, 
        annot_kws={"size": 18},
        linewidths=.5,
        cmap="YlGnBu",
        fmt='.0%', 
        vmax=1,
        vmin=0
    )

    ax.set_title("Average monthly income in jobs with the highest turnover rates (per "+col2+")\n\nShare Rate\n", 
                 **font_opts)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(labels="")
    plt.yticks(rotation=0)

    #####################################################
    #####################################################
    
    plt.subplot(312)
    
    turnover_rate = (
        df.groupby([col1, col2])["Attrition"].sum() / 
        df.groupby([col1, col2])["Attrition"].count()
    ).swaplevel().unstack(level=0)

    ax = sns.heatmap(
        turnover_rate,
        annot=True, 
        annot_kws={"size": 18},
        linewidths=.5,
        cmap="YlGnBu",
        fmt='.0%', 
        vmax=1, 
        vmin=0 
    )

    ax.set_title("Turnover Rate\n", **font_opts)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(labels="")
    plt.yticks(rotation=0)
    
    #####################################################
    #####################################################

    plt.subplot(313)

    salary_per_job = df.groupby([col1, col2])["MonthlyIncome"].median().unstack()

    ax = sns.heatmap(
        salary_per_job,
        annot=True, 
        annot_kws={"size": 18},
        linewidths=.5,
        cmap="YlGnBu",
        fmt='.0f'
    )

    ax.set_title("Salary Distribution\n", **font_opts)
    ax.set_xlabel("\n"+col2)
    ax.set_ylabel("")
    plt.yticks(rotation=0)


    plt.tight_layout()
    plt.plot();
    
def round_quantile(df, col):
    temp_df = (
        df.copy()
        .groupby([col])["Attrition"]
        .count()
        .reset_index()
        .sort_values(by=col)
    )
    temp_df["quantile"] = (temp_df["Attrition"] / temp_df["Attrition"].sum()).cumsum()
    temp_df["round_quantile"] = np.floor(temp_df["quantile"]*10)

    total = temp_df.groupby(["round_quantile"])["Attrition"].sum().reset_index()
    total.columns = ["round_quantile", "total"] 

    temp_df.drop_duplicates("round_quantile", inplace=True)
    temp_df = temp_df.merge(total, on="round_quantile", how="inner")
    
    return temp_df[["round_quantile", col, "total"]]

def merge_quantiles(df, col):
    
    overall = round_quantile(df, col)
    churned = round_quantile(df.loc[df["Attrition"] == 1], col)
    active = round_quantile(df.loc[df["Attrition"] == 0], col)
    
    temp_merge = churned.merge(active, how='inner', on="round_quantile")
    temp_merge = temp_merge.merge(overall, how='inner', on="round_quantile")
    temp_merge = temp_merge[["round_quantile", col+"_x", col+"_y", "total_x", "total_y", col]]
    temp_merge.columns = ["Round_Quantile",col+"_Churned",col+"_Active","Total_Churned","Total_Active","Overall_"+col]
    temp_merge["DELTA (Active - Churn)"] = temp_merge[col+"_Active"] - temp_merge[col+"_Churned"]
    temp_merge["Cumsum_Churn"] = temp_merge["Total_Churned"].cumsum() / temp_merge["Total_Churned"].sum() * 100
    if col == "YearsAtCompany":
        return temp_merge[["Round_Quantile","Overall_"+col,col+"_Active",col+"_Churned","Total_Churned","Cumsum_Churn"]]
    
    temp_merge["DELTA"] = np.round((temp_merge[col+"_Active"] - temp_merge[col+"_Churned"]) * 100 
                                       / temp_merge[col+"_Active"]).astype(int).astype(str) + "%"
    temp_merge = temp_merge[["Round_Quantile", "Overall_"+col, col+"_Active", col+"_Churned", 
                             "DELTA (Active - Churn)", "DELTA","Total_Churned", "Cumsum_Churn"]]
    
    return temp_merge