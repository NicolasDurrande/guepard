from typing import Set 
import glob
import json
import time

import numpy as np
import pandas as pd
import streamlit as st


st.title("Visualise results")

file_regex = st.text_input(label="Specify location of JSON files", value="./results/*",)
print(file_regex)

data = []
paths = []
for path in glob.glob(file_regex + ".json"):
    paths.append(path)
    with open(path) as json_file:
        data.append(json.load(json_file))

if len(paths) == 0:
    st.write("No results found - try a different regex")
    exit(0)

st.write("Results found:", len(data))
import pprint

pprint.pprint(paths)

# Show all ############################
df = pd.DataFrame.from_records(data)
df["paths"] = paths
st.write("Raw results")
st.write(df)

# Inspect failed experiments
st.write("Number of failed experiments")
df_failed = df[df.failed]
st.write(df_failed.groupby(['model', 'dataset']).agg({'split': 'count'}).reset_index())
df = df[~df.failed]
df_all = df.copy()

# Aggregate ############################
st.write("Aggregate")

default_groupby_keys = ["dataset", "model"]
average_over = "split"
all_metrics = ["rmse", "nlpd"]
all_datasets = list(df.dataset.unique())
all_models = list(df.model.unique())

groupby_keys = st.multiselect(
    label="Group by", options=list(df.columns), default=default_groupby_keys,
)

selected_metrics = st.multiselect(
    label="Metrics", options=all_metrics, default=all_metrics,
)
selected_datasets = st.multiselect(
    label="Datasets", options=all_datasets, default=all_datasets,
)

selected_models = st.multiselect(
    label="Models", options=all_models, default=all_models,
)

all_unique_elements = ["num_data", "input_dim"]

df = (
    df[df.dataset.isin(selected_datasets) & df.model.isin(selected_models)]
    .groupby(groupby_keys)
    .agg(
        {
            average_over: "count",
            **{element: "max" for element in all_unique_elements},
            **{metric: ["mean", "std"] for metric in selected_metrics},
        }
    )
)

st.dataframe(df)

def _format(mean, std) -> str:
    if std is None:
        return f"{mean:.2f} (n/a)"
    else:
        return f"{mean:.2f} ({std:.2f})"


df = df_all
table = []
for dataset in selected_datasets:
    N = df[df.dataset == dataset]['num_data'].values[0]
    D = df[df.dataset == dataset]['input_dim'].values[0]
    row = {'dataset': dataset, 'N': N, 'D': D}
    for metric in selected_metrics:
        for model in selected_models:
            vals = df[(df.dataset == dataset) & (df.model == model)][metric]
            m = np.mean(vals)
            s = np.std(vals) if len(vals) > 1 else None
            row[(model, metric)] = _format(m, s)
        
    table.append(row)

df_table = pd.DataFrame(table).set_index('dataset')
st.dataframe(df_table)

# Latex Table
# =============================================

# Only works for a single metric at the time.
selected_metrics = selected_metrics[:1]

def format_number(value, error, *, bold=False, possibly_negative=True):
    if np.isnan(value) or value is None:
        return ""
    if np.abs(value) > 10:
        return "F"
    if value >= 0 and possibly_negative:
        sign_spacer = "\\hphantom{-}"
    else:
        sign_spacer = ""
    if bold:
        bold_start, bold_end = "\\mathbf{", "}"
    else:
        bold_start, bold_end = "", ""
    if error is None:
        return f"${sign_spacer}{bold_start}{value:.2f}{bold_end} {{ \\pm \\scriptstyle X }}$"
    else:
        return f"${sign_spacer}{bold_start}{value:.2f}{bold_end} {{ \\pm \\scriptstyle {error:.2f} }}$"


def _format_dataset(name: str) -> str:
    if "wilson_" in name.lower():
        name = name[len("wilson_"):]
    return name.capitalize()

def _format_model(name: str) -> str:
    return name.replace('_', '-')


# Determine rank
ranks = (
    df[['dataset', 'model', selected_metrics[0], 'split']]
        .groupby(['dataset', 'model'])
        .agg({selected_metrics[0]: 'mean'})
        .pivot_table(index="dataset", columns="model")
        .rank(axis=1)
        .mean(axis=0)
)[selected_metrics[0]]
selected_models.sort(key=lambda m: ranks.at[m])


def n_best(df, col, col_err, *, smaller_is_better) -> Set[str]:
    ascending = smaller_is_better
    df = df.sort_values(col, ascending=ascending)
    best_indices = set()
    val0, err0 = df[[col, col_err]].iloc[0]
    i = 0
    while True:
        best_indices.add(df.index[i])

        # Compare with the next.
        val, err = df[[col, col_err]].iloc[i + 1]
        diff = abs(val0 - val)
        diff_err = np.sqrt(err0**2 + err**2)

        if diff > diff_err:
            # Significantly better.
            return best_indices
        else:
            # Not significantly better. Try the next.
            i += 1

df2 = (
    df[['dataset', 'model', selected_metrics[0], 'split']]
        .groupby(['dataset', 'model'])
        .agg({selected_metrics[0]: ['mean', 'std']})
        .pivot_table(index="model", columns="dataset")
)

best_for_dataset = {}
for dataset in selected_datasets:
    df_tmp = df2.xs(dataset, level='dataset', axis=1)[selected_metrics[0]]
    best_for_dataset[dataset] = n_best(df_tmp, 'mean', 'std', smaller_is_better=True)


table = f"""
\\begin{{tabular}}{{{'lll' + 'c' * (len(selected_models))}}}
\\toprule
dataset & N & D & {' & '.join(map(_format_model, selected_models))} \\\\
\\midrule
"""
selected_datasets.sort(key=lambda d: df[df.dataset == d]['num_data'].values[0])
for dataset in selected_datasets:
    N = df[df.dataset == dataset]['num_data'].values[0]
    D = df[df.dataset == dataset]['input_dim'].values[0]
    row = f"{_format_dataset(dataset)} & {N} & {D}"
    for metric in selected_metrics:
        for model in selected_models:
            neg_value_in_col = any(map(lambda v: '-' in v, df_table[(model, metric)].values))
            vals = df[(df.dataset == dataset) & (df.model == model)][metric]
            m = np.mean(vals)
            s = np.std(vals) if len(vals) > 1 else None
            row  += ' & ' + format_number(m, s, possibly_negative=neg_value_in_col, bold=model in best_for_dataset[dataset])

    table += (row + " \\\\ \n")

table += "\\midrule \n"
table += f"Avg. rank & & & {' & '.join(map(lambda m: f'{ranks.at[m]:.2f}', selected_models))} \\\\ \n"
table += "\\bottomrule \n"
table += "\\end{tabular}" 
st.text_area("Latex table", value=table, height=400)
