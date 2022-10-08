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

# df = pd.pivot_table(df, index=['dataset'], columns=['model'], fill_value=np.nan)
# df['N'] = df[('num_data', 'max', selected_models[0])].values
# df['D'] = df[('input_dim', 'max', selected_models[0])].values
# def f(r):
    
# df['nlpd'] = df.apply
# for metric in selected_metrics:
def _format(mean, std) -> str:
    if std is None:
        return f"{mean:.2f} (n/a)"
    else:
        return f"{mean:.2f} ({std:.2f})"


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

assert len(selected_metrics) == 1
table = f"""
\\begin{{tabular}}{{{'lll' + 'c' * (len(selected_models))}}}
dataset & N & D & {' & '.join(map(_format_model, selected_models))} \\\\
\\midrule
"""
for dataset in selected_datasets:
    N = df[df.dataset == dataset]['num_data'].values[0]
    D = df[df.dataset == dataset]['input_dim'].values[0]
    row = f"{_format_dataset(dataset)} & {N} & {D}"
    for metric in selected_metrics:
        for model in selected_models:
            vals = df[(df.dataset == dataset) & (df.model == model)][metric]
            m = np.mean(vals)
            s = np.std(vals) if len(vals) > 1 else None
            row  += ' & ' + format_number(m, s)

    table += (row + " \\\\ \n")

table += "\\bottomrule \n"
table += "\\end{tabular}" 
st.text_area("Latex table", value=table, height=400)