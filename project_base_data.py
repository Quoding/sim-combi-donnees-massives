# %%
import numpy as np
import pandas as pd
from math import ceil

N_WINDOWS = 5.1e9

# %%
cols = ["drug_name_A", "drug_name_B", "drug_name_C", "drug_name_D", "drug_name_E"]
df_out = pd.DataFrame([], columns=cols + ["exact_exposure_count"])  # Output dataframe
unique_drugs = pd.DataFrame(
    [], columns=["drug_name"]
)  # All unique drug names, used for columns in projected dataset

for i in range(1, 6):
    fn = f"base_data/db_drugs_{i}s.tsv"
    df = pd.read_csv(fn, delimiter="\t", float_precision="high")
    tmp_df = df[cols[:i] + ["exact_exposure_count"]]
    stacked_drugs = df[cols[:i]].T.stack().reset_index(name="drug_name")
    unique_drugs = pd.concat([unique_drugs, stacked_drugs])
    tmp_df[cols[i:]] = float("nan")

    df_out = pd.concat([df_out, tmp_df])


# %%
unique_drugs = unique_drugs["drug_name"].unique()


# %%
df_out = df_out.replace(r"^\s*$", np.nan, regex=True)  # Replace empty strings by np.nan

df_out = df_out.dropna(
    subset=["exact_exposure_count"]
)  # Drop rows which have no information on fraction
df_out["exact_exposure_count"] = pd.to_numeric(
    df_out["exact_exposure_count"], errors="coerce"
)
df_out = df_out.dropna(
    subset=["exact_exposure_count"]
)  # Drop rows which have no information on fraction


df_out["ratio_exact"] = (
    df_out["exact_exposure_count"] / N_WINDOWS
)  # Find the ratio of exact exposures across all the windows of the source dataset

df_out["ratio_exact"] = df_out["ratio_exact"] * (
    1 / df_out["ratio_exact"].sum()
)  # Rescale so that current combinations are basically all that ever existed (rescale sum of ratios to 1)
df_out = df_out[df_out["ratio_exact"] != 0]

df_out = df_out.drop(
    labels=["exact_exposure_count"], axis=1
)  # Drop the exact count, so we can rescale the dataset to an arbitrary amount of observations


# %%
print(f"{len(unique_drugs)=}")


# %%
sample_dataset = pd.DataFrame([], columns=unique_drugs)
num_rows_dataset = 100000

# %%
for index, row in df_out.iterrows():
    combination = tuple(row[:-1].dropna())
    fraction = float(row[-1])
    num_rows_combi = round(fraction * num_rows_dataset)
    dic = {key: [1] for key in combination}
    dummy_df = pd.DataFrame(dic, columns=unique_drugs).fillna(0)
    dummy_df = pd.DataFrame(
        np.repeat(dummy_df.values, num_rows_combi, axis=0), columns=dummy_df.columns
    )

    sample_dataset = pd.concat([sample_dataset, dummy_df])


# %%
sample_dataset.to_csv("new_data/sample_dataset.csv")

# %%
