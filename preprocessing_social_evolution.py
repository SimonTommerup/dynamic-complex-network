# %%
import os
import pandas as pd
from datetime import datetime

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
SURVEY_TIME = "12:00:00"

def original_fpath(file_name):
    datafolder = r"data/soc-evo/source"
    return os.path.join(datafolder, file_name)

def processed_fpath(file_name):
    datafolder = r"data/soc-evo/pre-processed"
    return os.path.join(datafolder, file_name)

def rename_columns(data_frame, previous_names, new_names):
    for idx, previous_name in enumerate(previous_names):
        data_frame.columns = [new_names[idx] if x == previous_name else x for x in data_frame.columns]

def preprocess_proximity(proximity_dataframe, probability_threshold=0.85):
    pdf = proximity_dataframe.copy()
    pdf = pdf[pdf["remote.user.id.if.known"].notna()]
    pdf = pdf[pdf["prob2"].notna()]
    pdf = pdf[pdf["prob2"] >= probability_threshold]
    pdf.pop("prob2")
    pdf = pdf[pdf["user.id"] != pdf["remote.user.id.if.known"]]
    rename_columns(pdf, ["user.id","remote.user.id.if.known"], ["source", "destination"])
    pdf["event_type"] = 1
    return pdf

def preprocess_sms(sms_dataframe):
    sdf = sms_dataframe.copy()
    sdf = sdf[sdf["dest.user.id.if.known"].notna()]
    sdf = sdf[sdf["user.id"] != sdf["dest.user.id.if.known"]]
    sdf.pop("incoming")
    sdf.pop("dest.phone.hash")
    rename_columns(sdf, ["user.id", "dest.user.id.if.known"], ["source", "destination"])
    sdf["event_type"] = 1
    return sdf

def preprocess_calls(calls_dataframe):
    cdf = calls_dataframe.copy()
    cdf = cdf[cdf["dest_user_id_if_known"].notna()]
    cdf = cdf[cdf["user_id"] != cdf["dest_user_id_if_known"]]
    cdf.pop("duration")
    cdf.pop("dest_phone_hash")
    rename_columns(cdf, ["user_id", "dest_user_id_if_known", "time_stamp"], ["source", "destination", "time"])
    cdf["event_type"] = 1
    return cdf

def preprocess_relationship(relationship_dataframe):
    rdf = relationship_dataframe.copy()
    rdf = rdf[rdf["relationship"] == "CloseFriend"]
    rdf["survey.date"] = [d + " " + SURVEY_TIME for d in rdf["survey.date"]]
    rename_columns(rdf, ["id.A", "id.B", "survey.date"], ["source", "destination", "time"])
    rdf.pop("relationship")
    rdf["event_type"] = 0
    return rdf

def preprocess_and_write_to_csv(file_names, dataframes, preprocessing_functions):
    for idx, dataframe in enumerate(dataframes):
        p = preprocessing_functions[idx](dataframe)
        file_path = processed_fpath("pp-" + file_names[idx])
        p.to_csv(file_path, index=False)

def sort_by_datetime(dataframe):
    sdf = dataframe.copy()
    sdf["time"] = pd.to_datetime(sdf.time, format=DATE_FORMAT)
    sdf.sort_values(by=["time"], inplace=True)
    return sdf

def remove_nan_values_and_node_loops(dataframe):
    df = dataframe.copy()
    df = df.dropna()
    df["source"] = [int(s) for s in df["source"]]
    df["destination"] = [int(d) for d in df["destination"]]
    df = df[df["source"] != df["destination"]]
    return df

def remove_repeated_associations(dataframe):
    df = dataframe.reset_index(drop=True)
    indices = []
    previous_associations = []
    all_events = [(e,u,v) for e,u,v in zip(df["event_type"],df["source"], df["destination"])]
    for idx, tup in enumerate(all_events):
        event_type = tup[0]
        u = tup[1]
        v = tup[2]
        if event_type == 0: # association
            if (u,v) in previous_associations:
                continue
            else:
                indices.append(idx)
                previous_associations.append((u,v))
                #previous_associations.append((v,u)) 
        elif event_type == 1: # communication
            indices.append(idx)
    df = df[df.index.isin(indices)]
    return df

def split_to_initial_associations_and_data_sets(dataframe):
    # rename node 84 to node 13 (magic observation)
    dataframe["source"] = [13 if x == 84 else x for x in dataframe["source"]]
    dataframe["destination"] = [13 if x == 84 else x for x in dataframe["destination"]]

    # zero index nodes
    dataframe["source"] = [x-1 for x in dataframe["source"]]
    dataframe["destination"] = [x-1 for x in dataframe["destination"]]

    # Initial: Association events before Sep 10 2008
    initial = dataframe.copy()
    initial = initial[pd.to_datetime(initial["time"]) < datetime.strptime("2008-09-10 00:00:00", DATE_FORMAT)]
    initial = initial[initial["event_type"]==0]
    initial_fpath = processed_fpath("soc-evo-initial-associations" + ".csv")
    initial.to_csv(initial_fpath, index=False)

    # Full data set: All events after Sep 10 2008 and until June 30 2009
    dataset = dataframe.copy()
    dataset = dataset[pd.to_datetime(dataset["time"]) >= datetime.strptime("2008-09-10 00:00:00", DATE_FORMAT)]
    dataset = dataset[pd.to_datetime(dataset["time"]) < datetime.strptime("2009-07-01 00:00:00", DATE_FORMAT)]
    insert_time_delta_column(dataset)
    full_fpath = processed_fpath("soc-evo-full-data-set" + ".csv")
    dataset.to_csv(full_fpath, index=False)

    # Train data set: All events after Sep 10 2008 and before May 1 2009
    train = dataset.copy()
    train = train[pd.to_datetime(train["time"]) < datetime.strptime("2009-05-01 00:00:00", DATE_FORMAT)]
    train_fpath = processed_fpath("soc-evo-train-data-set" + ".csv")
    train.to_csv(train_fpath, index=False)

    # Test data set: All events after May 1 2009 until June 30 2009
    test = dataset.copy()
    test = test[pd.to_datetime(test["time"]) >= datetime.strptime("2009-05-01 00:00:00", DATE_FORMAT)]
    test_fpath = processed_fpath("soc-evo-test-data-set" + ".csv")
    test.to_csv(test_fpath, index=False)

def insert_time_delta_column(dataframe):
    t0 = pd.to_datetime(dataframe["time"].iloc[0])
    time_deltas = [ t - t0 for t in pd.to_datetime(dataframe["time"]) ]
    time_deltas = [ t.total_seconds() for t in time_deltas ]
    dataframe["time_delta"] = time_deltas

def dataset_statistics(file_name):
    dataframe = pd.read_csv(processed_fpath(file_name))
    statistics = {}
    statistics["start_date"] = dataframe["time"].iloc[0]
    statistics["end_date"] = dataframe["time"].iloc[-1]
    statistics["len"] = len(dataframe)
    statistics["n_association_events"] = len([e for e in dataframe["event_type"] if e == 0])
    statistics["n_communication_events"] = len([e for e in dataframe["event_type"] if e == 1])
    return dataframe, statistics

def print_statistics(dataset_statistics, text):
    print(text + "\n")
    for k, v in dataset_statistics.items():
        print(f"{k}: {v}")
    print("\n")

if __name__ == "__main__":
    fnames = ["proximity.csv", "sms.csv", "calls.csv", "relationship.csv"]
    dataframes = [pd.read_csv(original_fpath(fname)) for fname in fnames]
    ppfuncs = [preprocess_proximity, preprocess_sms, preprocess_calls, preprocess_relationship]
    preprocess_and_write_to_csv(fnames, dataframes, ppfuncs)
    processed_fnames = [processed_fpath("pp-" + fname) for fname in fnames]
    processed_dataframes = [pd.read_csv(p_fname) for p_fname in processed_fnames]
    
    merged_dataframe = pd.concat(processed_dataframes)
    sorted_dataframe = sort_by_datetime(merged_dataframe)
    cleaned_dataframe = remove_nan_values_and_node_loops(sorted_dataframe)
    
    cleaned_dataframe = remove_repeated_associations(cleaned_dataframe)
    
    cleaned_dataframe.to_csv(processed_fpath("soc-evo.csv"), index=False)
    split_to_initial_associations_and_data_sets(cleaned_dataframe)
    
    init_df, init_stats = dataset_statistics("soc-evo-initial-associations.csv")
    full_df, full_stats = dataset_statistics("soc-evo-full-data-set.csv")
    train_df, train_stats = dataset_statistics("soc-evo-train-data-set.csv")
    test_df, test_stats = dataset_statistics("soc-evo-test-data-set.csv")

    print_statistics(init_stats, "Init:")
    print_statistics(full_stats, "Full:")
    print_statistics(train_stats, "Train:")
    print_statistics(test_stats, "Test:")







    


# %%
print(train_df.columns)
# %%
