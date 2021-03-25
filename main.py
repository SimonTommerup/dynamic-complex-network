import dyrep
import data_sets
import utils

if __name__=="__main__":
    device = "cuda:0"
    initial_associations = utils.read_csv(r"data\soc-evo\pre-processed\soc-evo-initial-associations.csv")
    training_data = data_sets.MITDataSet(r"data\soc-evo\pre-processed\soc-evo-train-data-set.csv")

    model = dyrep.DyRep(training_data, initial_associations, device=device)
    model.to(device)

    model = dyrep.train(model, training_data)


