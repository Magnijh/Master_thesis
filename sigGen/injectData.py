import os
import pandas as pd
import sigGen.synthetic_dataset as sd

def injectDataset(
    injectProcent: int,
    datasetRealPath: str |pd.DataFrame,
    output_path: str = os.path.dirname(__file__) + "/datasetOut/",
    nrOfSignatures: int = 1,
):
    
    # Create output folder if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if output_path[-1]!="/":
        output_path = output_path+"/"

    # Import dataset
    if isinstance(datasetRealPath, pd.DataFrame):
        realData = datasetRealPath
    else:
        realData = pd.read_table(datasetRealPath, index_col=0)

    # Getting the % vaule of columns
    columnsNum = realData.shape[1]
    columnsPercentage = int(columnsNum * (injectProcent / 100))
    columnsPercentage = columnsPercentage if columnsPercentage > 0 else 1
    synthData,sig,weights = sd.create_dataset(nrOfSignatures, columnsPercentage, "sigGen/signatures/custom-made.tsv",output_path,mutationRange=(realData.sum().min(),realData.sum().max()))

    # Concatenate along columns to add synthetic data as extra columns
    injectedData = pd.concat([realData, synthData], axis=1)

    # Shuffle the columns of the concatenated DataFrame
    injectedData = injectedData.sample(frac=1, axis=1)
    
    injectedData.to_csv(
        output_path + "injectedData.txt",
        sep="\t",
    )
    print(output_path)
    return synthData, injectedData, sig, weights


if __name__ == "__main__":
    injectDataset(10, "eval\datasets\split_on_underscore\Panc\Panc_96.txt")