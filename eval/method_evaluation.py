import os
import re
import numpy as np
import pandas as pd
from typing import Literal
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from scipy.optimize import linear_sum_assignment
import Sprint

##
# what we want to evaluate:
# 1. in the case of a unknown dataset
#   - how many COSMIC signatures are found
#       - at a given cosine similarity threshold ex. 0.8
#   - how many signatures are found altogether
#   - maby compare with other findings?
# 2. in the case of a known dataset
#   - how many of the known signatures are found
#       - at a given cosine similarity threshold ex. 0.8
#           - or maybe a range of thresholds ex. (0.8, 0.9, 0.95, 0.99)
#   - how many false positives are found
#   - how accurate are the weights
#       - mse, mae, rmse?
##


class MethodEvaluator:
    def __init__(self) -> None:
        #TODO: directory for data is static
        self.dataPath = os.path.dirname(__file__) + "/data"
        self.GRCh37 , self.GRCh38 = self._getCOSMICData()
        self.knownsig = {i:{"cos":0,"latent":""} for i in self.GRCh37.columns.to_list()}
        self.verbose = 1
        self.textfilepath= "system"
        self.outerrun = 0

    def _getCOSMICData(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not os.path.exists(self.dataPath):
            os.mkdir(self.dataPath)
        if not os.path.exists(
            self.dataPath + "/COSMIC_v3.4_SBS_GRCh37.txt"
        ) or not os.path.exists(
            self.dataPath + "/COSMIC_v3.4_SBS_GRCh38.txt"
            ):
            print("Downloading COSMIC data...")
            urlretrieve(
                "https://cog.sanger.ac.uk/cosmic-signatures-production/documents/COSMIC_v3.4_SBS_GRCh37.txt",
                self.dataPath + "/COSMIC_v3.4_SBS_GRCh37.txt",
            )
            urlretrieve(
                "https://cog.sanger.ac.uk/cosmic-signatures-production/documents/COSMIC_v3.4_SBS_GRCh38.txt",
                self.dataPath + "/COSMIC_v3.4_SBS_GRCh38.txt",
            )
            
            print("Done")

    

        GRCh37 = pd.read_table(
            self.dataPath + "/COSMIC_v3.4_SBS_GRCh37.txt", index_col=0
        )
        GRCh38 = pd.read_table(
            self.dataPath + "/COSMIC_v3.4_SBS_GRCh38.txt", index_col=0
        )

        return GRCh37, GRCh38
  

    def _cosineSimilarity(self, a: np.ndarray, b: np.ndarray):
        result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        if np.isnan(result):
            return 0
        return result
    
    def _print(self,text)->None:
        Sprint.sprint(text,self.verbose,self.textfilepath)
        
    def _makeCosineSimilarityMatrix(
        self,
        signatures: pd.DataFrame,
        knownSignatures: pd.DataFrame,
    ):
        
        over90 : list[list] = [[] for _ in range(signatures.shape[1])]
        ownsig = {i:{"cos":0,"known":""} for i in signatures.columns.to_list()}
        knownsig = {i:{"cos":0,"latent":"","index":0} for i in knownSignatures.columns.to_list()}
        signatures_numpy = signatures.to_numpy()
        knownSignatures_numpy = knownSignatures.to_numpy()
        cosMatrix = np.zeros((signatures.shape[1], knownSignatures_numpy.shape[1]))
        
        for i in range(signatures_numpy.shape[1]):
            for j in range(knownSignatures_numpy.shape[1]):
                cosMatrix[i, j] = self._cosineSimilarity(
                    signatures_numpy[:, i], knownSignatures_numpy[:, j]
                )
                if cosMatrix[i,j ] >= 0.90:
                    over90[i].append(j)
        
        self.cosMatrix = cosMatrix
        cosMatrixSorted =  np.sort(cosMatrix, axis=1)
        cosMatrixIndexSorted = np.argsort(cosMatrix, axis=1)
        for i in range(0,cosMatrixSorted.shape[0]):
            for j in range(cosMatrixSorted.shape[1]-1,-1,-1):
        
                if knownsig[knownSignatures.columns[cosMatrixIndexSorted[i,j]]]["latent"] == "":
                    knownsig[knownSignatures.columns[cosMatrixIndexSorted[i,j]]]["cos"] = cosMatrixSorted[i,j]
                    knownsig[knownSignatures.columns[cosMatrixIndexSorted[i,j]]]["latent"] = signatures.columns[i]
                    knownsig[knownSignatures.columns[cosMatrixIndexSorted[i,j]]]["index"] = i
                    ownsig[signatures.columns[i]]["cos"] = cosMatrixSorted[i,j]
                    ownsig[signatures.columns[i]]["known"] = knownSignatures.columns[cosMatrixIndexSorted[i,j]]
                    break

                elif knownsig[knownSignatures.columns[cosMatrixIndexSorted[i,j]]]["cos"] < cosMatrixSorted[i,j]:
                    knownsig[knownSignatures.columns[cosMatrixIndexSorted[i,j]]]["cos"] = cosMatrixSorted[i,j]
                    knownsig[knownSignatures.columns[cosMatrixIndexSorted[i,j]]]["latent"] = signatures.columns[i]
                    lastindex = knownsig[knownSignatures.columns[cosMatrixIndexSorted[i,j]]]["index"]
                    knownsig[knownSignatures.columns[cosMatrixIndexSorted[i,j]]]["index"] = i
                    temp = signatures.columns[i]
                    last_key = cosMatrixIndexSorted.shape[1]-1
                    
                    while last_key != -1: 
                        # print(ownsig[signatures.columns[test_index_sorted[last_key,lastused]]],test[last_key,lastused],last_key,lastused)
                        if knownsig[knownSignatures.columns[cosMatrixIndexSorted[lastindex,last_key]]]["latent"] == "":
                            knownsig[knownSignatures.columns[cosMatrixIndexSorted[lastindex,last_key]]]["cos"] = cosMatrixSorted[lastindex,last_key]
                            knownsig[knownSignatures.columns[cosMatrixIndexSorted[lastindex,last_key]]]["latent"] = signatures.columns[lastindex]
                            knownsig[knownSignatures.columns[cosMatrixIndexSorted[lastindex,last_key]]]["index"] = lastindex
                            ownsig[signatures.columns[lastindex]]["cos"] = cosMatrixSorted[lastindex,last_key]
                            ownsig[signatures.columns[lastindex]]["known"] = knownSignatures.columns[cosMatrixIndexSorted[lastindex,last_key]]
                            break
                        elif knownsig[knownSignatures.columns[cosMatrixIndexSorted[lastindex,last_key]]]["cos"] < cosMatrixSorted[lastindex,last_key]:

                            ownsig[signatures.columns[lastindex]]["cos"] = cosMatrixSorted[lastindex,last_key]
                            ownsig[signatures.columns[lastindex]]["known"]= knownSignatures.columns[cosMatrixIndexSorted[lastindex,last_key]]

                            knownsig[knownSignatures.columns[cosMatrixIndexSorted[lastindex,last_key]]]["cos"] = cosMatrixSorted[lastindex,last_key]
                            knownsig[knownSignatures.columns[cosMatrixIndexSorted[lastindex,last_key]]]["latent"] = signatures.columns[lastindex]
                            templastindex = knownsig[knownSignatures.columns[cosMatrixIndexSorted[i,j]]]["index"]
                            knownsig[knownSignatures.columns[cosMatrixIndexSorted[lastindex,last_key]]]["index"] = lastindex
                            last_key = cosMatrixIndexSorted.shape[1]
                            lastindex = templastindex
                            
                            
                        last_key = last_key - 1
                        if last_key == -1:
                            ownsig[signatures.columns[lastindex]]["cos"] = 0
                            ownsig[signatures.columns[lastindex]]["known"] = ""

                    ownsig[signatures.columns[i]]["cos"] = cosMatrixSorted[i,j]
                    ownsig[signatures.columns[i]]["known"] = knownSignatures.columns[j]
                    break

                
        self.knownsig = knownsig
        return cosMatrix

        
    def _evaluateAgainstKnownSignatures(
        self,
        signatures: pd.DataFrame,
        knownSignatures: pd.DataFrame,
    ):
        # self._deeperEvaluateAgainstKnownSignatures(signatures, knownSignatures)
        self.cosineSimilarityMatrix = self._makeCosineSimilarityMatrix(
            signatures, knownSignatures
        )
        
        # count how many rows have a column over 0.8 and 0.95
        listofnum = np.arange(0.85,1,0.01)
        numcos = []
        for i in listofnum:
            numcos.append(np.sum(np.sum(self.cosineSimilarityMatrix >= i, axis=1) >= 1))
        return numcos

    def _checkDuplicates(self, signatures: pd.DataFrame, knownSignatures: pd.DataFrame):
        #TODO problem by using hungarian algorithm is that he can choose a suboptimal solution for a sig at expense of the total score
        # hungarian algorithm to find the best match
        self.hungarian_row_ind, self.hungarian_col_ind = linear_sum_assignment(
            self.cosineSimilarityMatrix, maximize=True
        )

        # best match cosin values
        hungarianMatch = self.cosineSimilarityMatrix[
            self.hungarian_row_ind, self.hungarian_col_ind
        ]

        self.bestMatchCosin = [
            (np.round(i,6), j, k)
            for i, j, k in zip(
                hungarianMatch,
                signatures.columns[self.hungarian_row_ind],
                knownSignatures.columns[self.hungarian_col_ind],
            )
        ]

    def _evaluateAgainstKnownWeights(
        self,
        weights: pd.DataFrame,
        knownWeights: pd.DataFrame,
    ):
        # normalize weights
        weights = weights.div(weights.sum(axis=1), axis=0).to_numpy()
        knownWeights = knownWeights.div(knownWeights.sum(axis=1), axis=0).to_numpy()

        # compare (mse, mae, rmse) weights with known weights, given the index
        mse = []
        mae = []
        rmse = []
        for i in range(weights.shape[1]):
            mse.append(
                np.sum(
                    (
                        weights[self.hungarian_row_ind, i]
                        - knownWeights[self.hungarian_col_ind, i]
                    )
                    ** 2
                )
            )
            mae.append(
                np.sum(
                    np.abs(
                        weights[self.hungarian_row_ind, i]
                        - knownWeights[self.hungarian_col_ind, i]
                    )
                )
            )
            rmse.append(np.sqrt(mse[-1]))

        return np.average(mse), np.average(mae), np.average(rmse)


    def COSMICevaluate(
        self,
        signatures: pd.DataFrame | str,
        injectionedSignatures = pd.DataFrame | None,
        GRCh: Literal["GRCh37", "GRCh38"] | pd.DataFrame = "GRCh37",
        
    ):
        if isinstance(signatures, str):
            if re.search(r"\.tsv$", signatures):
                signatures = pd.read_csv(signatures, sep="\t", index_col=0)
            else:
                signatures = pd.read_csv(signatures, index_col=0)
        self.indexlist = signatures.columns.to_list()
        # Signatures found by the method
        numFoundSig = signatures.shape[1]

        if isinstance(GRCh,pd.DataFrame):
            knownSignatures = GRCh
        elif GRCh == "GRCh37":
            knownSignatures = self.GRCh37
        elif GRCh == "GRCh38":
            knownSignatures = self.GRCh38
        self.collist = knownSignatures.columns.to_list()
        # Known signatures found by the method
        numFoundCos = self._evaluateAgainstKnownSignatures(
            signatures, knownSignatures
        )

        # check for duplicate signatures
        self._checkDuplicates(signatures, knownSignatures)
        self.bestMatchCosintosave = self.bestMatchCosin.copy()
        numBest95 = len([x[0] for x in self.bestMatchCosin if x[0] >= 0.95])
        numBest99 = len([x[0] for x in self.bestMatchCosin if x[0] >= 0.99])
        # print(self.bestMatchCosin)
        if __name__ == "__main__":
            print(f"Signatures found: {numFoundSig}")
            print(f"Signatures with cosin > 0.85: {numFoundCos[0]}")
            print(f"Signatures with cosin > 0.95: {numFoundCos[9]}")
            print(f"Best match cosin > 0.95: {numBest95}")
            print(f"Best match cosin > 0.99: {numBest99}")
            print(f"Best match cosin: {self.bestMatchCosin}")
        bestMatchCosin = [x for x in self.bestMatchCosin if x[0] >= 0.60]
        if isinstance(injectionedSignatures, pd.DataFrame):
            a = self._evaluateAgainstKnownSignatures(
            signatures, injectionedSignatures
            )
            self._checkDuplicates(
                signatures, injectionedSignatures
            )
            injectionedSignatures = self.bestMatchCosin
        else:
            injectionedSignatures = (0,"0","0")
        returnlist =[]
        returnlist.append(numFoundSig)
        returnlist.extend(numFoundCos)
        returnlist.append(numBest95)
        returnlist.append(numBest99)
        returnlist.append(injectionedSignatures)
        returnlist.append(bestMatchCosin)
        
        return returnlist

    def evaluate(
        self,
        signatures: pd.DataFrame | str,
        weights: pd.DataFrame | str,
        knownSignatures: pd.DataFrame | str,
        knownWeights: pd.DataFrame | str,
    ):
        if isinstance(signatures, str):
            if re.search(r"\.(tsv)|(txt)$", signatures):
                signatures = pd.read_csv(signatures, sep="\t", index_col=0)
            else:
                signatures = pd.read_csv(signatures, index_col=0)
        if isinstance(weights, str):
            if re.search(r"\.(tsv)|(txt)$", weights):
                weights = pd.read_csv(weights, sep="\t", index_col=0)
            else:
                weights = pd.read_csv(weights, index_col=0)
        if isinstance(knownSignatures, str):
            if re.search(r"\.(tsv)|(txt)$", knownSignatures):
                knownSignatures = pd.read_csv(knownSignatures, sep="\t", index_col=0)
            else:
                knownSignatures = pd.read_csv(knownSignatures, index_col=0)
        if isinstance(knownWeights, str):
            if re.search(r"\.(tsv)|(txt)$", knownWeights):
                knownWeights = pd.read_csv(knownWeights, sep="\t", index_col=0)
            else:
                knownWeights = pd.read_csv(knownWeights, index_col=0)

        # Signatures found by the method
        numFoundSig = signatures.shape[1]

        if weights.shape[0] != numFoundSig:
            weights = weights.T

        assert weights.shape[1] == knownWeights.shape[1]

        # Known signatures found by the method
        numFoundCos80, numFoundCos95 = self._evaluateAgainstKnownSignatures(
            signatures, knownSignatures
        )

        # check for duplicate signatures
        self._checkDuplicates(signatures, knownSignatures)

        numBest95 = len([x[0] for x in self.bestMatchCosin if x[0] >= 0.95])
        numBest99 = len([x[0] for x in self.bestMatchCosin if x[0] >= 0.99])

        # accuracy of weights
        mse, mae, rmse = self._evaluateAgainstKnownWeights(
            weights,
            knownWeights,
        )

        if __name__ == "__main__":
            print(f"Signatures found: {numFoundSig}")
            print(f"Signatures with cosine > 0.8: {numFoundCos80}")
            print(f"Signatures with cosine > 0.95: {numFoundCos95}")
            print(f"Best match cosine > 0.95: {numBest95}")
            print(f"Best match cosine > 0.99: {numBest99}")
            print(f"Best match cosine: {self.bestMatchCosin}")
            print(f"Weight error (MSE): {mse}")
            print(f"Weight error (MAE): {mae}")
            print(f"Weight error (RMSE): {rmse}")

        return (
            numFoundSig,
            numFoundCos80,
            numFoundCos95,
            numBest95,
            numBest99,
            self.bestMatchCosin,
            mse,
            mae,
            rmse,
        )


if __name__ == "__main__":
    evaluator = MethodEvaluator()
    # results = evaluator.evaluate(
    #     "sigGen/datasetOut/sigMatrix.csv",
    #     "sigGen/datasetOut/weights.csv",
    #     "sigGen/datasetOut/sigMatrix.csv",
    #     "sigGen/datasetOut/weights.csv",
    # )
    results = evaluator.COSMICevaluate("eval/nmf_output/signatures.tsv", GRCh="GRCh37")
    print(results)
