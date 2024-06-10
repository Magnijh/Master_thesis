from SEEF import SEEF
import pickle
import os
import sys
from itertools import product
from AE.ptAE import mseAE
from plotting import plottingdata,injectionplotting

def readcheckpoint(part:str) -> dict:
    with open(f"checkpoints/checkpoint{part}.pickle","rb") as f:
        data= f.read()
    d = pickle.loads(data) 
    return d
 
   
def writecheckpoint(d:dict,part:str):
    with open(f"checkpoints/checkpoint{part}.pickle","wb") as f:
        pickle.dump(d,f)


def ischeckpoint(part:str) -> bool:
    return os.path.exists(f"checkpoints/checkpoint{part}.pickle")


def breastcheckbenchmark(hyperparameters:list,part:str):
    seef = SEEF("datasets/GEL/catalogues_Breast_SBS.tsv" ,
                mseAE,
                "outputbenchmark",
                latents=200,
                threshold=0.01,
                verbose=1,
                silhouette="cosine",
                cluster_method="kmeans",
                benchmark=True,
                clusterlist=["kmeans","cosine"],
                silhouettelist=["cosine","euclidean"],
                type_clusteringlist=["multi","single"],
                alphalist=[1.0],
                )

    seef._job_meta()
        
    for _ in range(len(hyperparameters)):
        seef.noise = hyperparameters[0][0]  
        seef.bootstrap = hyperparameters[0][1]
        seef.threshold = hyperparameters[0][2]
        seef.latents = hyperparameters[0][3]
        seef.injectionprocent = hyperparameters[0][4] 
        hyperparameters = hyperparameters[1:]
        for i in range(0,1):
            seef.outerrun = i
            seef.run()
        seef.mergerunresults()
        writecheckpoint(hyperparameters,part)    
    seef.mergeallresults()
    
    
    
def breastcheck():
    seef = SEEF("datasets/GEL/catalogues_Breast_SBS.tsv" ,
                mseAE,
                "output",
                latents=250,
                threshold=0.09,
                verbose=1,
                silhouette="cosine",
                cluster_method="kmeans",
                type_clustering="multi",
                benchmark=False,
                alpha=0.6,
                
                )        

    seef._job_meta()
    for i in range(0,3):
        seef.outerrun = i
        seef.run()
        seef.save_all_results("real","runs")
           
    
    
if __name__ == "__main__":
    args = sys.argv
    threshold = [0,0.001,0.003,0.005,0.007,0.009,0.01]
    if len(args) > 1:
        threshold = [threshold[int(args[1])]]
    part = args[1] if len(args) > 1 else "default"
    
    injection=[0,1,5,10]
    if len(args) > 2:
        injection = [injection[int(args[2])]]
        part = args[1]+args[2]
    
    noise= [0.0]
    bootstrap=[False]
    latents = [15,10]
    listoflist = [noise,bootstrap,threshold,latents,injection]
    hyperparameters = []
    for i in product(*listoflist):
        hyperparameters.append(i)
    
    if part != "default": 
        checkpointbool = ischeckpoint(part)
        if checkpointbool:
            hyperparameters = readcheckpoint(part)
 
    
    # breastcheckbenchmark(hyperparameters,part)
    plottingdata("outputbenchmark","testtitle","testxtitle")
    injectionplotting("outputbenchmark","testtitle",)