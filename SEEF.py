import os
from typing import Callable
import psutil
from eval.cluster import AE_cluster
from eval.method_evaluation import MethodEvaluator
from sigGen.synthetic_dataset import create_dataset
from sigGen.injectData import injectDataset
import pandas as pd
import numpy as np
import Sprint
import datetime
import torch
import platform,subprocess
import time

class SEEF():
    def __init__(
        self,
        dataset: str,
        method: Callable[[pd.DataFrame, int], tuple[np.ndarray, np.ndarray, float]],
        outputfolder:str,
        runs: int = 10,
        latents: int = 20,
        threshold: float = 0.0,
        bootstrap: bool = False,
        injectionprocent : int = 0,
        noise : float = 0.0,
        verbose : int = 1,
        outerrun : int = 1,
        cluster_method :str = "kmeans",
        silhouette : str = "cosine",
        type_clustering : str = "multi",
        alpha : float = 0.6,
        benchmark: bool = False,
        refsig: pd.DataFrame | str = "GRCh37",
        clusterlist = ["kmeans","cosine"],
        silhouettelist = ["cosine","euclidean"],
        type_clusteringlist = ["multi","single"],
        alphalist: list = [0.6,0.7,0.8,0.9,1.0],
        

    ):
        """
        SEEF class is the main class for the SEEF project. It is used to run the clustering and evaluation of the method.
        datasets should point to the dataset that should be used for the clustering and evaluation.
        method is the method that should be used for the clustering.
        outputfolder is the folder where the results should be saved.
        runs is the number of runs the method extraction should be run.
        latent is the number of latents that should used in the autoencoder.
        threshold is what cut of for mutation is.
        bootstrap is if bootstrap should be used.
        injectionprocent is the percentage of the dataset that should be injected with a custom made signature.
        noise is the noise that should be added to the dataset when running method extraction on it.
        outerrun is the current number of run is on, this is used to save the results in the correct folder.
        cluster_method is the method that should be used for clustering.
        silhouette is the metric that should be used for silhouette score.
        type_clustering is the type of clustering that should be used.
        alpha is the alpha value that should be used in the clustering.
        benchmark is if the method should be benchmarked this is based on running multiple different post-process values.
        refsig is the reference signature that should be used for the evaluation str is for COSMIC, and dataframe for other refsig.
        clusterlist is the list of cluster methods that should be used in the benchmark.
        silhouettelist is the list of silhouette metrics that should be used in the benchmark.
        type_clusteringlist is the list of type clustering that should be used in the benchmark.
        alphalist is the list of alpha values that should be used in the benchmark.
        """
        
        
        
        
        #Init all properties from this class and the other two classes Eval and Clustering
       
        self.Evaluator = MethodEvaluator()
        self._cluster_method = cluster_method
        self._silhouette = silhouette
        self._type_clustering = type_clustering
        self.Clustering = AE_cluster(dataset,method,
                                    outputfolder,runs,
                                    latents,threshold,
                                    bootstrap,
                                    injectionprocent,noise,
                                    cluster_method,silhouette,
                                    type_clustering,benchmark,
                                    alpha
                                    )
        
        self._dataset = dataset.replace("\\","/")
        self._method = method
        self._runs = runs 
        self._latents = latents
        self._threshold = threshold
        self._bootstrap = bootstrap
        self._results = []
        self._runresults = []
        self._benchresults = []
        self._outputfolder = outputfolder
        self._noise = noise
        
        self._injectionprocent = injectionprocent
        self._injectionRunned = False
        self._path = self._outputfolder + f"/results/{self._method.__name__}_{self._threshold}_{self._bootstrap}_{self.injectionprocent}_{self.noise}"
        self.textfilepath = "system"
        self.verbose = verbose
        self.Clustering.verbose = verbose
        self.Evaluator.verbose = verbose
        self.outerrun = outerrun
        self.Clustering.outerrun = outerrun
        self.Evaluator.outerrun = outerrun
        self.knownsig = []
        self.benchmark = benchmark
        self._keepdf = self.Clustering.df.copy()
        self.refsig = refsig
        self.clusterlist = clusterlist
        self.silhouettelist = silhouettelist
        self.type_clusteringlist = type_clusteringlist
        self.alphalist = alphalist
    
    #We need to use property obsever to update the Clustering object when the attributes are changed   
    #all attributes needs property and a setter to make this 
    @property
    def type_clustering(self) -> str:
        return self._type_clustering
    
    @type_clustering.setter
    def type_clustering(self, value: str) -> None:
        self.Clustering.type_clustering = value
        self._type_clustering = value
    
    @property
    def cluster_method(self) -> str:
        return self._cluster_method
    
    @cluster_method.setter
    def cluster_method(self, value: str) -> None:
        self.Clustering.cluster = value
        self._cluster_method = value
        
    @property
    def silhouette(self) -> str:
        return self._silhouette
    
    @silhouette.setter
    def silhouette(self, value: str) -> None:
        self.Clustering.silhouette_metric = value
        self._silhouette = value
    
    @property
    def outerrun(self) -> int:
        return self._outerrun
    
    @outerrun.setter
    def outerrun(self, value: int) -> None:
        self.Clustering.outerrun = value
        self.Evaluator.outerrun = value
        self._outerrun = value
    
    @property
    def noise(self) -> float:
        return self._noise
    
    @noise.setter
    def noise(self, value: float) -> None:
        self.Clustering.noise = value
        self._noise = value
    
    
    @property
    def  dataset(self) -> str:
        return self._dataset
    
    @dataset.setter
    def dataset(self, value: str) -> None:
        self.Clustering.dataset = value.replace("\\","/")
        self._dataset = value.replace("\\","/")
    
    @property
    def method(self) -> Callable[[pd.DataFrame, int], tuple[np.ndarray, np.ndarray, float]]:
        return self._method
    
    @method.setter
    def method(self, value: Callable[[pd.DataFrame, int], tuple[np.ndarray, np.ndarray, float]]) -> None:
        self.Clustering.method = value
        self._method = value
    
    
    @property
    def runs(self) -> int:
        return self._runs
    
    @runs.setter
    def runs(self, value: int) -> None:
        self.Clustering.runs = value
        self._runs = value
    
    @property
    def latents(self) -> int:
        return self._latents
    
    @latents.setter
    def latents(self, value: int) -> None:
        self._latents = value
        self.Clustering.latents = value
        self.Clustering.components = range(2,value)
    
    @property
    def threshold(self) -> float:
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float) -> None:
        self.Clustering.threshold = value
        self._threshold = value
    
    @property
    def bootstrap(self) -> bool:
        return self._bootstrap
    
    @bootstrap.setter
    def bootstrap(self, value: bool) -> None:
        self.Clustering.bootstrap = value
        self._bootstrap = value
    
    @property
    def injectionprocent(self) -> int:
        return self._injectionprocent
    
    @injectionprocent.setter
    def injectionprocent(self, value: int) -> None:
        self._injectionprocent = value
        self.injectionRunned = False
        if self._injectionprocent > 0:
            self.Clustering.df = self._keepdf.copy()
    
    @property
    def injectionRunned(self) -> bool:
        return self._injectionRunned
    
    @injectionRunned.setter
    def injectionRunned(self, value: bool) -> None:
        self._injectionRunned = value
        
    @property
    def path(self) -> str:
        return self._path
    
    @path.setter
    def path(self, value: str) -> None:
        self._path = value
        
        
    def _cpu(self)->str:
        returnstr = ""
        if os.name == 'nt':
            returnstr = "model name :" + subprocess.check_output(["wmic","cpu","get", "name"]).strip().decode("utf-8").split("\n")[1]
        else:
            command = 'cat /proc/cpuinfo | grep  "model name" |uniq'
            returnstr = subprocess.check_output(command, shell=True).strip().decode('utf-8')
        return returnstr
    
    def _run_meta(self) -> None:
        self._print("---------------------------------")
        self._print("Run meta data")
        self._print(f"Dataset : {self._dataset}")
        self._print(f"Method : {self._method.__name__}")
        self._print(f"Runs : {self._runs}")
        self._print(f"Latents : {self._latents}")
        self._print(f"Threshold : {self._threshold}")
        self._print(f"Bootstrap : {self._bootstrap}")
        self._print(f"Injectionprocent : {self._injectionprocent}")
        self._print(f"Noise : {self._noise}")
        self._print(f"Cluster method : {self._cluster_method}")
        self._print(f"Silhouette metric : {self._silhouette}")
        self._print(f"Type clustering : {self._type_clustering}")
        self._print("---------------------------------")
    
    def _job_meta(self)-> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._print("---------------------------------")
        self._print("Job meta data")
        self._print(f"Python version : {platform.python_version()}")
        self._print(f"time : {datetime.datetime.now()}")
        if device.type != "cuda":
            self._print(f"GPU : {torch.cuda.get_device_name(0)}")
            self._print(f"GPU count : {torch.cuda.device_count()}")
            self._print(f"GPU memory : {torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)} GB")
        self._print(f"{self._cpu()}")
        self._print(f"CPU architectures : {platform.uname().machine}")
        self._print(f"CPU core count : {os.cpu_count()}")
        self._print(f"RAM : {str(round(psutil.virtual_memory().total / (1024.0 **3),3))} GB")
        self._print("---------------------------------")

    def _dfcolumns(self,datasettype: str = "real") -> list:
        if datasettype == "real":
            return ["run","method","latents","runnr","cut_off","bootstrap","injection_percentage","noise","rows","columns",
           "cut_C>A","cut_C>G" ,"cut_C>T","cut_T>A","cut_T>C","cut_T>G","cluster_method","silhouette_metric",
                "optimal_selection","alpha","found", ">0.85",">0.86",">0.87",">0.88",">0.89",">0.90",">0.91",
                 ">0.92",">0.93",">0.94", ">0.95",">0.96",">0.97",">0.98",">0.99",">1",        
                "best>0.95","best>0.99","injections_match", "match"]
        elif datasettype == "synthetic":
            return [
                "method","cut_off","bootstrap",
                "rows","columns","found",">0.8",">0.90"
                ">0.95","best>0.95","best>0.99",
                "match","mse","mae","rmse",
                ]
        else:
            return None
    
    def _save_results(self, datasettype: str = "real" ) -> pd.DataFrame:
        columns = self._dfcolumns(datasettype)
        return pd.DataFrame(
            [self.result],
            columns=columns,
            )   
            
    def COSMIC_evaluate(self, signature_file:str, genome_build:str|pd.DataFrame) -> None:
        t1 = time.time()
        self._print("Running evaluating")
        result = self._preparameter()
        result.append(self.Clustering.cluster)
        result.append(self.Clustering.silhouette_metric)
        result.append(self.Clustering.type_clustering)
        result.append(self.Clustering.alpha)
        
        if self.injectionprocent > 0:
            results2 =self.Evaluator.COSMICevaluate(
                                    signature_file,self.sigMatrix, genome_build)
        else:
            results2 = self.Evaluator.COSMICevaluate(signature_file,None,genome_build)
        self.result = result + [x for x in results2]
         
        df = pd.DataFrame.from_dict(self.Evaluator.knownsig,orient="index")[["cos","latent"]]   
        df.loc[df["cos"]>0].to_csv(self.path+"/run_"+str(self.outerrun)+"/knownsig.tsv",sep="\t",index_label="signature")
        df["runnr"]=self.outerrun
        self.knownsig.append(df)
        self._save_results("real").sort_values(by=["method","bootstrap"
                    ,"cut_off"],ascending=[False,False,True]).to_csv(self.path+"/run_"+str(self.outerrun)+"/result.tsv",index=False,sep="\t")
        self._runresults.append(self.result)
        self._results.append(self.result)
        datasetname = self._dataset.split("/")[-1].split(".")[0]
        pd.DataFrame(self.Evaluator.cosMatrix,columns=self.Evaluator.GRCh38.columns.to_list()
                     ).to_csv(self.path+"/run_"+str(self.outerrun)+"/cosineMatrix.tsv",sep="\t",index_label="latent")
        self._benchresults.append([datasetname]+self.result)
        tempdf =pd.DataFrame(self.Evaluator.bestMatchCosintosave,columns=["cos","latent","sig"])
        tempdf["runnr"]=self.outerrun
        tempdf.to_csv(self.path+"/run_"+str(self.outerrun)+"/hungarianMatchCosine.tsv",sep="\t",index=False)
        
        
        self._print(f"Time elapsed: {time.time()-t1}")
    
        
    def _create_injection_dataset(self):
        localpath = self.path + "/injections"
        tempp = 0
        if not os.path.exists(localpath):
            os.makedirs(localpath)
        if os.path.exists(localpath+"/injectedData.txt"):
            self.injection_df = pd.read_table(localpath+"/injectedData.txt",index_col=0)
            tempp +=1
        if os.path.exists(localpath+"/sigMatrix.csv"):
            self.sigMatrix =pd.read_table(localpath+"/sigMatrix.csv",index_col=0)
            tempp +=1
        if tempp<1:
            self.Clustering.df = self._keepdf.copy()
            self.df, self.injection_df, self.sigMatrix, self.weights = injectDataset(self._injectionprocent,self.Clustering.df,localpath,nrOfSignatures=1)

    def _print(self,text):
        Sprint.sprint(text,self.verbose,self.textfilepath)
    
    def _preparameter(self):
        result = [self.outerrun,self.method.__name__,self.latents,self.runs,self.threshold,self.bootstrap,self.injectionprocent,
                    self.noise,
                    self.Clustering._df_cutted.shape[0],self.Clustering._df_cutted.shape[1],
                    self.Clustering.count_mut["C>A"],self.Clustering.count_mut["C>G"],
                    self.Clustering.count_mut["C>T"],self.Clustering.count_mut["T>A"],
                    self.Clustering.count_mut["T>C"],self.Clustering.count_mut["T>G"],
                    ]
        columns = self._dfcolumns("real")
        columns = columns[:len(result)]
        pd.DataFrame([result],columns=columns).to_csv(self.path+"/run_"+str(self.outerrun)+"/preparameter.tsv",sep="\t",index=False)
        return result
        
    def run(self):
        self._run_meta()
        a = { w:[] for w in self.Evaluator.GRCh38.columns.to_list()}
        if self.benchmark:
            self.path = self._outputfolder +f"/results/{self._method.__name__}_L{self.latents}_T{self.threshold}_B{self.bootstrap}_I{self.injectionprocent}_N{self.noise}"
        else:
            self.path = self._outputfolder +f"/results/{self._method.__name__}_L{self.latents}_T{self.threshold}_B{self.bootstrap}_I{self.injectionprocent}_N{self.noise}_CM{self.cluster_method}_SM{self._silhouette}_TC{self._type_clustering}"
        self.Clustering.path = self.path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        if self.injectionprocent > 0 and self.injectionRunned == False:
            self._create_injection_dataset()
            self.injectionRunned = True
        if self.injectionprocent > 0:
            self.Clustering.df = self.injection_df.copy()
        self.Clustering.run(clusterlist=self.clusterlist,
                            silhouettelist=self.silhouettelist,
                            type_clusteringlist=self.type_clusteringlist,alphalist=self.alphalist)
        if self.benchmark:
            self._benchmark_eval()
            
        else:
            self.COSMIC_evaluate(self.path+"/run_"+str(self.outerrun)+"/signatures.tsv",self.refsig)
    
    def _benchmark_eval(self):
        print("Running benchmark eval")
        print(self.path)
        pre =self._preparameter()
        sigdirlist = sorted(os.listdir(self.path+"/run_"+str(self.outerrun)+"/signatures"))
        
        if not os.path.exists(self.path+"/run_"+str(self.outerrun)+"/cosine_matrix"):
            os.makedirs(self.path+"/run_"+str(self.outerrun)+"/cosine_matrix")
        if not os.path.exists(self.path+"/run_"+str(self.outerrun)+"/knownsigs"):
            os.makedirs(self.path+"/run_"+str(self.outerrun)+"/knownsigs")
        if not os.path.exists(self.path+"/run_"+str(self.outerrun)+"/hungarianMatchCosine"):
            os.makedirs(self.path+"/run_"+str(self.outerrun)+"/hungarianMatchCosine")    
    
        allresults = []
        for i in range(len(sigdirlist)):
            result = pre.copy()
            sigsplitted = sigdirlist[i].split(".")[0].split("_")
            
            result.append(sigsplitted[0])
            result.append(sigsplitted[1])
            result.append(sigsplitted[2])
            result.append(sigsplitted[3])
            
            results = self.Evaluator.COSMICevaluate(self.path+"/run_"+str(self.outerrun)+"/signatures/"+sigdirlist[i],None,self.refsig)
            cos_sim_df = pd.DataFrame(self.Evaluator.cosMatrix,columns=self.Evaluator.collist,index=self.Evaluator.indexlist)
            cos_sim_df.to_csv(self.path+"/run_"+str(self.outerrun)+"/cosine_matrix/"+sigdirlist[i],sep="\t",index_label="latent")
            
            result.extend([x for x in results])
            tempknowndf = pd.DataFrame.from_dict(self.Evaluator.knownsig,orient="index")[[
                "cos","latent"]]
            tempknowndf["runnr"]=self.outerrun
            tempknowndf.to_csv(self.path+"/run_"+str(self.outerrun)+"/knownsigs/"+sigdirlist[i],sep="\t",index_label="signature")
            temphungdf =pd.DataFrame(self.Evaluator.bestMatchCosintosave,columns=["cos","latent","sig"])
            temphungdf["runnr"]=self.outerrun
            temphungdf.to_csv(self.path+"/run_"+str(self.outerrun)+"/hungarianMatchCosine/"+sigdirlist[i],sep="\t",index=False)
            allresults.append(result)
        pd.DataFrame(allresults,columns=self._dfcolumns("real")).sort_values(by=["method","bootstrap"
                    ,"cut_off"],ascending=[False,False,True]).to_csv(
                        self.path+"/run_"+str(self.outerrun)+"/result.tsv",index=False,sep="\t")

            
          
    def create_synthetic_dataset(self, sigN:int, datapoints:int, signpath:str):
        self.synthetic_dataset,self.synthetic_sigMatrix, self.synthetic_weights = create_dataset(sigN,datapoints,signpath)
        
    def save_all_results(self, datasettype : str = "real",save_type:str="all") -> None:
        columns = self._dfcolumns(datasettype)
        
        pd.DataFrame(
            self._results if save_type == "all" else self._runresults,
            columns=columns,  
        ).sort_values(
                by=["method","bootstrap","cut_off"],
                ascending=[False,False,True]
                      ).to_csv(
                          self._outputfolder+"/results.tsv" if save_type == "all" else self.path+"/results.tsv",
                          index=False,sep="\t"
                          )
        if save_type != "all":  
                        
            df = pd.concat(self.knownsig)[["cos","latent","runnr"]]
            df.loc[df["cos"]>0].to_csv(
                        self._outputfolder+"/knownsig.tsv" if save_type == "all" else self.path+"/knownsigs.tsv",
                        index=True,sep="\t",index_label="signature")
        
    def save_bench_results(self,folder:str, datasettype : str = "real") -> None:
        columns = self._dfcolumns(datasettype)
        pd.DataFrame(
            self._benchresults,
            columns=columns,
            
        ).sort_values(by=["dataset","method","bootstrap"
                    ,"cut_off"],ascending=[True,False,False,True]).to_csv(folder+"results.tsv",index=False,sep="\t")

    def mergerunresults(self):
        listofdf = []
        listofruns = sorted(os.listdir(self.path))
        for run in listofruns:
            if "run" in run:
                listofdf.append(pd.read_csv(self.path+"/"+run+"/result.tsv",delimiter="\t"))
        df = pd.concat(listofdf)
        df.to_csv(self.path+"/results.tsv",index=False,sep="\t")
        
    def mergeallresults(self):
        listofdf = []
        listofruns = sorted(os.listdir(self._outputfolder+"/results"))
        for run in listofruns:
            listofdf.append(pd.read_csv(self._outputfolder+"/results/"+run+"/results.tsv",delimiter="\t"))
        pd.concat(listofdf).to_csv(self._outputfolder+"/results.tsv",index=False,sep="\t")