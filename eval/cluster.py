import numbers
import time
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from SigProfilerAssignment import Analyzer
from typing import Callable
import warnings
from numpy.random import Generator, PCG64
from functools import partial
from numbers import Number
import Sprint
import torch.multiprocessing as mp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
# Ignore Kmeans future warning
warnings.filterwarnings("ignore", category=FutureWarning)
# ignore NMF convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def clustering(runs:int,components:range,prelim_signatures,metric:str = "euclidean",cluster:str="kmeans",random_state:int =1)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    c_cluster_centroids = []
    c_silhouette_scores = []
    c_inertia_scores = []
    for j in components:
        # t1 = time.time()
        if cluster == "kmeans":
            km = KMeans(n_clusters=j,random_state=random_state,verbose=0,tol=0.000001).fit(prelim_signatures[runs])
            tempcentroids = km.cluster_centers_
            tempcentroids[tempcentroids<0]=0
            c_cluster_centroids.append(tempcentroids)
            c_silhouette_scores.append(
                silhouette_score(prelim_signatures[runs], km.labels_,metric=metric)
            )
            c_inertia_scores.append(km.inertia_)
        if cluster == "cosine":
            
            centroids,cosine_score,silhouette_scores = cosine_clustering(prelim_signatures[runs],j,random_state=random_state,metric=metric,runs=runs)
            c_cluster_centroids.append(centroids)
            c_silhouette_scores.append(silhouette_scores)
            c_inertia_scores.append(cosine_score)
            
            
    return c_cluster_centroids,c_silhouette_scores,c_inertia_scores,cluster
            
def multi_run(run:int, method, df:pd.DataFrame, components:int, bootstrap:bool,noise:float):
    if bootstrap:
        df = _MatLabBootstrapCancerGenomes(df)
    else:
        df = df
    W, H, l = method(df, components,noise)
    return W, H, l

def _MatLabBootstrapCancerGenomes(df:pd.DataFrame) -> pd.DataFrame:
    genomes = df.to_numpy()
    normGenomes = genomes / np.tile(np.sum(genomes, axis=0), (genomes.shape[0], 1))
    dataframe = pd.DataFrame()
    for i in range(genomes.shape[1]):
        dataframe[i]= np.transpose(np.random.multinomial(np.sum(genomes[:,i]), normGenomes[:,i],))
    dataframe.index = df.index
    return dataframe

class k_cluster:
    def __init__(
        self,
        dataset: str,
        method: Callable[[pd.DataFrame, int], tuple[np.ndarray, np.ndarray, float]],
        outputfolder:str,
        runs: int = 10,
        components: tuple = (2, 6),
        threshold: float = 0.00,
        bootstrap: bool = False,
        injectionprocent: int = 0,
        noise: float = 0.0,
        outerrun: int = 0,
        verbose: int = 1,
        cluster: str = "Kmeans",
        silhouette_metric: str = "euclidean",
        type_clustering:str = "multi",
        alpha: int = 1,
        benchmark: bool = False
    ):
        assert dataset != None, "<<"
        self._dataset = dataset
        #TODO dataset needs a getter and setter to be dynamic
        if re.search(r"\.(tsv)|(txt)$", dataset):
            self._df = pd.read_table(dataset, index_col=0)
        else:
            self._df = pd.read_csv(dataset, index_col=0)
        
        if self._df.index.name != "Type":
            self._df.index.name = "Type"
        self._df = self._df.sort_index()
        assert runs > 0, "runs must be greater at least 1"
        self.runs = runs
        assert components[0] > 1, "components[0] must be greater than 1"
        assert (
            components[1] > components[0]
        ), "components[1] must be greater than components[0]"
        self.components = range(components[0], components[1] + 1)
        #TODO assert for threshold
        assert isinstance(threshold,Number), "threshold must be a number" 
        assert 0 <= threshold <= 1, "threshold must be between 0 and 1"
        self.threshold = threshold
        assert isinstance(bootstrap,bool), "bootstrap must be a boolean"
        self.bootstrap = bootstrap
        assert isinstance(noise,Number), "noise must be a number"
        self.noise = noise
        assert callable(method), "method must be a function"
        self.method = method
        assert outputfolder != None, "outputfolder must be a string"
        self.output_path = outputfolder
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        self.path = self.output_path + f"/results/{self.method.__name__}_{self.threshold}_{self.bootstrap}_{injectionprocent}_{self.noise}"
        self.verbose = verbose
        self.textfilepath = "system"
        self.outerrun = outerrun
        mp.set_start_method("spawn")
        assert cluster in ["kmeans","cosine"], "cluster must be kmeans or cosine"
        self.cluster = cluster
        assert isinstance(silhouette_metric,str), "silhouette_metric must be a string"
        self.silhouette_metric = silhouette_metric
        assert type_clustering in ["single","multi"], "type_clustering must be single or multi"
        self.type_clustering = type_clustering
        assert isinstance(alpha,Number), "alpha must be an number"
        assert alpha > 0, "alpha must be greater than 0"
        self.alpha = alpha
        assert isinstance(benchmark,bool), "benchmark must be a boolean"
        self.benchmark = benchmark
        
            
    @property
    def dataset(self):
        return self._dataset
        
    @dataset.setter
    def dataset(self, value):
        if re.search(r"\.(tsv)|(txt)$", value):
            self._df = pd.read_csv(value, sep="\t", index_col=0)
        else:
            self._df = pd.read_csv(value, index_col=0)
        self._dataset = value
        
        
    @property
    def df(self):
        return self._df
    
    @df.setter
    def df(self, value):
        self._df = value
        
    def _print(self,text):
        Sprint.sprint(text,self.verbose,self.textfilepath)
        
    def run(self,clusterlist:list = ["kmeans","cosine"],
            type_clusteringlist:list = ["single","multi"],
            silhouettelist:list = ["cosine","euclidean"],
            alphalist:list=[0.6]):
        if not os.path.exists(self.path+"/run_"+str(self.outerrun)):
            os.makedirs(self.path+"/run_"+str(self.outerrun))
        self._print("Running cutting off")
        self._cut_off()
        t1 = time.time()
        self._print("Running method_cluster")
        self._method_iteration()
        self._print(f"Time elapsed: {time.time()-t1}")
        t1 = time.time()
        self._print("Finding signatures")
        if self.benchmark:
            self._cluster_signatures_benchmark(clusterlist,type_clusteringlist,silhouettelist,alphalist)
        else:
            self._cluster_signatures()
            
            self._print(f"Found {len(self.signatures)} signatures")
            self._print("Finding exposures")
            self._find_exposures()
        self._print(f"Time elapsed: {time.time()-t1}")
        
        
        
        
    def _cluster_signatures_benchmark(self,clusternames:list = ["kmeans","cosine"],
                                      type_clusteringlist:list = ["single","multi"]
                                      ,metric:list = ["cosine","euclidean"],
                                      alphalist:list=[1.0]):
        pass

    def _cut_off(self):
        self.df_cut_off : pd.DataFrame = self._df.loc[(self._df.sum(axis="columns")/self._df.sum().sum())<=self.threshold, :]
        
        self._count_mut_type_cut_off()
        self._df_cutted : pd.DataFrame = self._df.loc[(self._df.sum(axis="columns")/self._df.sum().sum())>self.threshold, :]
        self._df_cutted[self._df_cutted<0.000001]=0.000001
    
    def _count_mut_type_cut_off(self):
        self.count_mut = {"C>A":0, "C>G":0, "C>T":0, "T>A":0, "T>C":0, "T>G":0, "Indel":0, "Other":0}
        for i in self.df_cut_off.index.to_list():
            self.count_mut[i[2:5]]+=1
        self._print(self.count_mut)
        
    
    def _method_iteration(self):
        W_iterations = []
        H_iterations = []
        losses = []
        for i in self.components:
            W_concat = []
            H_concat = []
            loss = []

            for _ in range(self.runs):
                W, H, l = multi_run(0,self.method,self._df_cutted,i,self.bootstrap,self.noise)
                W_concat.append(W)
                H_concat.append(H)
                loss.append(l)
            W_iterations.append(np.stack(W_concat))
            H_iterations.append(np.stack(H_concat))
            losses.append(loss)

        self.prelim_signatures = [np.hstack(x).T for x in W_iterations]
        self.H_iterations = [np.hstack(x).T for x in H_iterations]
        self.avg_loss = [np.mean(x) for x in losses]

    def _cluster_signatures(self):
        pass

    def _find_exposures(self):
        # save signatures
        signatures = pd.DataFrame(self.signatures, columns=self._df_cutted.index).T
        signatures.columns = [f"Sig{i}" for i in range(1, signatures.shape[1] + 1)]
        self.signaturescut = pd.DataFrame(0,index=self.df_cut_off.index, columns=signatures.columns)
        signatures = pd.concat([signatures,self.signaturescut])
        signatures = signatures.reset_index(drop=False)
        signatures.sort_values(by="Type", inplace=True)
        signatures.set_index("Type", inplace=True)
        signatures.to_csv(self.path+"/run_"+str(self.outerrun) + f"/signatures.tsv", sep="\t")
        self.final_signatures = signatures
        # find exposures
        # Analyzer.cosmic_fit(
        #     samples=self.dataset,
        #     output=self.path+"/run_"+str(self.outerrun) + "/output_weights",
        #     signature_database=self.path+"/run_"+str(self.outerrun) + f"/signatures.tsv",
        #     input_type="matrix",
        # )  
              
    def _find_exposures_benchmark(self,cluster,type_clustering,metric,alpha):
        # save signatures
        if not os.path.exists(self.path+"/run_"+str(self.outerrun) + "/signatures"):
            os.makedirs(self.path+"/run_"+str(self.outerrun) + "/signatures")
        signatures = pd.DataFrame(self.signatures, columns=self._df_cutted.index).T
        signatures.columns = [f"Sig{i}" for i in range(1, signatures.shape[1] + 1)]
        
        signaturescut = pd.DataFrame(0,index=self.df_cut_off.index, columns=signatures.columns)
        signatures = pd.concat([signatures,signaturescut])
        signatures = signatures.reset_index(drop=False)
        signatures.sort_values(by="Type", inplace=True)
        signatures.set_index("Type", inplace=True)
        signatures.to_csv(self.path+"/run_"+str(self.outerrun) +"/signatures"+ f"/{cluster}_{type_clustering}_{metric}_{alpha}.tsv", sep="\t")
              
        
    def _plot_aux_loss(self, cluster_components, aux_loss, xlabel, ylabel):
        plt.figure(figsize=(10,6))
        plt.plot(cluster_components, aux_loss)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(self.path+"/run_"+str(self.outerrun) + "/aux_loss.png")
        
    
    def _enumerate_cluster(self, cluster_centroids, silhouette_scores, inertia_scores):      
        cluster_components = []
        for i, centroids in enumerate(cluster_centroids):
            silhouette_alone = silhouette_scores[i]
            inertia_alone = inertia_scores[i]
            if self.cluster == "kmeans":
                silhouette_hat = (
                    np.array(silhouette_alone) - np.mean(silhouette_alone)
                ) / np.std(silhouette_alone)
                inertia_hat = (np.array(inertia_alone) - np.mean(inertia_alone)) / np.std(
                    inertia_alone
                )
                aux_loss = inertia_hat - self.alpha * silhouette_hat
                argaux = np.argmin(aux_loss)
                
            if self.cluster == "cosine":
                aux_loss = [a+self.alpha*b for a,b in zip(inertia_alone,silhouette_alone)]
                argaux = np.argmax(aux_loss) 
                   
            cluster_components.append(argaux + self.components[0])
            cluster_centroids[i] = centroids[argaux]
            silhouette_scores[i] = silhouette_alone[argaux]
            inertia_scores[i] = inertia_alone[argaux]
        return cluster_centroids, silhouette_scores, inertia_scores, cluster_components
    
    def _enumerate_cluster_benchmark(self, cluster_centroids, silhouette_scores, inertia_scores,cluster,alpha):      
        cluster_components = []
        local_centroid = [[]]*len(cluster_centroids)
        local_silhouette = [[]]*len(cluster_centroids)
        local_inertia = [[]]*len(cluster_centroids)
        for i, centroids in enumerate(cluster_centroids):
            silhouette_alone = silhouette_scores[i]
            inertia_alone = inertia_scores[i]
            if cluster[i] == "kmeans":
                silhouette_hat = (
                    np.array(silhouette_alone) - np.mean(silhouette_alone)
                ) / np.std(silhouette_alone)
                inertia_hat = (np.array(inertia_alone) - np.mean(inertia_alone)) / np.std(
                    inertia_alone
                )
                aux_loss = inertia_hat - (alpha * silhouette_hat)
                argaux = np.argmin(aux_loss)
                
            if cluster[i] == "cosine":
                aux_loss = [a+(alpha*b) for a,b in zip(inertia_alone,silhouette_alone)]
                argaux = np.argmax(aux_loss) 
                   
            cluster_components.append(argaux + self.components[0])
            local_centroid[i] = centroids[argaux]
            local_silhouette[i] = silhouette_alone[argaux]
            local_inertia[i] = inertia_alone[argaux]
        return local_centroid, local_silhouette, local_inertia, cluster_components



class AE_cluster(k_cluster):
    def __init__(
        self,
        dataset: str,
        method: Callable[[int,pd.DataFrame, int,bool,int], tuple[np.ndarray, np.ndarray]],
        outputfolder:str,
        runs: int = 10,
        latents: int = 64,
        threshold: float = 0.0,
        bootstrap: bool = False,
        injectionprocent: float = 0.0,
        noise : float = 0.0,
        cluster : str = "kmeans",
        silhouette_metric: str = "euclidean",
        type_clustering: str = "multi",
        benchmark: bool = False,
        alpha: int = 1,
        multi_processnr: int = 20
    ):
        self.latents = latents
        self.multi_processnr = multi_processnr
        super().__init__(
            dataset, 
            method,
            outputfolder, 
            runs, 
            (2, latents - 1), 
            threshold, 
            bootstrap,
            injectionprocent,
            noise,
            cluster=cluster,
            silhouette_metric=silhouette_metric,
            type_clustering=type_clustering,
            alpha=alpha,
            benchmark=benchmark,
            )

    
    def _method_iteration(self):
        self.prelim_signatures = []
        self.H_iterations = []
        self.avg_loss = []
        
        pool_AE = partial(multi_run,method=self.method,df=self._df_cutted ,components=self.latents,bootstrap=self.bootstrap,noise=self.noise)
        
        with mp.Pool(self.multi_processnr) as pool:
            for W,H,l in pool.map(pool_AE, range(self.runs)):
                self.prelim_signatures.append(W.T)
                self.H_iterations.append(H)
                self.avg_loss.append(l)
                 
        if not os.path.exists(f"{self.path}/run_{self.outerrun}/prelim_signatures"):
            os.makedirs(f"{self.path}/run_{self.outerrun}/prelim_signatures")
        for i in range(self.runs):
            np.savetxt(f"{self.path}/run_{self.outerrun}/prelim_signatures/prelim_signatures_run{i}.csv",self.prelim_signatures[i],delimiter=",")
        if not os.path.exists(f"{self.path}/run_{self.outerrun}/exposure"):
            os.makedirs(f"{self.path}/run_{self.outerrun}/exposure")
        for i in range(self.runs):
            np.savetxt(f"{self.path}/run_{self.outerrun}/exposure/exposure_run{i}.csv",self.H_iterations[i],delimiter=",")
    
    def _latent_checker(self):
        component = self.latents
        for i in range(len(self.prelim_signatures)):
            arr = self.prelim_signatures[i]
            arr = arr[arr.sum(axis=1)>0]
            arr = arr/arr.sum(axis=1,keepdims=True)
            arr = np.unique(arr,axis=0)
            self.prelim_signatures[i] = arr
            
            if arr.shape[0]<component:
                component = arr.shape[0]
        return component
     
    def _cluster_signatures_benchmark(self,
                                      clusternames:list = ["kmeans","cosine"],
                                      type_clusteringlist:list = ["single","multi"]
                                      ,metric:list = ["cosine","euclidean"],
                                      alphalist:list=[1.0]):
        
        """
        clustering benchmark is used to find the best cluster for the signatures
        Clusternames: list of cluster methods to use implemented are ["kmeans","cosine"]
        type_clusteringlist: list of type clustering to use implemented are ["single","multi"]
        metric: list of metric to use calculate silhouette using sklearn, used are ["cosine","euclidean"]
        alphalist: list of alpha values to use for the benchmark used are [0.6,0.7,0.8,0.9,1.0]
        """
        component = self._latent_checker()
        component  = range(2,component)
        
        
        
        
        for clustered in clusternames:
            
            for metrici in metric:
                pool = mp.Pool(self.multi_processnr) #20 because both cluster will be run the same time
                pool_clustering = partial(clustering,prelim_signatures=self.prelim_signatures,
                                        components=component,metric=metrici,
                                        cluster=clustered,random_state=42)
                results_list = pool.map(pool_clustering, range(self.runs))
                pool.close()
                pool.join()
                clusterlist = []
                cluster_centroids = []
                silhouette_scores = []
                inertia_scores = []
                for i_cluster_centroids, i_silhouette_scores, i_inertia_scores,cluster in results_list:
                        
                    cluster_centroids.append(i_cluster_centroids)
                    silhouette_scores.append(i_silhouette_scores)
                    inertia_scores.append(i_inertia_scores)     
                    clusterlist.append(cluster)
                
                tempcluster = [clustered]*len(cluster_centroids)
                for alpha in alphalist:
                    a_inner_centroids, a_inner_silhouette, a_inner_inertia, a_cluster_components=self._enumerate_cluster_benchmark(cluster_centroids, silhouette_scores, inertia_scores,tempcluster,alpha)
                
                    for type_cluster in type_clusteringlist:
                    
                        if type_cluster == "single":
                            
                            
                            if clustered == "kmeans":
                                silhouette_hat = (
                                    np.array(a_inner_silhouette) - np.mean(a_inner_silhouette)
                                ) / np.std(a_inner_silhouette)
                                loss_hat = (self.avg_loss - np.mean(self.avg_loss)) / np.std(self.avg_loss)
                                aux_loss = loss_hat - alpha * silhouette_hat
                                self.signatures = a_inner_centroids[np.argmin(aux_loss)]
                            if clustered == "cosine":
                                aux_loss = [a+alpha*b for a,b in zip(a_inner_inertia,a_inner_silhouette)]
                                self.signatures = a_inner_centroids[np.argmax(aux_loss)]
                            self._find_exposures_benchmark(clustered,type_cluster,metrici,alpha)
                            # self._plot_aux_loss(
                            #     cluster_components,
                            #     aux_loss,
                            #     "components/clusters",
                            #     "Auxiliary loss",
                            # )
                            
                            
                        if type_cluster == "multi":
                            nparr = np.concatenate(a_inner_centroids,axis=0)
                            nparr = nparr/nparr.sum(axis=1,keepdims=True)  
                            arr = np.unique(nparr,axis=0)
                            o_cluster_centroids = []
                            o_silhouette_scores = []
                            o_inertia_scores = []
                            o_cluster_components = []
                            o_clusterlist = []
                            o_range = range(2,arr.shape[0])
                            larr = [arr]*1
                            pool = mp.Pool(1)
                            pool_clustering = partial(clustering,prelim_signatures=larr,
                                                        components=o_range,metric=metrici,
                                                        cluster=clustered,random_state=42,)
                            inner_resutls = pool.map(pool_clustering, range(1))
                            pool.close()
                            pool.join()
                            
                            for i_cluster_centroids, i_silhouette_scores, i_inertia_scores,cluster in inner_resutls:        
                                o_cluster_centroids.append(i_cluster_centroids)
                                o_silhouette_scores.append(i_silhouette_scores)
                                o_inertia_scores.append(i_inertia_scores)
                                o_clusterlist.append(cluster)
                                
                            o_cluster_centroids, \
                            o_silhouette_scores, \
                            o_inertia_scores, \
                            o_cluster_components=self._enumerate_cluster_benchmark(o_cluster_centroids, 
                                                                                o_silhouette_scores, 
                                                                            o_inertia_scores,o_clusterlist,alpha)
                            
                            if clustered == "kmeans":
                                
                                aux_loss = [a*(alpha-b) for a,b in zip(o_inertia_scores,o_silhouette_scores)]
                                self.signatures = o_cluster_centroids[np.argmin(aux_loss)]
                            
                            
                            if clustered == "cosine":
                                aux_loss = [a+alpha*b for a,b in zip(o_inertia_scores,o_silhouette_scores)]
                                self.signatures = o_cluster_centroids[np.argmax(aux_loss)]
                            self._find_exposures_benchmark(clustered,type_cluster,metrici,alpha)
                            
                    
       
       
    def _cluster_signatures(self):
        component = self._latent_checker()
        
        component = range(2,component)#TODO check that component is higher than 4
        cluster_centroids = []
        silhouette_scores = []
        inertia_scores = []
        
        pool = mp.Pool(self.multi_processnr)
        #TODO _kmeans is taking 99.6% of the time for the function
        
        pool_clustering = partial(clustering,prelim_signatures=self.prelim_signatures,
                                components=component,metric=self.silhouette_metric,
                                cluster=self.cluster,random_state=42)

        results_list = pool.map(pool_clustering, range(self.runs))
        pool.close()
        pool.join()
        
        for c_cluster_centroids, c_silhouette_scores, c_inertia_scores,cluster in results_list:  
            
            cluster_centroids.append(c_cluster_centroids)
            silhouette_scores.append(c_silhouette_scores)
            inertia_scores.append(c_inertia_scores)  
        if self.cluster == "kmeans":
            cluster_centroids, silhouette_scores, inertia_scores, cluster_components=self._enumerate_cluster(cluster_centroids, silhouette_scores, inertia_scores)
            
            if self.type_clustering == "single":
                silhouette_hat = (
                    np.array(silhouette_scores) - np.mean(silhouette_scores)
                ) / np.std(silhouette_scores)
                loss_hat = (self.avg_loss - np.mean(self.avg_loss)) / np.std(self.avg_loss)
                aux_loss = loss_hat - self.alpha * silhouette_hat
            
            if self.type_clustering == "multi":
                nparr = np.concatenate(cluster_centroids,axis=0)
                nparr = nparr/nparr.sum(axis=1,keepdims=True)
                arr = np.unique(nparr,axis=0)
                o_cluster_centroids = []
                o_silhouette_scores = []
                o_inertia_scores = []
                o_cluster_components = []
                
                o_range = range(2,arr.shape[0]-1)
                larr = [arr]*1 
                pool = mp.Pool(1)
                pool_clustering = partial(clustering,prelim_signatures=larr,
                                            components=o_range,metric=self.silhouette_metric,
                                            cluster=self.cluster,random_state=42)

                results_list = pool.map(pool_clustering, range(5))
                pool.close()
                pool.join()
                
                for i_cluster_centroids, i_silhouette_scores, i_inertia_scores,cluster in results_list:        
                    o_cluster_centroids.append(i_cluster_centroids)
                    o_silhouette_scores.append(i_silhouette_scores)
                    o_inertia_scores.append(i_inertia_scores)
                    
                o_cluster_centroids, o_silhouette_scores, o_inertia_scores, o_cluster_components=self._enumerate_cluster(o_cluster_centroids, o_silhouette_scores, o_inertia_scores)    
                aux_loss = [a*(self.alpha-b) for a,b in zip(o_inertia_scores,o_silhouette_scores)]
            
            self.signatures = cluster_centroids[np.argmin(aux_loss)]
            
        if self.cluster == "cosine":
            cluster_centroids, silhouette_scores, inertia_scores, cluster_components=self._enumerate_cluster(cluster_centroids, silhouette_scores, inertia_scores)
                
            if self.type_clustering == "old":
                aux_loss = [a+self.alpha*b for a,b in zip(inertia_scores,silhouette_scores)]
                
            if self.type_clustering == "new":
                nparr = np.concatenate(cluster_centroids,axis=0)
                nparr = nparr/nparr.sum(axis=1,keepdims=True)
                arr = np.unique(nparr,axis=0)
                o_cluster_centroids = []
                o_silhouette_scores = []
                o_inertia_scores = []
                o_cluster_components = []
                o_range = range(2,arr.shape[0]-1)
                larr = [arr]*1 
                pool = mp.Pool(1)
                pool_clustering = partial(clustering,prelim_signatures=larr,
                                            components=o_range,metric=self.silhouette_metric,
                                            cluster=self.cluster,random_state=42)

                results_list = pool.map(pool_clustering, range(5))
                pool.close()
                pool.join()
                
                for i_cluster_centroids, i_silhouette_scores, i_inertia_scores,cluster in results_list:        
                    o_cluster_centroids.append(i_cluster_centroids)
                    o_silhouette_scores.append(i_silhouette_scores)
                    o_inertia_scores.append(i_inertia_scores)
                        
                o_cluster_centroids, o_silhouette_scores, o_inertia_scores, o_cluster_components=self._enumerate_cluster(o_cluster_centroids, o_silhouette_scores, o_inertia_scores)                    
                aux_loss = [a+self.alpha*b for a,b in zip(o_inertia_scores,o_silhouette_scores)]        
                                
            self.signatures = cluster_centroids[np.argmax(aux_loss)]
            
       
        
    
        

def cosine_clustering(n_points,n_cluster:int = 3,max_iter:int = 300,threshold:float =0.00001,random_state:None|int=None,metric="euclidean",runs:int=0):
    """
    This function is used to cluster the points using cosine similarity
    therefore it is important to normalize the data so each node sums to one before using this function 
    """
    
    np.random.seed(random_state)
    random_centroid = np.random.choice(n_points.shape[0], n_cluster, replace=False)
    
    c_points = n_points[random_centroid]
    next_centroid = c_points.copy()
    breakhistory = 3
    breakcheck = 0
    last_score = np.array([0]*c_points.shape[0],dtype=float)
    emptycheck = 0
    for cur_iter in range(max_iter):
        inneremptycheck = 0
        next_total_points = np.array([0]*random_centroid.shape[0],dtype=float)
        cos_score = np.array([0]*c_points.shape[0],dtype=float)
        cluster = np.array([0]*n_points.shape[0],dtype=int)
        cos_sim = 1-pairwise_distances(n_points,c_points,metric="cosine")
        cluster = np.argmax(cos_sim,axis=1)
        for i in range(n_points.shape[0]):
            next_centroid[cluster[i]] = next_centroid[cluster[i]]+n_points[i]
            next_total_points[cluster[i]] += 1
            cos_score[cluster[i]] += cos_sim[i][cluster[i]]
            
        for i in np.setdiff1d(np.arange(0,random_centroid.shape[0],1),cluster):
            inneremptycheck = 1
            tempflag = True
            tt = 0
            while tempflag == True:
                tempbest =np.argsort(cos_sim[:,1])[::-1]
                removebest = cluster[tempbest[tt]]
                if next_total_points[removebest] >1:
                    tempbest = tempbest[tt]
                    tempflag = False
                else:
                    tt=tt+1
                    
            cluster[tempbest]=i
            next_total_points[removebest] -= 1
            next_total_points[i] += 1
            cos_score[i] += cos_sim[tempbest][i]
            cos_score[removebest] -= cos_sim[tempbest][removebest]
            next_centroid[i] = np.add(next_centroid[i],n_points[tempbest])
            next_centroid[removebest] = np.subtract(next_centroid[removebest],n_points[tempbest])
        
        if inneremptycheck > 0:
            emptycheck += 1
                                
        next_centroid = next_centroid/next_centroid.sum(axis=1).reshape(-1,1)
        cos_score = cos_score/next_total_points
        
        c_points = next_centroid.copy()
        
        if cur_iter > 3 and (cos_score-last_score).mean() < threshold:
            breakcheck += 1
            if breakhistory == breakcheck:
                break
        else:
            breakcheck = 0
            
        if emptycheck > 1:
            break
            
        last_score = cos_score.copy()
    sil_score = silhouette_score(n_points,cluster,metric=metric)
    return c_points,cos_score.mean(),sil_score



