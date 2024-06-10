import pandas as pd
from plotnine import ggplot, aes, geom_boxplot, labs, theme, scale_color_manual, element_text,scale_y_continuous
import os
import patchworklib as pw
import warnings
warnings.filterwarnings('ignore')
import numpy as np
def createggplots(df:pd.DataFrame,filter:str,refletter:str):

# Melt the dataframe for easier plotting with plotnine
    df_melted = df.melt(id_vars=[filter],
                        value_vars=['>0.85', '>0.87', '>0.89', '>0.90', '>0.91', 
                                    '>0.92', '>0.93', '>0.94', '>0.95', '>0.96', 
                                    '>0.97', '>0.98', '>0.99',"best>0.95","best>0.99","found"], 
                        var_name='Threshold', value_name='Value')

    # Split the melted DataFrame into two parts
    thresholds_part1 = ['>0.85', '>0.87', '>0.89', '>0.90', '>0.91', '>0.92', '>0.93','>0.94',]
    thresholds_part2 = [ '>0.95', "best>0.95",'>0.96', '>0.97', '>0.98',]
    thresholds_part3 = [ '>0.99',"best>0.99"]
    thresholds_part4 = ["found"]

    df_part1 = df_melted[df_melted['Threshold'].isin(thresholds_part1)]
    df_part2 = df_melted[df_melted['Threshold'].isin(thresholds_part2)]
    df_part3 = df_melted[df_melted['Threshold'].isin(thresholds_part3)]
    df_part4 = df_melted[df_melted['Threshold'].isin(thresholds_part4)]


    atp =df_part1[filter].unique().tolist()
    # print(sorted(atp))
    atf =pd.Categorical(df_part1[filter],ordered=True,categories=atp)
    # Set the color palette for distinct colors
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Plot for the first part
    # plot1 = (ggplot(df_part1, aes(x='Threshold', y='Value', fill=f"factor({filter})")) +
    #     geom_boxplot(alpha=0.5, outlier_shape='o',width=0.85) +
    #     scale_color_manual(values=palette) +
    #     labs(x='Threshold Columns', y='Value', title='Distribution of Values by Threshold (Part 1)', fill=filter) +
    #     # scale_x_discrete(labels=thresholds_part1,name="threshold columns") +
    #     theme(axis_text_x=element_text(angle=45, hjust=1)) +
    #     theme(legend_position='right'))

    atp =df_part2[filter].unique().tolist()
    atf =pd.Categorical(df_part2[filter],ordered=True,categories=atp)
    # Plot for the second part
    plot2 = (ggplot(df_part2, aes(x='Threshold', y='Value', fill=f"factor({filter})")) +
        geom_boxplot(alpha=0.5, outlier_shape='o',width=0.83) +
        scale_color_manual(values=palette) +
        labs(x='Cosine similarity Threshold',title=f"{refletter}1" ,y='Mean count of signatures', fill=" \n& ".join(filter.split("__"))) +
        theme(axis_text_x=element_text(hjust=1,size=25)) +
        theme(legend_position='right')+
        theme(axis_text_y=element_text(size=25))+
        theme(legend_text=element_text(size=25))+
        theme(legend_title=element_text(size=25))+
        theme(axis_title=element_text(size=28))+
        theme(plot_title=element_text(size=25,weight="bold"))
        
        )

    atp =df_part3[filter].unique().tolist()
    atf =pd.Categorical(df_part3[filter],ordered=True,categories=atp)
    plot3 = (ggplot(df_part3, aes(x='Threshold', y='Value', fill=f"factor({filter})")) +
        geom_boxplot(alpha=0.5, outlier_shape='o',width=0.9) +
        scale_color_manual(values=palette) +
        labs(x='Cosine similarity Threshold',title=f"{refletter}2", y='Mean count of signatures', fill=" \n& ".join(filter.split("__"))) +
        
        theme(axis_text_x=element_text(hjust=1,size=25)) +
        theme(legend_position='right')+
        theme(axis_text_y=element_text(size=25))+
        theme(legend_text=element_text(size=25))+
        theme(legend_title=element_text(size=25))+
        theme(axis_title=element_text(size=28))+
        theme(plot_title=element_text(size=25,weight="bold"))
        
        )

 

    # Save the plots to PNG files
    # plot1.save(f"boxplot_{filter}_part1.png", width=12, height=8)
    # plot2.save(f"zimg/signal/boxplot_kmeans_new_0.6_{filter}_part2.png", width=12, height=7)
    # plot3.save(f"zimg/signal/boxplot_kmeans_new_0.6_{filter}_part3.png", width=10, height=7)
    return plot2,plot3

def plottingdata(pathstr,title:str,xtitle:str):
    df = pd.read_csv(pathstr+"/results.tsv",sep="\t")
    valuelist = ["found",">0.85",">0.87",
                 ">0.89",">0.90",">0.91",
                 ">0.92",">0.93",">0.94",
                 ">0.95",">0.96",">0.97",
                 ">0.98",">0.99","best>0.95",
                 "best>0.99"]
    noneinjectiondf = df.loc[df["injectionprocent"]==0]
    if not os.path.exists(pathstr+"/images"):
        os.makedirs(pathstr+"/images")
    predf = noneinjectiondf.groupby(["method","runnr","latents","cut_off","bootstrap","noise"])[valuelist].mean()
    predf = predf.reset_index()
    refletter = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    for filtering in ["latents","cut_off","bootstrap","noise"]:
        
        plot2,plot3 = createggplots(predf,filtering,refletter.pop(0))
        # plot2.save(f"{pathstr}/images/boxplot_pre_{filtering}_part1.png", width=12, height=8)
        # plot3.save(f"{pathstr}/images/boxplot_pre_{filtering}_part2.png", width=12, height=8)
        g1 = pw.load_ggplot(plot2,figsize=(11,4))
        g2 = pw.load_ggplot(plot3,figsize=(7.5,4))
        g12 = (g1|g2)
        g12.case.set_title(title,fontsize=30,pad=35)
        g12.case.set_xlabel(xtitle,fontsize=30)
        g12.savefig(f"{pathstr}/images/boxplot_pre_{filtering}.png")
        
    postdf = noneinjectiondf.groupby(["alpha","type_clustering","cluster_method","silhouette_metric"])[valuelist].mean()
    postdf = postdf.reset_index()
    refletter = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    for filtering in ["alpha","type_clustering","silhouette_metric"]:
        postdf[f"cluster_method_{filtering}"] =  df["cluster_method"].astype(str)  + " "+postdf[filtering].astype(str)  
        filtering= f"cluster_method_{filtering}"
        plot2,plot3 = createggplots(postdf,filtering,refletter.pop(0))
        # plot2.save(f"{pathstr}/images/boxplot_post_{filtering}_part1.png", width=12, height=8)
        # plot3.save(f"{pathstr}/images/boxplot_post_{filtering}_part2.png", width=12, height=8)
        g1 = pw.load_ggplot(plot2,figsize=(11,4))
        g2 = pw.load_ggplot(plot3,figsize=(7.5,4))
        g12 = (g1|g2)
        g12.case.set_title(title,fontsize=30,pad=35)
        g12.case.set_xlabel(xtitle,fontsize=30)
        g12.savefig(f"{pathstr}/images/boxplot_post_{filtering}.png")
    
    
# Assuming your data is stored in a DataFrame named 'df'
def injectionplotting(pathstr,title:str):
    
    prefilter = ["method","runnr","latents","cut_off","bootstrap","noise","injection_percentage"]
    df=pd.read_table(pathstr+"/results.tsv",sep="\t")
    if not os.path.exists(pathstr+"/images"):
        os.makedirs(pathstr+"/images")
    tuplesofinjection=(df["injections_match"])
    listofvalues = []
    for t in tuplesofinjection:
        listofvalues.append([float(t.split(",")[0][2:]),t.split(",")[1][2:-1],t.split(",")[2][2:-3]])
        
    
    dfa = pd.DataFrame(listofvalues,columns=["cos_sim","latent","name"])
    
    dfa[prefilter]=df[prefilter]
    dfa = dfa.groupby(prefilter)["cos_sim"].agg(["mean","std","min","max"])
    dfa = dfa.reset_index()
    
    
# Melt the dataframe for easier plotting with plotnine
    df_melted = dfa.melt(id_vars=["injection_percentage"],
                        value_vars=['mean','std','min','max'], 
                        var_name='injection', value_name='Value')

# Split the melted DataFrame into two parts
    thresholds_part1 = ['mean','min','max']
    thresholds_part2 = ['std']
    df_part1 = df_melted[df_melted['injection'].isin(thresholds_part1)]

    df_part2 = df_melted[df_melted['injection'].isin(thresholds_part2)]
# df_part4 = df_melted[df_melted['Threshold'].isin(thresholds_part4)]


# Set the color palette for distinct colors
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Plot for the first part
    plot1 = (ggplot(df_part1, aes(x='injection', y='Value', fill=f'factor(injection_percentage)')) +
        geom_boxplot(alpha=0.5, outlier_shape='o',width=0.9) +
        scale_color_manual(values=palette) +
        labs(x="",y='Cosine similarity', title=title, fill="Injection\n\nPercentage\n\n\n") +
        # scale_x_discrete(labels=thresholds_part1,name="threshold columns") +
        theme(axis_text_x=element_text(hjust=1,size=25)) +
        theme(legend_position='right')+
        theme(axis_text_y=element_text(size=25))+
        theme(legend_text=element_text(size=25))+
        theme(legend_title=element_text(size=25))+
        theme(axis_title=element_text(size=28))+
        theme(plot_title=element_text(size=25,weight="bold")) + 
        scale_y_continuous(breaks=np.arange(0.0,1.1,0.1),limits=[0.3,1])
        
        )
        

    plot2 = (ggplot(df_part2, aes(x='injection', y='Value', fill=f'factor(injection_percentage)')) +
        geom_boxplot(alpha=0.5, outlier_shape='o',) +
        scale_color_manual(values=palette) +
        labs(x='Threshold Columns', y='Value', title=title, fill="injection_percentage") +
        # scale_x_discrete(labels=thresholds_part1,name="threshold columns") +
        theme(axis_text_x=element_text(angle=45, hjust=1)) +
        theme(legend_position='right'))

# Save the plots to PNG files
    plot1.save(pathstr+"/images/injectionplot_part1.png", width=16, height=8)
    plot2.save(pathstr+"/images/injectionplot_part2.png", width=16, height=8)
# plot4.save("boxplot_part4.png", width=16, height=8)