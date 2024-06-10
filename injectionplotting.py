import pandas as pd
from plotnine import ggplot, aes, geom_boxplot, labs, theme, scale_color_manual, element_text,scale_y_continuous
import numpy as np
import os

# Assuming your data is stored in a DataFrame named 'df'
def injectionplotting(pathstr):
    
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
    
    
# Melt the dataframe for easier plotting with plotnine
    df_melted = dfa.melt(id_vars=[filter],
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
    plot1 = (ggplot(df_part1, aes(x='injection', y='Value', fill=f'factor({filter})')) +
        geom_boxplot(alpha=0.5, outlier_shape='o',width=0.9) +
        scale_color_manual(values=palette) +
        labs(x="",y='Cosine similarity', title='Distribution of cosine similarity by \ninjection percentage trained on PCAWG', fill="Injection\n\nPercentage\n\n\n") +
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
        

    plot2 = (ggplot(df_part2, aes(x='injection', y='Value', fill=f'factor({filter})')) +
        geom_boxplot(alpha=0.5, outlier_shape='o',) +
        scale_color_manual(values=palette) +
        labs(x='Threshold Columns', y='Value', title='Distribution of Values by Threshold (Part 1)', fill=filter) +
        # scale_x_discrete(labels=thresholds_part1,name="threshold columns") +
        theme(axis_text_x=element_text(angle=45, hjust=1)) +
        theme(legend_position='right'))

# Save the plots to PNG files
    plot1.save(pathstr+"/images/injectionplot_part1.png", width=16, height=8)
    plot2.save(pathstr+"/images/injectionplot_part2.png", width=16, height=8)
# plot4.save("boxplot_part4.png", width=16, height=8)