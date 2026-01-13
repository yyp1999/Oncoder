import Oncoder
import torch
import pandas as pd
from Oncoder import Autoencoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

def plotHeatmap(filepath,modelpath):
    traindata = pd.read_csv(filepath,sep='\t',index_col=0)
    model = torch.load(modelpath,map_location=device,weights_only = False)
    val_x,val_y = Oncoder.generate_simulated_data(filepath,prior=[0.8,0.2],samplenum=100,random_state=1)

    pred_y,methyatlas = Oncoder.predict(val_x,model)
    learned_atlas=pd.DataFrame(methyatlas.clamp(min=0,max=1).cpu().detach().numpy(),index=['plasma','tumor'],columns=traindata.index)

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init=10)
    row_clusters = kmeans.fit_predict(traindata)
    column_order = np.argsort(row_clusters)
    learned_atlas = learned_atlas.iloc[[1,0], column_order]

    traindata_plot = traindata.T
    traindata_plot = traindata_plot.iloc[:, column_order]
    
    
    g_learned = sns.clustermap(learned_atlas,cmap='RdYlBu_r', annot=False,row_cluster=False,col_cluster=False,xticklabels=False,method='complete')
    plt.suptitle('Learned Methylation Atlas', fontsize=20, y=0.90)
    g_learned.ax_heatmap.tick_params(axis='y', labelsize=18)


    g_raw = sns.clustermap(traindata_plot,cmap='RdYlBu_r', annot=False,row_cluster=False,col_cluster=False,xticklabels=False,method='complete')
    plt.suptitle('Reference Methylation Atlas', fontsize=20, y=0.90)
    g_raw.ax_heatmap.tick_params(axis='y', labelsize=14)
    return g_learned, g_raw, learned_atlas


def main():
    parser = argparse.ArgumentParser(description="Accepts the raw input training set and the model to visualize the performance of Oncoder in learning methylation profiles.")

    parser.add_argument('--file', type=str, required=True, help='Path to training set ')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save photo') 
    args = parser.parse_args()

    input_filepath = args.file
    input_modelpath = args.model
    input_save_dir = args.save_dir

    os.makedirs(input_save_dir, exist_ok=True)

    print("ploting ")
    g_learned,g_raw, learned_atlas= plotHeatmap(input_filepath, input_modelpath)
    
    print("saving ")
    model_basename = os.path.basename(input_modelpath)
    model_name = os.path.splitext(model_basename)[0]
    savepath_learned = os.path.join(input_save_dir, f"{model_name}_heatmap_learned.jpg")
    savepath_learned_csv = os.path.join(input_save_dir, f"{model_name}_heatmap_learned.csv")
    savepath_raw = os.path.join(input_save_dir, f"{model_name}_heatmap_raw.jpg")

    print(f"Saving learned atlas to: {savepath_learned_csv}")
    learned_atlas.to_csv(savepath_learned_csv)
    print(f"Saving learned heatmap to: {savepath_learned}")
    g_learned.savefig(savepath_learned, dpi=300, bbox_inches='tight')
    print(f"Saving raw heatmap to: {savepath_raw}")
    g_raw.savefig(savepath_raw, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
