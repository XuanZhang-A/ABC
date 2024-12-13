import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class EmbeddingVisualizer:
    def __init__(self, n_components=2, config=None) -> None:
        self.config = config
        self.tsne = TSNE(n_components=n_components, verbose=1, random_state=123)
        self.color_palette = sns.color_palette("hls", self.config["arch"]["args"]["num_classes"]+1)
        self.color_palette[-1] = (0.0, 0.0, 0.0) # assign black to ood

    def fit(self, x, y):
        """Train TSNE

        Args:
            x (List): feature vectors
            y (List): labels
        """
        self.x = np.concatenate(x, axis=0)
        self.y = np.array(y).flatten()
        self.projected_x = self.tsne.fit_transform(self.x)
    
    def visualize(self, epoch=999, fname=None, save_dir=None, image_count=[]):
        if save_dir is None:
            save_dir = "out"
        save_dir = save_dir + "/tsne"
        os.makedirs(save_dir, exist_ok=True)

        df = pd.DataFrame()
        df["y"] = self.y
        df['legend'] = df['y'].apply(lambda x: f"class_{int(x)} ({image_count[int(np.clip(x, 0, len(image_count)-1))]})")
        df["comp-1"] = self.projected_x[:,0]
        df["comp-2"] = self.projected_x[:,1]
        df = df.sort_values('y')

        if fname is None:
            fname = str(epoch).zfill(5)+".png"
        else:
            fname = fname+".png"

        tsne_plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.legend.to_list(), palette=self.color_palette, data=df, s=10)
        tsne_plot.set(title=fname) 

        legend = tsne_plot.legend()
        legend.set_bbox_to_anchor((1,1))
        # legend.set_loc('upper left')
        # for handle in legend.legendHandles:
        #     handle.set_markersize(10)
        fig = tsne_plot.get_figure()
        fig.set_figheight(20)
        fig.set_figwidth(20)
        fig.savefig(save_dir+"/"+fname) 
        plt.clf()