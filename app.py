from flask import Flask, render_template, request
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import itertools
from scipy import linalg
from sklearn import mixture
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans

app = Flask(__name__)
@app.route('/')
def student():
   return render_template('UserInterface.html')



@app.route('/result',methods = ['GET', 'POST'])
def result():
    if request.method == 'POST':
        result = request.form
        items = result.items
        datasetLink = request.form['Dataset Link']
        algo = request.form['Algorithm Name']
        clusterFactor1 = request.form['Cluster Factor 1']
        clusterFactor2 = request.form['Cluster Factor 2']
        clusterFactor3 = request.form['Cluster Factor 3']
        hoverFactor1 = request.form['Hover Factor 1']
        hoverFactor2 = request.form['Hover Factor 2']
        hyperParam = []
        if (algo=='DBSCAN'):
            hyperParam1 = request.form['Hyper Parameter 1']
            hyperParam2 = request.form['Hyper Parameter 2']
            hyperParam1 = int(hyperParam1)
            hyperParam1 = hyperParam1/100
            hyperParam2 = int(hyperParam2)
            hyperParam = [hyperParam1, hyperParam2]
        if (algo=='OPTICS'):
            hyperParam1 = request.form['Hyper Parameter 1']
            hyperParam2 = request.form['Hyper Parameter 2']
            hyperParam3 = request.form['Hyper Parameter 3']
            hyperParam1 = int(hyperParam1)
            hyperParam1 = hyperParam1/100
            hyperParam2 = int(hyperParam2)
            hyperParam2 = hyperParam2/100
            hyperParam3 = int(hyperParam3)
            hyperParam = [hyperParam1, hyperParam2, hyperParam3]
        if (algo=='GMM'):
            hyperParam1 = request.form['Hyper Parameter 1']
            hyperParam1 = int(hyperParam1)
            hyperParam = [hyperParam1]
        if (algo=='HRCL'):
            hyperParam1 = request.form['Hyper Parameter 1']
            hyperParam1 = int(hyperParam1)
            hyperParam = [hyperParam1]
        if (algo=='CTRD'):
            hyperParam1 = request.form['Hyper Parameter 1']
            hyperParam1 = int(hyperParam1)
            hyperParam = [hyperParam1]

        clusterFactors = [clusterFactor1, clusterFactor2, clusterFactor3]
        hoverFactors = [hoverFactor1, hoverFactor2]
        
        data = pd.read_csv(datasetLink)
        
        CustomerData1 = data.sort_values(by=clusterFactors[0], ascending=True)
        df = CustomerData1[[clusterFactors[0]]]
        num_clusters = 2

        def calculate_zscore(df, columns):
            df = df.copy()
            for col in columns:
                df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

            return df

        def one_hot_encode(df, columns):
            concat_df = pd.concat([pd.get_dummies(df[col], drop_first=False, prefix=col) for col in columns], axis=1)
            one_hot_cols = concat_df.columns

            return concat_df, one_hot_cols

        numeric_cols = df.select_dtypes(include=np.number)
        cat_cols = df.select_dtypes(include='object')
        
        normalized_df = calculate_zscore(df, numeric_cols)
        normalized_df = normalized_df[numeric_cols.columns]

        cat_one_hot_df, one_hot_cols = one_hot_encode(df, cat_cols)
        cat_one_hot_norm_df = calculate_zscore(cat_one_hot_df, one_hot_cols)

        processed_df = pd.concat([normalized_df, cat_one_hot_norm_df], axis=1)
        CustomerData2 = pd.concat([CustomerData1, processed_df], axis=1)


        factor1list = CustomerData2[clusterFactors[0]].unique()
        factor2list = CustomerData2[clusterFactors[1]].unique()
        CustomerData2 = CustomerData2.reset_index()

        CustomerData2[clusterFactors[0]+' Value'] = " "
        compiledvalname = (clusterFactors[0]+" Value")

        i=0
        for i in range(((len(factor1list)))):
            compiledname = (clusterFactors[0]+"_"+factor1list[i])
            for j in range(len(CustomerData2)):
                temp = CustomerData2[compiledname].values[j]
                if (temp>0):
                    CustomerData2.loc[j,compiledvalname] = temp   

        RegionData1 = CustomerData2.sort_values(by=clusterFactors[1], ascending=True)
        df = RegionData1[[clusterFactors[1]]]

        num_clusters = 2

        def calculate_zscore(df, columns):    
            df = df.copy()
            for col in columns:
                df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

            return df    

        def one_hot_encode(df, columns):
            concat_df = pd.concat([pd.get_dummies(df[col], drop_first=False, prefix=col) for col in columns], axis=1)
            one_hot_cols = concat_df.columns

            return concat_df, one_hot_cols 
        
        numeric_cols = df.select_dtypes(include=np.number)
        cat_cols = df.select_dtypes(include='object')
        
        normalized_df = calculate_zscore(df, numeric_cols)
        normalized_df = normalized_df[numeric_cols.columns]

        cat_one_hot_df, one_hot_cols = one_hot_encode(df, cat_cols)
        cat_one_hot_norm_df = calculate_zscore(cat_one_hot_df, one_hot_cols)

        processed_df = pd.concat([normalized_df, cat_one_hot_norm_df], axis=1)
        RegionData2 = pd.concat([RegionData1, processed_df], axis=1)
        RegionData2 = RegionData2.reset_index()
        factor2list.sort()

        RegionData2[clusterFactors[1]+' Value'] = " "

        compiledvalname = (clusterFactors[1]+" Value")
        i=0
            
        for i in range(((len(factor2list)))):
            compiledname = (clusterFactors[1]+"_"+factor2list[i])            
                    
            for j in range(len(RegionData2)):
                temp = RegionData2[compiledname].values[j]
                if (temp>0):
                    RegionData2.loc[j,compiledvalname] = temp

        PaymentData2 = RegionData2
        name1 = clusterFactors[0]+' Value'
        name2 = clusterFactors[1]+' Value'
        name3 = clusterFactors[2]+' Value'

        df = PaymentData2

        if (algo=='DBSCAN'):
            X=df[[name1,name2,clusterFactors[2]]]
            X = StandardScaler().fit_transform(X)

            db = DBSCAN(eps=hyperParam[0], min_samples=hyperParam[1]).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            from sklearn.manifold import TSNE
            X_embedded = TSNE(n_components=2).fit_transform(X)
            df["x_component"]=X_embedded[:,0]
            df["y_component"]=X_embedded[:,1]
            df["labels"]=labels

        if (algo=='OPTICS'):

            np.random.seed(0)
            n_points_per_cluster = 250

            X=df[[name1,name2,clusterFactors[2]]]
            X = StandardScaler().fit_transform(X)
            clust = OPTICS(min_samples=hyperParam[2], xi=hyperParam[1], min_cluster_size=hyperParam[0])
            clust.fit(X)

            labels_050 = cluster_optics_dbscan(
                reachability=clust.reachability_,
                core_distances=clust.core_distances_,
                ordering=clust.ordering_,
                eps=0.5,
            )
            labels_200 = cluster_optics_dbscan(
                reachability=clust.reachability_,
                core_distances=clust.core_distances_,
                ordering=clust.ordering_,
                eps=2,
            )
            space = np.arange(len(X))
            reachability = clust.reachability_[clust.ordering_]
            labels = clust.labels_[clust.ordering_]

            plt.figure(figsize=(10, 7))
            G = gridspec.GridSpec(2, 3)
            ax1 = plt.subplot(G[0, :])
            ax2 = plt.subplot(G[1, 0])
            ax3 = plt.subplot(G[1, 1])
            ax4 = plt.subplot(G[1, 2])

            colors = ["g.", "r.", "b.", "y.", "c."]
            for klass, color in zip(range(0, 5), colors):
                Xk = space[labels == klass]
                Rk = reachability[labels == klass]
                ax1.plot(Xk, Rk, color, alpha=0.3)
            ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
            ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
            ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
            ax1.set_ylabel("Reachability (epsilon distance)")
            ax1.set_title("Reachability Plot")

            colors = ["g.", "r.", "b.", "y.", "c."]
            for klass, color in zip(range(0, 5), colors):
                Xk = X[clust.labels_ == klass]
                ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
            ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], "k+", alpha=0.5)
            ax2.set_title("Automatic Clustering\nOPTICS")

            from sklearn.manifold import TSNE
            X_embedded = TSNE(n_components=2).fit_transform(X)
            df["x_component"]=X_embedded[:,0]
            df["y_component"]=X_embedded[:,1]
            df["labels"]=labels
            
            colors = ["g", "greenyellow", "olive", "r", "b", "c"]
            for klass, color in zip(range(0, 6), colors):
                Xk = X[labels_050 == klass]
                ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker=".")
            ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], "k+", alpha=0.1)
            ax3.set_title("Clustering at 0.5 epsilon cut\nDBSCAN")

            colors = ["g.", "m.", "y.", "c."]
            for klass, color in zip(range(0, 4), colors):
                Xk = X[labels_200 == klass]
                ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
            ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], "k+", alpha=0.1)
            ax4.set_title("Clustering at 2.0 epsilon cut\nDBSCAN")

            plt.tight_layout()
        if (algo=='GMM'):
            n_samples = 1000
            np.random.seed(0)
            X=df[[name1,name2,clusterFactors[2]]]
            X = X.values
            lowest_bic = np.infty
            bic = []
            temporary = hyperParam[0] + 1
            n_components_range = range(1, temporary)
            cv_types = ["spherical", "tied", "diag", "full"]
            for cv_type in cv_types:
                for n_components in n_components_range:
                    gmm = mixture.GaussianMixture(
                        n_components=n_components, covariance_type=cv_type
                    )
                    gmm.fit(X)
                    bic.append(gmm.bic(X))
                    if bic[-1] < lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = gmm

            bic = np.array(bic)
            color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
            clf = best_gmm
            bars = []

            plt.figure(figsize=(8, 6))
            spl = plt.subplot(2, 1, 1)
            for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
                xpos = np.array(n_components_range) + 0.2 * (i - 2)
                bars.append(
                    plt.bar(
                        xpos,
                        bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                        width=0.2,
                        color=color,
                    )
                )
            plt.xticks(n_components_range)
            plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
            plt.title("BIC score per model")
            xpos = (
                np.mod(bic.argmin(), len(n_components_range))
                + 0.65
                + 0.2 * np.floor(bic.argmin() / len(n_components_range))
            )
            plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
            spl.set_xlabel("Number of components")
            spl.legend([b[0] for b in bars], cv_types)

            splot = plt.subplot(2, 1, 2)
            Y_ = clf.predict(X)
            for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
                v, w = linalg.eigh(cov)
                if not np.any(Y_ == i):
                    continue
                plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

            from sklearn.manifold import TSNE
            X_embedded = TSNE(n_components=2).fit_transform(X)
            df["x_component"]=X_embedded[:,0]
            df["y_component"]=X_embedded[:,1]
            labels = clf.fit_predict(X)
            df["labels"]=labels

            plt.xticks(())
            plt.yticks(())
            plt.title(
                f"Selected GMM: {best_gmm.covariance_type} model, "
                f"{best_gmm.n_components} components"
            )
            plt.subplots_adjust(hspace=0.35, bottom=0.02)
        if (algo=='HRCL'):
            df=data
            k = hyperParam[0]
            m = df.groupby([clusterFactors[0], clusterFactors[1]])[clusterFactors[2]].sum().reset_index()
            c1 = m.loc[:, ['TRx']]

            dendrogram = sch.dendrogram(sch.linkage(c1, method = "ward"))
            #plt.show()

            dforig = df
            compiledvalname0 = (clusterFactors[0]+" Value")
            compiledvalname1 = (clusterFactors[1]+" Value")
            m=RegionData2[[compiledvalname0,compiledvalname1,'TRx']]
            #m = RegionData2.groupby([compiledvalname0, compiledvalname1])[clusterFactors[2]].sum().reset_index()
            m = m.reset_index()

            cluster = AgglomerativeClustering(n_clusters=k, affinity = 'euclidean', linkage = 'ward')
            prediction = cluster.fit_predict(m)
            df=RegionData2
            #df = RegionData2.groupby([compiledvalname0, compiledvalname1])[clusterFactors[2]].sum().reset_index()
            df['clusters'] = prediction    
            df['x_component'] = RegionData2[compiledvalname0]
            df['y_component'] = RegionData2[compiledvalname1]
        if (algo=='CTRD'):
            c = clusterFactors[0]
            mdf = df.loc[:, [clusterFactors[0], clusterFactors[1], clusterFactors[2]]]
            mdf2 = mdf.values
            compiledvalname0 = (clusterFactors[0]+" Value")
            compiledvalname1 = (clusterFactors[1]+" Value")
            k = 3
            top_value = hyperParam[0]+1
            k_rng = range(1,top_value)
            sse = []
            for k in k_rng:
                kmns = KMeans(n_clusters = k)
                kmns.fit(mdf[[clusterFactors[2]]])
                sse.append(kmns.inertia_)
            plt.plot(k_rng,sse)

            kproto = KPrototypes(n_clusters=k, verbose = 2)
            clusters = kproto.fit_predict(mdf2, categorical=[0, 1])
            clusterList=[]
            for c in clusters:
                clusterList.append(c)
            mdf['clusters'] = clusterList
            df = RegionData2
            df['clusters'] = clusterList

        if (algo == 'DBSCAN'):
            fig2 = px.scatter_3d(df, x="x_component", y="y_component",z=clusterFactors[2],
                                color="labels",hover_data=[clusterFactors[0],clusterFactors[1],clusterFactors[2],hoverFactors[0],hoverFactors[1]],
                                labels={
                                "x_component": clusterFactors[0],
                                "y_component": clusterFactors[1],
                                "labels": "Cluster"
                            },)
            fig2.update_layout(title=name1 + " vs "+ name2 + " vs " + name3 + " Clustering Visualization")
        elif (algo=='OPTICS'):
            fig2 = px.scatter_3d(df, x="x_component", y="y_component",z=clusterFactors[2],
                                color="labels",hover_data=[clusterFactors[0],clusterFactors[1],clusterFactors[2],hoverFactors[0],hoverFactors[1]],
                                labels={
                                "x_component": clusterFactors[0],
                                "y_component": clusterFactors[1],
                                "labels": "Cluster"
                            },)
            fig2.update_layout(title=name1 + " vs "+ name2 + " vs " + name3 + " Clustering Visualization")
        elif (algo=='GMM'):
            fig2 = px.scatter_3d(df, x="x_component", y="y_component",z=clusterFactors[2],
                                color="labels",hover_data=[clusterFactors[0],clusterFactors[1],clusterFactors[2],hoverFactors[0],hoverFactors[1]],
                                labels={
                                "x_component": clusterFactors[0],
                                "y_component": clusterFactors[1],
                                "labels": "Cluster"
                            },)
            fig2.update_layout(title=name1 + " vs "+ name2 + " vs " + name3 + " Clustering Visualization")
        elif (algo=='HRCL'):
            #plt.show()
            fig2 = px.scatter_3d(df, x=compiledvalname0, y=compiledvalname1,z=clusterFactors[2],
                                color="clusters",hover_data=[clusterFactors[0],clusterFactors[1],clusterFactors[2],hoverFactors[0],hoverFactors[1]],
                                labels={
                                "x_component": clusterFactors[0],
                                "y_component": clusterFactors[1],
                                "labels": "Cluster"
                            },)
            fig2.update_layout(title=name1 + " vs "+ name2 + " vs " + name3 + " Clustering Visualization")

        elif (algo=='CTRD'):
            #plt.show()
            fig2 = px.scatter_3d(df, x=compiledvalname0, y=compiledvalname1,z=clusterFactors[2],
							color="clusters",hover_data=[clusterFactors[0],clusterFactors[1],clusterFactors[2],hoverFactors[0],hoverFactors[1]],
							labels={
							"x_component": clusterFactors[0],
                            "y_component": clusterFactors[1],
							"labels": "Cluster"
						},)
            fig2.update_layout(title=compiledvalname0 + " vs "+ compiledvalname1 + " vs " + clusterFactors[2] + " Clustering Visualization")

        #fig2.write_html(r"C:/Users/Owner/Desktop/Final Visualization/FinalPlot.html",auto_open=True)
        fig2.write_html("C:\FinalVisualizationResults\FinalVisualization.html",auto_open=True)
        
        return render_template("result.html",result = result)

if __name__ == '__main__':
   app.run(debug = True)