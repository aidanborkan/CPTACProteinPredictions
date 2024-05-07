# Edward Lau 2021
# This code uses the CPTAC package to download CPTAC data for machine learning.


import cptac
import pandas as pd
import re
import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

# Add the path to sys.path
predict_protein_path = 'C:\\Users\\borkana\\CPTAC_Protein\\predict_protein'
sys.path.append(predict_protein_path)

# Now you can import the modules
try:
    from download_cptac import *
except ImportError as e:
    print(f"An error occurred with download_cptac import: {e}")

try:
    from select_features import *

except ImportError as e:
    print(f"An error occurred with select_features import: {e}")

try:
    from train_model import *
except ImportError as e:
    print(f"An error occurred with train_model import: {e}")

class LearnCPTAC(object):

    def __init__(self, cptac_df):

        from get_proteins import GetProtein, GetComplex

        self.df = cptac_df

        self.all_proteomics = [re.sub('_proteomics', "", protein) for protein in self.df.columns if
                               protein.endswith('_proteomics')]

        self.all_transcriptomics = [re.sub('_transcriptomics', "", transcript) for transcript in self.df.columns if
                                    transcript.endswith('_transcriptomics')]

        self.shared_proteins = [protein for protein in self.all_proteomics if protein in self.all_transcriptomics]

        self.tx_to_include = "self"
        self.train_method = "simple"

        self.stringdb = GetProtein()
        self.corumdb = GetComplex()

        pass

     def learn_all_proteins(self, methods, tx_to_include="self"):
        self.tx_to_include = tx_to_include
        learning_results = []

        for method in methods:
            self.train_method = method  # Set the current training method
            protein_metrics = []
            
            #modified by AB: test each method to replicate work of Dr. Lau

            for i, protein in enumerate(tqdm(self.shared_proteins, desc=f'Learning proteins with {method}')):
                metrics_df = self.learn_one_protein(protein)
                if metrics_df is not None and not metrics_df.empty:
                    protein_metrics.append(metrics_df)

                # Print progress messages every 100 proteins
                if i % 100 == 0 and protein_metrics:
                    corr_values = [metric['corr_test'].values[0] for metric in protein_metrics]
                    r2_values = [metric['r2_test'].values[0] for metric in protein_metrics]
                    nrmse = [metric['nrmse'].values[0] for metric in protein_metrics]

                    tqdm.write('{0}: {1}, r: {2}, R2: {3}, med.r: {4}, med.R2: {5}, med.NRMSE: {6}'.format(
                        i,
                        protein,
                        round(corr_values[-1], 3),
                        round(r2_values[-1], 3),
                        round(np.median([r for r in corr_values if not np.isnan(r)]), 3),
                        round(np.median([r2 for r2 in r2_values if not np.isnan(r2)]), 3),
                        round(np.median([nr for nr in nrmse if not np.isnan(nr)]), 3),
                    ))

            learning_results.extend(protein_metrics)

        return pd.concat(learning_results, ignore_index=True)

    def learn_one_protein(self, protein_to_do, returnModel=False):
        y_df = self.df[[protein_to_do + '_proteomics']]
        y_df = y_df.dropna(subset=[protein_to_do + '_proteomics'])

        if self.tx_to_include == "self":
            proteins_to_include = [protein_to_do]

        else:
            raise Exception('tx to include is not self')

        # Join X and Y
        xy_df = self.df[[tx + '_transcriptomics' for tx in proteins_to_include]].join(y_df, how='inner').copy().dropna()

        # Skip proteins with fewer than 20 samples
        if len(xy_df) < 20:
            return None

        # Do train-test split
        x = xy_df.iloc[:, :-1]
        y = xy_df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

        # Select model based on the training method
        if self.train_method == 'linreg':
            model = LinearRegression(n_jobs=-1)
        elif self.train_method == 'elastic':
            model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.95], cv=5, fit_intercept=False, n_jobs=-1)
        elif self.train_method == 'boosting':
            model = GradientBoostingRegressor(n_estimators=1000, max_depth=3, subsample=0.5, learning_rate=0.025, random_state=2)
        else:
            raise Exception('Invalid training method')

        # Train the model
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Calculate evaluation metrics
        if np.std(y_test_pred) == 0:
            corr_test = 0
        else:
            corr_test = np.corrcoef(y_test, y_test_pred)[0][1]
        r2_test = r2_score(y_test, y_test_pred)
        nrmse = np.sqrt(mean_squared_error(y_test, y_test_pred) / (np.max(y_test) - np.min(y_test)))

        # Create a DataFrame to store metrics
        metrics_df = pd.DataFrame({
            'train_method': [self.train_method],
            'protein': [protein_to_do],
            'corr_test': [corr_test],
            'r2_test': [r2_test],
            'num_samples': [len(xy_df)],
            'nrmse': [nrmse]
        })

        return metrics_df
# Create an instance of LearnCPTAC with your dataset
#z_df_latest = pd.read_csv('path/to/your/z_df_latest.csv')  # Example input, in this case I upload the downloaded and scaled/transformed data
comb = LearnCPTAC(z_df_fixed)

# Train with multiple methods and gather metrics
all_methods = ['linreg', 'elastic', 'boosting']
results = comb.learn_all_proteins(methods=all_methods)

# Save to a CSV file
results.to_csv('combined_metrics.csv', index=True)
