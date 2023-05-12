#%%
import mlflow 
from typing import List, Dict, Tuple
from mlflow.entities import *
from datetime import datetime
#1. Setup Client
#2. Create The Model Registry
#3. Get Runs 
#4. Filter based the RMSE <= 10
#5. Get the Run Id and Rmse ( For lineAge of model)
#6. Register the models !
#7. Make all to staging
#8. Test All Models !
#%%
TRACKING_URI : str = "sqlite:///durationPred.db"
REGISTERED_MODEL_NAME : str = "NYC-Regressor-model"
CURRENT_DATE : str = (datetime.now().date())
FILTER_STRING : str = "metrics.RMSE <= 7"
experiments_id : List[str] = ['1']
client = mlflow.MlflowClient(tracking_uri=TRACKING_URI)

try:
    client.create_registered_model(REGISTERED_MODEL_NAME, tags=dict(developer="primitif"),
                                    description = f"NYC Model Manager Brah, Created at : {CURRENT_DATE}")
except Exception as e:
    print(e)

                               
top_best_runs : List[Tuple[Run, str]] = []
runs = client.search_runs(
    experiment_ids= experiments_id,
    filter_string= FILTER_STRING,
    order_by=["metrics.RMSE ASC"],
    max_results=10
)

for run in runs :
    top_best_runs.append(
        (run.data.metrics["RMSE"], run.info.run_id))
print(len(top_best_runs))
#%%

for i in range(len(top_best_runs)):
    mlflow.register_model(model_uri= f"runs:/{top_best_runs[i][1]}/model",
                          name=REGISTERED_MODEL_NAME)
#%%
## Go to Staging
STAGING = "Staging"
PRODUCTION = "Production"
for  i in range(len(top_best_runs)):
    client.transition_model_version_stage(REGISTERED_MODEL_NAME,
                                          i+1,
                                          STAGING,
                                          False)

#%% TODO 
# Basically Testing this models to fit the saved data  ( But my code didnt save the dataVector) So later
client.transition_model_version_stage(REGISTERED_MODEL_NAME, 1, PRODUCTION, False)

# %%