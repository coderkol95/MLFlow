import mlflow
import json

table=mlflow.load_table(artifact_file='comparison_table.json')
table['overall_score'] = table['r2']-table['mae']-table['rmse']
table.sort_values(['overall_score'], ascending=False, inplace=True)
print(table.head())

top_model={"runID":table.head(1)['runID'].values[0],
           "alpha":table.head(1)['alpha'].values[0],
           "l1_ratio":table.head(1)['l1_ratio'].values[0]}

print(f"Top model from run {top_model['runID']} with alpha {top_model['alpha']} and l1_ratio {top_model['l1_ratio']}")

with open("top_model.json","w") as f:
    json.dump(top_model,f)

result = mlflow.register_model(
    f"runs:/{top_model['runID']}/model", "dummy_data_predictor",tags={"alpha":top_model['alpha'],'l1_ratio':top_model['l1_ratio']}
)