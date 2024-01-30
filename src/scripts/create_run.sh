mlflow run --experiment-name $1 \
    --entry-point $2 \
    -P dataset_path=$3 \
    -P model_name=$4