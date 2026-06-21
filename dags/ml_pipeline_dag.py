from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "jesseline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="credit_risk_ml_pipeline",
    default_args=default_args,
    description="End-to-end credit risk ML pipeline",
    start_date=datetime(2024, 7, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mle", "credit-risk"],
) as dag:

    data_processing = BashOperator(
        task_id="data_processing",
        bash_command="cd /app && python main_processing_data.py",
    )

    build_training_dataset = BashOperator(
        task_id="build_training_dataset",
        bash_command="cd /app && python training_Dataset.py",
    )

    model_training = BashOperator(
        task_id="model_training",
        bash_command="cd /app && python model_train.py",
    )

    model_inference = BashOperator(
        task_id="model_inference_jul_to_dec_2024",
        bash_command="""
        cd /app &&
        for d in 2024-07-01 2024-08-01 2024-09-01 2024-10-01 2024-11-01 2024-12-01
        do
          python model_inference.py --snapshotdate "$d"
        done
        """,
    )

    model_monitoring = BashOperator(
        task_id="model_monitoring",
        bash_command="""
        cd /app &&
        python model_monitoring.py \
        --predictionpath datamart/gold/model_predictions/champion_model \
        --labelstorepath datamart/gold/label_store \
        --outputpath datamart/gold/model_monitoring \
        --monitoringstart 2024-07-01 \
        --monitoringend 2024-12-01
        """,
    )

    data_processing >> build_training_dataset >> model_training >> model_inference >> model_monitoring