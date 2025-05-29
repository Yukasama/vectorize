# import time
# import pytest
# from vectorize.training.models import TrainingTask
# from vectorize.common.task_status import TaskStatus

# @pytest.mark.training
# def test_training_task_status_failed(client, db_session):
#     """Test that a failed training task is marked as FAILED in the database."""
#     payload = {
#         "model_path": "data/models/does_not_exist",  # Invalid path to force failure
#         "dataset_paths": [
#             "data/datasets/__rm_-rf__2F_0a9d5e87-e497-4737-9829-2070780d10df.csv"
#         ],
#         "output_dir": "data/models/trained_models/should_not_exist",
#         "epochs": 3,
#         "learning_rate": 0.00005,
#         "per_device_train_batch_size": 8
#     }
#     response = client.post("/training/train", json=payload)
#     assert response.status_code == 202
#     task_id = response.json()["task_id"]

#     # Poll the DB for the task status (wait for background task to finish)
#     for _ in range(20):
#         db_session.expire_all()
#         task = db_session.get(TrainingTask, task_id)
#         if task and task.task_status == TaskStatus.FAILED:
#             break
#         time.sleep(0.5)
#     else:
#         assert False, "Task did not reach FAILED status in time"

#     assert task.error_msg
#     assert "model path" in task.error_msg.lower()
