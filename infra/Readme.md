
# Generate Data Analytics File

Scripts in the infra folder are used to automatically generate data for the analytics platform.
The workflow `produce_data.yml` is triggered upon the completion of other workflows and sends analytic data for that workflow.

Steps:
- Download the workflow and job information using the GitHub API
- Download workflow artifacts
- Download job logs
- Call `generate_data.py` to create a report file in JSON format
- Upload the file to the data analytics SFTP server

To run this manually, execute the following commands from the root folder:
```
./infra/download_workflow_data.sh tenstorrent/tt-forge-fe 11236784732 1
GITHUB_EVENT_NAME=test python infra/src/generate_data.py --run_id 11236784732
```
Where 11236784732 is the run_id of workflow we are generating data for

## Running tests for infra code

To run infra script tests install python dependancies and run pytest.

```
cd infra
pytest --junitxml=pytest.xml --cov-report=term-missing --cov=src
```

## Running jobs manually

You can run the scripts locally, first to download the necesary files, and seccond one to generate report
```
./infra/download_workflow_data.sh tenstorrent/tt-forge-fe 11253719387 1
GITHUB_EVENT_NAME=test python3 infra/src/generate_data.py --run_id 11253719387
```

## Manually trigger data workflow

You can also trigger data workflow manually for some commit to test
```
gh workflow run "[internal] Produce analytic data" --ref vmilosevic/data_collection -f test_workflow_run_id=11253719387
```
