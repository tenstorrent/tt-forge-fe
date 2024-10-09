
# Generate data analytics file

Scripts in infra folder are used to automatically generate data for analytics platform.
_produce_data.yml workflow is trieegered on completition of other workflows and sends analytic data for that workflow.

Steps:
- Download the workflow and job information using github API
- Download workflow artifacts
- Download job logs
- Call generate_data.py to create report file in json format
- Upload file to data anlytics SFTP server 

To run this manually call from root folder
```
./infra/download_workflow_data.sh tenstorrent/tt-forge-fe 11236784732 1
GITHUB_EVENT_NAME=test python infra/src/generate_data.py --run_id 11236784732
```
Where 11236784732 is the run_id of workflow we are generating data for

## Testing infra code

To run infra script tests install python dependancies and run pytest.

```
cd infra
pytest --junitxml=pytest.xml --cov-report=term-missing --cov=src  
```
