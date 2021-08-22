# NM Test

## Questions & Answers
> Please make note of the major choices you make (tools, techniques, designs & assumptions)


#### Tools:
- Using jupyter notebook to explore the dataset
- Using library h5py to read/write metadata in the h5 files
- Using matplotlib to visualise the generated/aggregated marks

#### Assumptions:
- Assuming we are training a CNN model to label Roof and Solar panel, we are only interest the labels marked as 'present' and 'not present', 
hence in the code, I have normalised all values >= 1 to just 1, everything else is 0. 
- At the moment, the 1s are just a single pixel in the center of a grid, I assume when training the CNN model, you would fill all the pixels in the grids so that the 1s will cover the entire roof/solar panel.

> If you have used frameworks then write a brief note to describe how your solution will be deployed to the cloud.

- This solution didn't use any framework, to scale up to a bigger dataset, I think we could use AWS S3 bucket to host the dataset, hook a AWS Lambda function to the s3 bucket so that when new jobs added to the bucket, the aggregation function will be triggered, and in term archive the result to another s3 bucket.
- If processing a large amount of data in a short time is critical, some sort of parallelism will be needed - potentially using Spark/Kubernetes to distribute the data processing may be a better solution, however I have very limited experience working with Spark/Kubernetes at the moment.   







## Installation
```
> git clone https://github.com/mondor/nmtest.git
> cd nmtest
> conda env create -n nmtest -f requirements.yml
> conda activate nmtest 
```
Unzip the Test Dataset "New_Data", move it into the "data" folder, such that the project 
has the following structure:
```
.
│   
├── nmtest
├── tests
├── main.py   
├── data                     
│   ├── New_Data
│   │   ├── 1011
│   │   │   ├── 1_1011_2013-06-29.tar
│   │   │   └── 2_1011_2013-06-29.tar
│   │   │   └── ...
│   │   ├── 1012
│   │   ├── 1029
│   │   ├── ...
│   │   ├── attribute_manifest.csv

```

## Execute the program
```
> cd nmtest
> conda activate nmtest
> python main.py
```

## Run Tests
```
> cd nmtest
> conda activate nmtest
> PYTHONPATH=. pytest tests/
```

## Type checks
```
> cd nmtest
> conda activate nmtest
> stubgen nmtest
> python -m mypy main.py 
```

## Visualise the aggregated marks
```
> cd nmtest
> conda activate nmtest
> jupyter notebook
> # Open notebook Visualise.ipynb on your browser
> # Execute all the ceils 
> # Or Open Visualise.html to see my output

```