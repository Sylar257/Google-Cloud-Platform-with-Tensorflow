# Google Cloud Platform guide

Steps to set up fresh project workspace.

## Create Storage Bucket

![Create_storage_bucket](images/Create_storage_bucket.png)

## Launch AI Platform Notebooks

#### Step 1

Click on **Navigation Menu**. Navigate to **AI Platform**, then to **Notebooks**.

![Open new notebook](https://cdn.qwiklabs.com/jH5%2FIspr88moxsjUKc2TQdpRvqUfNvj098DG3ZRQe%2B4%3D)

#### Step 2

On the Notebook instances page, click ![NEW INSTANCE](https://cdn.qwiklabs.com/YI0InqyQhTRNsEIGzrufyXjMtsdrwKwspeNXtPlPPeY%3D). Create the environment configuration that we want. Specify Region and zone, GPU and RAM configurations in the **CUSTOMIZE** settings.

![Create new VM](https://cdn.qwiklabs.com/%2BrmInxmD0MC2J6eKAClJVNuLvRJFR4vHFLsamqi%2F9wY%3D)

#### Step 3

Click **Open JupyterLab**. A JupyterLab window will open in a new tab

![JupyterLab](https://cdn.qwiklabs.com/pwhtoT2OOAq8DamQHYjk1TYcoAuOQTwkoYM5qglV1AQ%3D)

From here, you will have a **Jupyter notebook environment** set-up and running in GCP. Of course, we can clone our existing notebooks(e.g. from *github*) into this workspace.

## Exploratory Data Analysis (EDA)

When working with large datasets, it’s usually not idea to have everything stored in the local PC. What can easily do in this situation is to use **SQL** to load our data of interest into a *Pandas DataFrame* with limited volume(say only 200 entries.)

![SQL_load_data](images/SQL_load_data.png)

We can investigate our dataset using a portion of our data. Of course, in reality, we should a bigger volume and make sure to randomly sample them in order to obtain a close estimation of true distribution.

Next up, we can create a handy function that allows us to run **SQL** queries on the entire dataset retrieving only the information that we want to investigate. This helps us to build an intuition of how different **features** correlates with our dependent variable, thus allow us to have a good understanding when choosing the right feature to build our dataset/model later on.

```python
# Create a function that finds any one of the feature's corelation with average weight and number of babies
def get_distinct_values(column_name):
    # {0} refers to the first input variable. In this case, "column_name"
    # This happens because the .format() at the end
    query = """
    SELECT
        {0},
        COUNT(1) AS num_babies,
        AVG(weight_pounds) AS avg_wt
    FROM
        publicdata.samples.natality
    WHERE
        year>2000
    GROUP BY
        {0}
    """.format(column_name)
    return pd.io.gbq.read_gbq(query, project_id=project_id, dialect='standard')
```

For instance, we take take a look at the co-relation between the `mother_age` and `num_babies` & `avg_wt`.

![Investigate_mothers_age](images/Investigate_mothers_age.png)

and also theco-relation between the `plurality` and `num_babies` & `avg_wt`.

![Investigate_plurality](images/Investigate_plurality.png)

## Creating our dataset

#### Criterion of a good feature

When creating a dataset for our machine model, we don’t have to use all of the features comes with the original source. We have the freedom to pick the most “relevant” features or to compose new one based on old features, both of these two steps might have a positive impact on our model’s performance.

What *features* should we consider to incorporate:

*   Be related to the **objective**

*   Be known at **prediction-time**
*   Be **numeric** with meaningful magnitude(or we should do the feature engineering for conversion)
*   Have enough examples
*   Bring human insight to the problem

#### `FARM_FINGERPRINT(value)`

This is a **Hash Function** in SQL that takes in a value(such as a *date* or a *string*) and computes the *fingerprint* of it(output format is **hashed numeric type**. *The output of this function for a particular input will never change*. Which means, `FARM_FINGERPRINT('26-jan-2012')` is always going to return the same value whenever we call.(and it’s different from calling on all other values)

Here is a example of usage:

```sql
#standardSQL
SELECT
	data,
	airline,
	departure_airport,
	departure_schedule,
	arrival_airport,
	arrival_delay
FROM
	`bigquery-samples.airline_ontime_data.flights`
WHERE
	MOD(ABS(FARM_FINGERPRINT(date)),10)<8
```

The last line, would give us **randomly** sampled 80% of data. In addition, we can assign `MOD(ABS(FARM_FINGERPRINT(date)),10)=8` to **valid_set** and `MOD(ABS(FARM_FINGERPRINT(date)),10)=9` to **test_set**, so that we have a randomly spitted 80%-10%-10% dataset where **the same date will not appear in different set bracket** so as to avoid data leakage.

Here is another trick: when dealing with large datasets, we should always be prototyping on a small(randomly sampled) fraction of the entire dataset so that we can iterate fast. Once we have finalized the model design then we deploy the model on the entire original dataset. `RAND()` can easily help us to do just that:

```SQL
#standardSQL
SELECT
	data,
	airline,
	departure_airport,
	departure_schedule,
	arrival_airport,
	arrival_delay
FROM
	`bigquery-samples.airline_ontime_data.flights`
WHERE
	MOD(ABS(FARM_FINGERPRINT(date)),10)<8 AND RAND() < 0.01
```

The `RAND()` will generate a random number between `0` and `1` each time, and we are only keeping the query data if the value is smaller than 0.01 which means it’s a 1% chance. Hence, we are getting 1% of the entire dataset.

