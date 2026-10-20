import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 1.1- IMPORT AND FIXES/FILTERS
# --------------------------------------------------------------------------

dataset_path = "procrastination_survey_dataset.csv"
target_column = "procrastination_score"


with open(dataset_path, "r", encoding = "utf-8-sig", newline = "") as f:
    csv_rows = list(csv.reader(f))

if not csv_rows:
    raise ValueError ("CSV file not found/empty")



# COLUMN NAMES
header = csv_rows[0]

# (IGNORE BLANK ROWS)
raw_data_rows = [ row for row in csv_rows[1:] if row]
blank_rows_count = len(csv_rows) - len(raw_data_rows) -1

# Expected nº of fields from header
expected_column_count = len(header)

print (f"\nExpected columns from the header: {expected_column_count}")
print (f"\nNon-empty data rows: {len(raw_data_rows)}")
print (f"\nBlank rows left out: {blank_rows_count}\n")


# Counting number of fields in every non-empty data row

row_width_counts = (pd.Series([len(row) for row in raw_data_rows])
                    .value_counts()
                    .sort_index())
    

print (f"\n\nObserved row widths: ")
for field_count, row_count in row_width_counts.items():
    print (f"Rows with {field_count} fields: {row_count}")
print ("\n")

# rows shorter/longer than header
short_rows = [row for row in raw_data_rows
            if len(row) < expected_column_count]
    

overlong_widths = sorted({len(row)
                        for row in raw_data_rows
                        if len(row) > expected_column_count})
    

if short_rows:
    raise ValueError ("Some rows contain fewer fuelds than the header")
if len(overlong_widths) >1:
    raise ValueError ("Several different overlong row widths were found")

# Checking for extra fields
extra_field_count = (overlong_widths[0] - expected_column_count
                     if overlong_widths else 0)

print (f"\nExtra fields per affected row: {extra_field_count}")


# Derive the position of planning_style from the actual header
if "planning_style" not in header:
    raise ValueError("The expected planning_style column is missing.")

planning_style_index = header.index("planning_style")

correctly_aligned_rows = [
    row for row in raw_data_rows
    if len (row) == expected_column_count
]

overlong_rows = [
    row for row in raw_data_rows
    if len(row) > expected_column_count
]


# values = numbers?
def is_numeric_value(value):
    converted_value = pd.to_numeric(
        value,
        errors="coerce"
    )

    return pd.notna(converted_value)


# Inspect the planning_style position in correctly aligned rows
correct_style_text_count = sum(
    isinstance(row[planning_style_index], str)
    and bool(row[planning_style_index].strip())
    and not is_numeric_value(row[planning_style_index])
    for row in correctly_aligned_rows
)

print(
    "\nNon-empty textual values at the planning_style position "
    "in correctly aligned rows:",
    correct_style_text_count
)

# Check whether the extra field appears immediately before planning_style
if extra_field_count > 0:

    numeric_at_expected_position = sum(
        is_numeric_value(row[planning_style_index])
        for row in overlong_rows
    )

    shifted_style_text_count = sum(
        isinstance(
            row[planning_style_index + extra_field_count],
            str
        )
        and bool(
            row[planning_style_index + extra_field_count].strip()
        )
        and not is_numeric_value(
            row[planning_style_index + extra_field_count]
        )
        for row in overlong_rows
    )

    print(
        "Affected rows with a numeric value at the expected "
        "planning_style position:",
        numeric_at_expected_position
    )

    print(
        "Affected rows with a textual value after skipping "
        "the extra fields:",
        shifted_style_text_count
    )

    # Correct only if the observed structure supports this decision
    repair_pattern_verified = (
        correct_style_text_count == len(correctly_aligned_rows)
        and numeric_at_expected_position == len(overlong_rows)
        and shifted_style_text_count == len(overlong_rows)
        and len(overlong_rows) > 0
    )

    if not repair_pattern_verified:
        raise ValueError(
            "The extra-field pattern could not be verified. "
            "Inspect the CSV before attempting a correction."
        )

    print("\nThe extra-field pattern was verified.")

else:
    print("\nNo overlong data rows require correction.")


# which participant rows are corrected
repaired_participant_ids = []


def repair_extra_field(row):
    """
    Remove an extra field only when its position and type
    match the structure verified above.
    """

    if (
        extra_field_count > 0
        and len(row) == expected_column_count + extra_field_count
    ):

        expected_style_value = row[
            planning_style_index + extra_field_count
        ]

        if (
            is_numeric_value(row[planning_style_index])
            and isinstance(expected_style_value, str)
            and expected_style_value.strip()
            and not is_numeric_value(expected_style_value)
        ):

            participant_index = header.index("participant_id")

            repaired_participant_ids.append(
                row[participant_index]
            )

            # Remove the extra field using the derived index
            return (
                row[:planning_style_index]
                + row[planning_style_index + extra_field_count:]
            )

    raise ValueError(
        "An overlong row does not match the verified repair pattern."
    )


# Let Pandas handle the data types after the structural check
bad_line_handler = (
    repair_extra_field
    if extra_field_count > 0
    else "error"
)


# 1.2- EXPLORATION
# --------------------------------------------------------------------------

df = pd.read_csv(
    dataset_path,
    encoding="utf-8-sig",
    engine="python",
    on_bad_lines=bad_line_handler
)

print("\nLoaded dataset shape:", df.shape)
print("Corrected rows:", len(repaired_participant_ids))


# Nº of rows and columns
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Column names and data types
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

# Missing values per column
missing_values = df.isnull().sum()
missing_columns = missing_values[missing_values > 0]

print(f"\nColumns with missing values: {missing_columns}")

# Duplicates
print (f"\nDuplicate complete rows: {df.duplicated().sum()}")
print (f"Duplicate participant ID's: {df["participant_id"].duplicated().sum()}")


# Feature's values statistics
print (f"\nNumerical descriptive statistics: ")
print (df.describe(include=[np.number]).T.round(1))




# 1.3- CATEGORIES AND INCONSISTENCIES
# --------------------------------------------------------------------------

# Identify textual columns from their actual data types
text_columns = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()


# Features and value countability
text_columns = [
    column
    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns
    if column != "participant_id"
]

for column in text_columns:
    print(f"\nCategories in {column}:")
    print(df[column].value_counts(dropna=False))

    print("\n"*3)


# 1.4- INVALID & UNUSUAL VALUES
# --------------------------------------------------------------------------

# Checking for values surpassing 24 hours
hours = 24

per_day_features = [
    "average_sleep_hours",
    "daily_screen_time_hours",
    "daily_social_media_hours",
    "work_hours_per_day",
    "study_hours_per_day",
    "commute_hours_per_day"
]

per_day_features = [
    column
    for column in per_day_features
    if column in df.columns
]

print("Checks for daily-hour variables:")

for column in per_day_features:

    values = pd.to_numeric(df[column], errors="coerce")

    outside_day = (
        (values < 0)
        | (values > hours)
    ).sum()

    invalid_numeric_values = (
        df[column].notna() & values.isna()
    ).sum()

    print(
        f"{column}: "
        f"minimum={values.min()}, "
        f"maximum={values.max()}, "
        f"outside physical limits={outside_day}, "
        f"non-numeric values={invalid_numeric_values}"
    )


# Checking inconsistencies between social-media time and total screen time
if {
    "daily_social_media_hours",
    "daily_screen_time_hours"
}.issubset(df.columns):

    social_media_time = pd.to_numeric(
        df["daily_social_media_hours"],
        errors="coerce"
    )

    screen_time = pd.to_numeric(
        df["daily_screen_time_hours"],
        errors="coerce"
    )

    print(
        "\nRows where social-media time exceeds screen time:",
        (social_media_time > screen_time).sum()
    )


# Check whether work, study and commuting fit within a day
time_columns = [
    "work_hours_per_day",
    "study_hours_per_day",
    "commute_hours_per_day"
]

if set(time_columns).issubset(df.columns):

    daily_commitments = df[time_columns].sum(axis=1)

    print(
        "Rows where work, study and commuting exceed "
        "the hours in a day:",
        (daily_commitments > hours).sum()
    )


# Inspect the observed scale of questionnaire items
procrastination_item_columns = [
    column
    for column in df.columns
    if column.startswith("P") and column[1:].isdigit()
]

if procrastination_item_columns:

    p_items = df[procrastination_item_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    item_minimum = p_items.min().min()
    item_maximum = p_items.max().max()

    target_values = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    outside_item_range = (
        (target_values < item_minimum)
        | (target_values > item_maximum)
    ).sum()

    print("\nObserved range of the P-Set items:")
    print (f"Minimum: {item_minimum}")
    print (f"Maximum: {item_maximum}")

    print(
        "Target scores outside the observed item range:",
        outside_item_range
    )



# 2.1- DATA PREPARATION

print ("DATA PREPARATION", "-"*30)

# Predictors connected to screen time, sleep and lifestyle
feature_columns = [
    # Screen time
    "daily_screen_time_hours",
    "daily_social_media_hours",

    # Sleep
    "average_sleep_hours",
    "sleep_quality",

    # Lifestyle and daily routine
    "physical_activity_frequency",
    "work_hours_per_day",
    "study_hours_per_day",
    "commute_hours_per_day",

    # Digital habits
    "social_media_check_frequency",
    "social_media_during_work_or_study",
    "social_media_as_task_avoidance",
    "notifications_enabled",

    # Planning and wellbeing
    "planning_style",
    "stress_rating",
    "mood_rating",
    "energy_rating",
    "concentration_difficulty",
    "feeling_overwhelmed_frequency",
    "work_life_balance",
    "perceived_control"
]


print (f"Selected predictors: {len(feature_columns)}")
print (f"Target: {target_column}")

print (f"\nSelected feature names: {feature_columns}")


# Explicitly identify columns excluded from modelling
excluded_columns = sorted(
    set(df.columns)
    - set(feature_columns)
    - {target_column}
)

print("\nColumns excluded from modelling:")
print(excluded_columns)

# Identify direct questionnaire items and derived score columns
direct_questionnaire_items = [
    column
    for column in df.columns
    if (
        column.startswith("P")
        or column.startswith("S")
    )
    and column[1:].isdigit()
]

derived_score_columns = [
    column
    for column in df.columns
    if column.endswith("_score")
    and column != target_column
]

print("\nDirect questionnaire items excluded:")
print(direct_questionnaire_items)

print("\nOther derived scores excluded:")
print(derived_score_columns)


model_data = df[
    feature_columns + [target_column]
].copy()

# Check missing values only in the columns required for modelling
missing_model_values = model_data.isnull().sum()

print("\nMissing values in selected columns:")
print(
    missing_model_values[missing_model_values > 0]
    if missing_model_values.any()
    else "None"
)

# Remove incomplete modelling records, if any
rows_before_cleaning = len(model_data)

model_data = model_data.dropna(
    subset=feature_columns + [target_column]
)

rows_removed = rows_before_cleaning - len(model_data)

print("\nRows before cleaning:", rows_before_cleaning)
print("Rows removed:", rows_removed)
print("Rows remaining:", len(model_data))

# Separate predictors and target
x_raw = model_data[feature_columns].copy()

y = pd.to_numeric(
    model_data[target_column],
    errors="raise"
).astype(float)

print("\nPredictor matrix shape:", x_raw.shape)
print("Target shape:", y.shape)



# 2.2- TEST & TRAINING


# Proportion 80-20
test_perc = 0.20


indices = np.random.permutation(len(x_raw))
test_size = int(np.ceil(test_perc * len(x_raw)))

if test_size <= 0 or test_size >= len(x_raw):
    raise ValueError("The train/test split is not valid.")

test_indices = indices[:test_size]
train_indices = indices[test_size:]


x_train_raw = x_raw.iloc[train_indices].copy()
x_test_raw = x_raw.iloc[test_indices].copy()

# Separate the target variable.
y_train = y.iloc[train_indices].to_numpy(dtype=float)
y_test = y.iloc[test_indices].to_numpy(dtype=float)

# Display the resulting dataset sizes.
print(f"\n\nTotal observations: {len(x_raw)}")
print(f"Training observations: {len(x_train_raw)}")
print(f"Testing observations: {len(x_test_raw)}\n\n")



print (f"Training proportion: {round(len(x_train_raw) / len(x_raw), 4)}")
print (f"Testing proportion: {round(len(x_test_raw) / len(x_raw), 4)}\n\n")


# 2.3- CATEGORIZING NUMERIVAL VALUES

categorical_columns = x_train_raw.select_dtypes(include = ["object", "string", "category"]).columns.tolist()
print (f"Categorical predictors: {categorical_columns}\n\n")

# categories --> binary columns
x_train_encoded = pd.get_dummies(
    x_test_raw,
    columns = categorical_columns,
    drop_first = True,
    dtype = float
)

x_test_encoded = pd.get_dummies(
    x_test_raw,
    columns = categorical_columns,
    drop_first = True,
    dtype = float
)

# align test col. & training col.
x_test_encoded = x_test_encoded.reindex(
    columns = x_train_encoded.columns,
    fill_value = 0
)


print (f"Encoded training shape: {x_train_encoded.shape}")
print (f"Encoded testing shape: {x_test_encoded.shape}")
print (f"\n Enconded feature names: {x_train_encoded.columns.tolist()}\n\n")



# 2.4 - DATA PREPARATION FOR LR

x_train_lr = x_train_encoded.copy()
x_test_lr = x_test_encoded.copy()

numerical_columns = x_train_raw.select_dtypes(
    include = np.number
).columns.tolist()

# mean & std from trainig data only
train_means = x_train_lr[numerical_columns].mean()
train_stds = x_train_lr[numerical_columns].std(ddof=0)

# checking for columns with zero-variance
zero_variance_columns = train_stds[
    train_stds == 0
].index.tolist()

print (f"Numerical columns to scale: {numerical_columns}")
print (f"Zero-variance columns: {zero_variance_columns}")


# if std =0, std --> =1
train_stds = train_stds.replace(0,1)

x_train_lr[numerical_columns] = (
    x_test_lr[numerical_columns] - train_means
) / train_stds

print(f"\nScaled training data preview: {x_train_lr.head()}")