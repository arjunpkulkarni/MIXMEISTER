import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_mixing_scores(data_frame, relevant_features):
    weights = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    feature_values = data_frame[relevant_features]

    print("Feature Values DataFrame:")
    print(feature_values)
    
    print("Shape of feature_values:", feature_values.shape)
    print("Shape of weights:", weights.shape)
    
    mixing_scores = np.dot(feature_values, weights)
    print("Mixing Scores:", mixing_scores)
    return mixing_scores


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def main():
    print("[INFO] Extracting Arguments")
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args, _ = parser.parse_known_args()

    print("[INFO] Reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    print("Selecting relevant features")
    relevant_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo', 'valence']

    # Generate mixing scores using the function from mixing_scores_generator.py
    train_df['mixing score'] = generate_mixing_scores(train_df, relevant_features)

    X_train = train_df[relevant_features]
    y_train = train_df['mixing score']

    print(train_df.head())

    # Train Linear Regression Model
    print("Training Model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model persisted at:", model_path)

    y_pred_missing = []  # Initialize y_pred_missing
    
    # Fill -999.0 and NaN values in 'mixing score' column of test data with predicted scores
    X_missing = test_df[relevant_features][(test_df['mixing score'] == -999.0) | (test_df['mixing score'].isnull())]
    if not X_missing.empty:
        y_pred_missing = model.predict(X_missing)

        missing_indices = test_df.index[(test_df['mixing score'] == -999.0) | (test_df['mixing score'].isnull())]
        print("Missing Indices:", missing_indices)
    
        for idx, pred_score in zip(missing_indices, y_pred_missing):
            test_df.at[idx, 'mixing score'] = pred_score

        print("Updated test_df:")
        print(test_df.head())

        # Print 'mixing score' column before and after filling -999.0 and NaN values
        print("Before filling -999.0 and NaN:")
        print(test_df[(test_df['mixing score'] == -999.0) | (test_df['mixing score'].isnull())]['mixing score'])
    
        # Fill -999.0 and NaN values as before
        print("After filling -999.0 and NaN:")
        print(test_df[(test_df['mixing score'] == -999.0) | (test_df['mixing score'].isnull())]['mixing score'])


    # Save the modified test dataset
    modified_test_path = os.path.join(args.test, args.test_file)
    test_df.to_csv(modified_test_path, index=False)
    print("Modified test dataset saved at:", modified_test_path)

    print(test_df.head())


if __name__ == "__main__":
    main()
