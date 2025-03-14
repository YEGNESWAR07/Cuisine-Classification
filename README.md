# Cuisine-Classification

## Description
This Code leverages powerful machine learning techniques to develop an intelligent restaurant recommendation system. By combining content-based filtering with a Random Forest Classifier, it aims to deliver personalized dining suggestions based on user preferences. Here’s a step-by-step breakdown of the process:

1.**Data Loading and Preprocessing:**
   - Import essential libraries like `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.
   - Load the dataset and handle missing values, especially for categorical variables.

2. **Categorical Encoding**
   - Identify categorical columns and employ `OneHotEncoder` to transform them into numerical format.
   - Merge encoded features back into the original DataFrame, ensuring a complete numerical dataset.

3. **Feature and Target Definition**
   - Define the features (`X`) and the target variable (`y`). The target is derived from the 'Cuisines' column, utilizing a simple label encoding approach.

4. **Model Training and Validation**
   - Split the dataset into training and testing sets.
   - Train the Random Forest Classifier with optimized parameters for better performance.
   - Make predictions using the trained model and calculate essential metrics: accuracy, precision, recall, and F1-score.

5. **Evaluation and Visualization**
   - Generate a classification report and confusion matrix to evaluate model performance.
   - Plot precision-recall curves and feature importance to gain insights into model behavior.

Sure, here are the sections for usage, installation, and license:

## 🖥️ Usage
To use this restaurant recommendation system, follow these steps:

1. **Load the Dataset**
   ```python
   import pandas as pd

   df = pd.read_csv('path_to_your_dataset.csv')
   ```

2. **Preprocess the Data**
   ```python
   # Handle missing values
   for col in df.select_dtypes(include=['object']).columns:
       df[col] = df[col].fillna('Unknown')

   # Encode categorical variables
   from sklearn.preprocessing import OneHotEncoder
   encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
   encoded_features = encoder.fit_transform(df[categorical_cols])

   feature_names = encoder.get_feature_names_out(categorical_cols)
   encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
   df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
   ```

3. **Define Features and Target**
   ```python
   # Define target variable
   df_temp = pd.read_csv('path_to_your_dataset.csv')
   df_temp['Cuisines'] = df_temp['Cuisines'].fillna('Unknown')
   target_labels = df_temp['Cuisines'].factorize()[0]
   df['cuisine_label'] = target_labels

   X = df.drop('cuisine_label', axis=1)
   y = df['cuisine_label']
   ```

4. **Train the Model**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   rf_model = RandomForestClassifier(
       n_estimators=100,
       max_depth=10,
       min_samples_split=5,
       min_samples_leaf=2,
       random_state=42
   )
   rf_model.fit(X_train, y_train)
   ```

5. **Make Predictions and Evaluate**
   y_pred = rf_model.predict(X_test)

   from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

   accuracy = accuracy_score(y_test, y_pred)
   precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
   recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
   report = classification_report(y_test, y_pred, zero_division=0)

   print(f"Accuracy: {accuracy:.2f}")
   print(f"Precision: {precision:.2f}")
   print(f"Recall: {recall:.2f}")
   print("\nClassification Report:")
   print(report)
  

## 🔧 Installation
To install and set up the project, follow these steps:

1. **Create and activate a virtual environment**
   ```bash
   python -m venv restaurant_env
   ```

   - For Windows:
     ```bash
     restaurant_env\Scripts\activate
     ```
   - For macOS/Linux:
     ```bash
     source restaurant_env/bin/activate
     ```

2. **Install required packages**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   pip install jupyter notebook
   ```

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for more details.

    

