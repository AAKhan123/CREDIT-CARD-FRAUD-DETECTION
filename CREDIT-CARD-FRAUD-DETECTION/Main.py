import flet as ft
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def main(page: ft.Page):
    page.title = "CREDIT CARD FRAUD DETECTION"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.CrossAxisAlignment.START  # Ensure content starts from the top
    page.scroll = ft.ScrollMode.AUTO  # Enable scrolling for the whole page

    uploaded_file = None
    file_picker = ft.FilePicker(on_result=lambda e: handle_file_picker_result(e))
    page.overlay.append(file_picker)  # Ensure file picker is added to the page overlay

    def upload_file_dialog(e):
        nonlocal uploaded_file
        file_picker.pick_files(allowed_extensions=["csv"])
        page.update()
    
    def handle_file_picker_result(e):
        nonlocal uploaded_file
        if e.files:
            uploaded_file = e.files[0].path
            update_data_analysis_tab()
            page.update()
    
    def update_data_analysis_tab():
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                head_text.value = str(df.head())
                tail_text.value = str(df.tail())
                describe_text.value = str(df.describe())
                missing_data_text.value = str(df.isnull().sum())
                outliers_text.value = str(detect_outliers(df))
            except Exception as ex:
                head_text.value = f"Error loading data: {str(ex)}"
                tail_text.value = ""
                describe_text.value = ""
                missing_data_text.value = ""
                outliers_text.value = ""
        else:
            head_text.value = ""
            tail_text.value = ""
            describe_text.value = ""
            missing_data_text.value = ""
            outliers_text.value = ""
        
        page.update()
    
    def detect_outliers(df):
        outliers = []
        for column in df.select_dtypes(include='number').columns:  # Only numeric columns
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers.extend(df[(df[column] < lower_bound) | (df[column] > upper_bound)][column].values.tolist())
        return outliers
    
    def classify(e):
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                target_column = 'Class'
                
                if target_column not in df.columns:
                    classification_results_text.value = f"Error: '{target_column}' column not found in the dataset."
                    page.update()
                    return
                
                X = df.drop(target_column, axis=1)  # Assuming the target variable is named 'Class'
                y = df[target_column]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                if model_selected.value == "Model 1 (Logistic Regression)":
                    model = LogisticRegression(max_iter=200)  # Increase max_iter if needed
                elif model_selected.value == "Model 2 (Decision Tree)":
                    model = DecisionTreeClassifier()
                else:
                    classification_results_text.value = "Error: Model selection not recognized."
                    page.update()
                    return
                
                model.fit(X_scaled, y)
                predictions = model.predict(X_scaled)
                prediction_df = pd.DataFrame({"Actual": y, "Predicted": predictions})
                classification_report_str = classification_report(y, predictions)
                accuracy = accuracy_score(y, predictions)
                
                if model_selected.value == "Model 1 (Logistic Regression)":
                    fraud_indices = prediction_df[(prediction_df['Actual'] == 1) & (prediction_df['Predicted'] == 1)].index
                    non_fraud_indices = prediction_df[(prediction_df['Actual'] == 0) & (prediction_df['Predicted'] == 0)].index
                    fraud_text = f"Fraudulent transactions:\n{df.loc[fraud_indices]}"
                    non_fraud_text = f"Non-fraudulent transactions:\n{df.loc[non_fraud_indices]}"
                    classification_results_text.value = f"Logistic Regression Classification results:\n\n{fraud_text}\n\n{non_fraud_text}\n\n{classification_report_str}\n\nAccuracy: {accuracy}"
                
                elif model_selected.value == "Model 2 (Decision Tree)":
                    fraud_indices = prediction_df[(prediction_df['Actual'] == 1) & (prediction_df['Predicted'] == 1)].index
                    non_fraud_indices = prediction_df[(prediction_df['Actual'] == 0) & (prediction_df['Predicted'] == 0)].index
                    fraud_text = f"Fraudulent transactions:\n{df.loc[fraud_indices]}"
                    non_fraud_text = f"Non-fraudulent transactions:\n{df.loc[non_fraud_indices]}"
                    classification_results_text.value = f"Decision Tree Classification results:\n\n{fraud_text}\n\n{non_fraud_text}\n\n{classification_report_str}\n\nAccuracy: {accuracy}"
                
            except Exception as ex:
                classification_results_text.value = f"Error during classification: {str(ex)}"
        else:
            classification_results_text.value = "Error: No file uploaded."
        
        page.update()
    
    head_text = ft.Text(value="", size=15, weight="bold")
    tail_text = ft.Text(value="", size=15, weight="bold")
    describe_text = ft.Text(value="", size=15, weight="bold")
    missing_data_text = ft.Text(value="", size=15, weight="bold")
    outliers_text = ft.Text(value="", size=15, weight="bold")
    classification_results_text = ft.Text(value="", size=15, weight="bold")
    
    model_selected = ft.Dropdown(
        width=300,
        options=[
            ft.dropdown.Option("Model 1 (Logistic Regression)"),
            ft.dropdown.Option("Model 2 (Decision Tree)"),  # Update the dropdown option
        ],
    )
    
    page.add(
        ft.Tabs(
            tabs=[
                ft.Tab(
                    text="Data Analysis",
                    content=ft.Column(
                        [
                            ft.Text("Upload CSV File:"),
                            ft.ElevatedButton("Browse", on_click=upload_file_dialog),
                            ft.Divider(),
                            ft.Text("Head:"),
                            head_text,
                            ft.Divider(),
                            ft.Text("Tail:"),
                            tail_text,
                            ft.Divider(),
                            ft.Text("Describe:"),
                            describe_text,
                            ft.Divider(),
                            ft.Text("Missing Data:"),
                            missing_data_text,
                            ft.Divider(),
                            ft.Text("Outliers:"),
                            outliers_text,
                        ],
                        scroll=True,
                    ),
                ),
                ft.Tab(
                    text="Classification",
                    content=ft.Column(
                        [
                            ft.Text("Choose a Model:"),
                            model_selected,
                            ft.ElevatedButton("Classify", on_click=classify),
                            ft.Divider(),
                            ft.Text("Classification Results:"),
                            classification_results_text,
                        ],
                        scroll=True,
                    ),
                ),
            ]
        )
    )

ft.app(target=main)
