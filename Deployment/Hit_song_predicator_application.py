import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tempfile
import mlflow
import mlflow.sklearn


#Running MLFLOW: mlflow ui --backend-store-uri "file:///C:/Users/Hani/OneDrive/Desktop/Dalas/TODAY DALAS/mlruns

st.title("🎵 Hit Song Prediction")

st.sidebar.header("About")
st.sidebar.info(
    "This app predicts whether a song will be a hit based on various features of the song's lyrics and other characteristics. "
    "You can upload your own dataset, choose a model, and evaluate its performance on the test data."
)

model_option = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Networks"])

if model_option == "Random Forest":
    st.sidebar.subheader("🌲 Random Forest Classifier")
    st.sidebar.markdown(
        "Random Forest is an **ensemble learning method** that combines the predictions of several base estimators "
        "to improve the overall performance and avoid overfitting. It works well for both classification and regression tasks."
    )
    st.sidebar.markdown(
        "🔑 **Key Features:**\n"
        "- Combines multiple decision trees\n"
        "- Reduces overfitting\n"
        "- Works well with both small and large datasets"
    )
elif model_option == "Gradient Boosting":
    st.sidebar.subheader("⚡ Gradient Boosting Classifier")
    st.sidebar.markdown(
        "Gradient Boosting is a **boosting algorithm** that builds an ensemble of decision trees by training each tree "
        "to correct the errors made by the previous one. It is known for its high performance, especially on small to medium datasets."
    )
    st.sidebar.markdown(
        "🔑 **Key Features:**\n"
        "- Focuses on reducing bias and variance\n"
        "- Often outperforms Random Forest on structured data\n"
        "- Requires careful tuning"
    )
elif model_option == "Neural Networks":
    st.sidebar.subheader("🧠 Neural Networks")
    st.sidebar.markdown(
        "Neural Networks are a powerful class of models that simulate the human brain's structure to learn complex patterns. "
        "They are especially useful for non-linear relationships and large datasets."
    )
    st.sidebar.markdown(
        "🔑 **Key Features:**\n"
        "- Excels at capturing non-linear patterns\n"
        "- Requires more data and tuning\n"
        "- Can be used for both regression and classification"
    )
else:
    st.sidebar.subheader("🔗 Logistic Regression")
    st.sidebar.markdown(
        "Logistic Regression is a **statistical method** for binary classification. It estimates the probability that an observation "
        "belongs to one of two classes based on input features."
    )
    st.sidebar.markdown(
        "🔑 **Key Features:**\n"
        "- Simple and easy to implement\n"
        "- Suitable for binary classification\n"
        "- Outputs probabilities for predictions"
    )

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    drop_cols = ['artist','song','release_date','1st_word','longest_word','anomaly',
                 '2nd_word','3rd_word','4th_word','5th_word','5th_word_percentage',
                 '4th_word_percentage','popularity','year','1st_occurence','2nd_occurence',
                 '3rd_occurence','4th_occurence','5th_occurence']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    st.write("### Preview of Cleaned Data", df.head())

    X = df.drop('hit', axis=1)
    y = df['hit']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    if model_option == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_option == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    elif model_option == "Neural Networks":
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    else:
        model = LogisticRegression(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("📊 Model Evaluation")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write("**Confusion Matrix:**", conf_matrix)
    st.write("**Classification Report:**")
    st.json(class_report)

    correct_percentage = (y_test == y_pred).mean() * 100
    incorrect_percentage = (y_test != y_pred).mean() * 100

    st.write(f"✅ Correct Predictions: {correct_percentage:.2f}%")
    st.write(f"❌ Incorrect Predictions: {incorrect_percentage:.2f}%")

    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    mlflow.set_experiment("Hit Song Prediction v4")

    # MLflow integration
    with mlflow.start_run():
        mlflow.log_param("model", model_option)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        # Log confusion matrix
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig_cm.savefig(tmp.name)
            mlflow.log_artifact(tmp.name, "confusion_matrix")

        st.success("✅ Metrics and model logged to MLflow")

    st.subheader("🎯 Predict if a Song Will Be a Hit")

    input_data = {}
    feature_names = X.columns

    st.write("### Input Song Features")
    for feature in feature_names:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        input_data[feature] = st.slider(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

    if st.button("Predict Hit Probability"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)
        prediction = model.predict(input_pca)[0]
        prediction_proba = model.predict_proba(input_pca)[0][1]

        st.success(f"🎵 **Prediction:** {'Hit' if prediction == 1 else 'Not a Hit'}")
        st.info(f"📈 **Probability of Hit:** {prediction_proba:.2%}")

    if st.button("Save Model"):
        joblib.dump(model, 'hit_song_model.pkl')
        st.success("Model saved as hit_song_model.pkl")
