from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
from werkzeug.utils import secure_filename

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


app = Flask(__name__)
app.secret_key = 'ta_clef_secrete'  # nécessaire pour utiliser session

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['dataset']
        target_column = request.form['target']
        model_choice = request.form['model']
        task = request.form['task']  # 'classification' ou 'regression'

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Lire dataset
            df = pd.read_csv(filepath)

            if target_column not in df.columns:
                return "Colonne cible introuvable.", 400

            X = df.drop(columns=[target_column])
            y = df[target_column]

            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            os.makedirs('static/plots', exist_ok=True)  # Crée dossier plots s'il n'existe pas

            if task == 'classification':
                if model_choice == 'random_forest':
                    model = RandomForestClassifier()
                elif model_choice == 'xgboost':
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                elif model_choice == 'logistic_regression':
                    model = LogisticRegression(max_iter=1000)
                else:
                    return "Modèle non reconnu pour classification", 400

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                matrix = confusion_matrix(y_test, y_pred)

                # Matrice de confusion heatmap
                plt.figure(figsize=(8,6))
                sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
                plt.title('Matrice de confusion')
                plt.ylabel('Vrai label')
                plt.xlabel('Prédit')
                plt.tight_layout()
                plt.savefig('static/plots/confusion_matrix.png')
                plt.close()

                # Importance des variables (si disponible)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    features = X.columns

                    plt.figure(figsize=(10, 6))
                    plt.barh(features, importances, color='skyblue')
                    plt.xlabel("Importance")
                    plt.title("Importance des variables")
                    plt.tight_layout()
                    plt.savefig("static/plots/feature_importance.png")
                    plt.close()

                session['task'] = task
                session['accuracy'] = round(accuracy * 100, 2)
                session['report'] = report
                session['matrix'] = matrix.tolist()

            elif task == 'regression':
                if model_choice == 'random_forest':
                    model = RandomForestRegressor()
                elif model_choice == 'xgboost':
                    model = XGBRegressor()
                else:
                    return "Modèle non reconnu pour régression", 400

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Scatter plot Réel vs Prédit
                plt.figure(figsize=(8,6))
                plt.scatter(y_test, y_pred, alpha=0.6)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel('Valeurs Réelles')
                plt.ylabel('Valeurs Prédites')
                plt.title('Valeurs Réelles vs Prédites')
                plt.tight_layout()
                plt.savefig('static/plots/regression_scatter.png')
                plt.close()

                session['task'] = task
                session['mse'] = round(mse, 4)
                session['r2'] = round(r2, 4)

            else:
                return "Tâche non reconnue", 400

            return redirect(url_for('visualisations'))

    return render_template('upload.html')



@app.route('/visualisations')
def visualisations():
    task = session.get('task', None)

    if task == 'classification':
        accuracy = session.get('accuracy', None)
        report = session.get('report', None)
        matrix = session.get('matrix', None)
        return render_template("visualisations.html", task=task, accuracy=accuracy, report=report, matrix=matrix)

    elif task == 'regression':
        mse = session.get('mse', None)
        r2 = session.get('r2', None)
        return render_template("visualisations.html", task=task, mse=mse, r2=r2)

    else:
        return "Aucun résultat à afficher.", 400


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        print(f"Message reçu de {name} ({email}): {message}")

        # Ici, tu peux ajouter une fonction pour envoyer un mail ou stocker le message

        return "Merci pour votre message ! Nous vous répondrons bientôt."

    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
