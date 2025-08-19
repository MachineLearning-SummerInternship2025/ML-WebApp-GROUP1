import matplotlib
matplotlib.use('Agg')  # Backend pour les environnements sans GUI
import matplotlib.pyplot as plt
import base64
import datetime
from io import BytesIO, StringIO
import json
import os

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify 
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
import traceback
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
import requests




# === Config Flask ===
app = Flask(__name__)
app.secret_key = 'azertyuiop'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

from flask_migrate import Migrate

migrate = Migrate(app, db)

import json
def from_json_filter(s):
    return json.loads(s)

app.jinja_env.filters['from_json'] = from_json_filter

# === Modèles SQLAlchemy ===
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    task_type = db.Column(db.String(50), nullable=False)
    target_column = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float)
    mse = db.Column(db.Float)
    r2 = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    confusion_matrix = db.Column(db.Text)
    roc_curve = db.Column(db.Text)
    regression_plot = db.Column(db.Text)
    feature_importance = db.Column(db.Text)
    preview_data = db.Column(db.Text)


with app.app_context():
    db.create_all()

from sklearn.model_selection import GridSearchCV

PARAM_GRIDS = {
    'classification': {
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        },
        'xgboost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        },
        'logistic_regression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear']
        },
        'decision_tree': {
            'max_depth': [None, 5, 10],
            'criterion': ['gini', 'entropy']
        },
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        'naive_bayes': {},
        'knn': {
            'n_neighbors': [3, 5, 7]
        },
        'dummy': {}
    },
    'regression': {
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        },
        'xgboost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        },
        'linear_regression': {},
        'ridge': {
            'alpha': [0.1, 1.0, 10.0]
        },
        'lasso': {
            'alpha': [0.1, 1.0, 10.0]
        },
        'decision_tree': {
            'max_depth': [None, 5, 10]
        },
        'svr': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        'knn': {
            'n_neighbors': [3, 5, 7]
        },
        'dummy': {}
    }
}

MODEL_OPTIONS = {
    'classification': {
        'random_forest': ('Random Forest', RandomForestClassifier()),
        'logistic_regression': ('Logistic Regression', LogisticRegression(max_iter=1000)),
        'decision_tree': ('Decision Tree', DecisionTreeClassifier()),
        'svm': ('Support Vector Machine', SVC(probability=True)),
        'naive_bayes': ('Naive Bayes', GaussianNB()),
        'knn': ('K-Nearest Neighbors', KNeighborsClassifier()),
        'dummy': ('Dummy Classifier', DummyClassifier(strategy='most_frequent'))
    },
    'regression': {
        'random_forest': ('Random Forest', RandomForestRegressor()),
        'linear_regression': ('Linear Regression', LinearRegression()),
        'ridge': ('Ridge Regression', Ridge()),
        'lasso': ('Lasso Regression', Lasso()),
        'decision_tree': ('Decision Tree', DecisionTreeRegressor()),
        'svr': ('Support Vector Regression', SVR()),
        'knn': ('K-Nearest Neighbors', KNeighborsRegressor()),
        'dummy': ('Dummy Regressor', DummyRegressor(strategy='mean'))
    }
}

MODEL_OPTIONS_3 = {
    'classification': {
        'logistic': ('Logistic Regression', LogisticRegression(max_iter=1000)),
        'knn': ('KNN', KNeighborsClassifier()),
        'rf': ('Random Forest', RandomForestClassifier())
    },
    'regression': {
        'lr': ('Linear Regression', LinearRegression()),
        'rf': ('Random Forest', RandomForestRegressor()),
        'svr': ('SVR', SVR())
    }
}

PARAM_GRIDS1 = {
    'classification': {
        'logistic': {'C': [0.1, 1.0]},
        'knn': {'n_neighbors': [3, 5]},
        'rf': {'n_estimators': [50, 100]}
    },
    'regression': {
        'lr': {},
        'rf': {'n_estimators': [50, 100]},
        'svr': {'C': [0.1, 1.0], 'kernel': ['linear', 'rbf']}
    }
}

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')



@app.route('/')
def index():
    return render_template('index.html', active_page='index')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Les mots de passe ne correspondent pas.")
            return redirect(url_for('signup'))

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Nom d'utilisateur ou email déjà utilisé.")
            return redirect(url_for('signup'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Compte créé avec succès.")
        return redirect(url_for('login'))

    return render_template('auth/signup.html', active_page='signup')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form['username']
        password = request.form['password']

        user = User.query.filter((User.username == username_or_email) | (User.email == username_or_email)).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash("Connexion réussie.")
            return redirect(url_for('index'))
        else:
            flash("Identifiants incorrects.")
            return redirect(url_for('login'))

    return render_template('auth/login.html', active_page='login')


@app.route('/logout')
def logout():
    session.clear()
    flash("Déconnexion réussie.")
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    selected_task = request.args.get('task', 'classification')

    try:
        if request.method == 'POST':
            # === Nettoyage des données ===
            if 'clean_data' in request.form:
                filepath = session.get('dataset_path')
                filename = session.get('filename', '')
                if not filepath or not os.path.exists(filepath):
                    flash("Fichier non trouvé pour nettoyage.", "error")
                    return redirect(request.url)

                df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
                df.dropna(inplace=True)

                preview = df.head(10).to_dict('records')
                stats = {
                    'rows': df.shape[0],
                    'columns': list(df.columns),
                    'missing': df.isnull().sum().to_dict(),
                    'dtypes': df.dtypes.astype(str).to_dict()
                }

                flash("✅ Données nettoyées avec succès !", "success")
                return render_template('upload.html',
                                       preview=preview,
                                       stats=stats,
                                       columns=list(df.columns),
                                       filename=filename,
                                       model_options=MODEL_OPTIONS[selected_task],
                                       all_model_options=MODEL_OPTIONS,
                                       selected_task=selected_task,
                                       active_page='upload')

            # === Téléversement du fichier ===
            file = request.files.get('dataset')
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)

                session['dataset_path'] = filepath
                session['filename'] = filename
            else:
                filepath = session.get('dataset_path')
                if not filepath or not os.path.exists(filepath):
                    flash("Aucun fichier existant trouvé. Merci de téléverser un fichier.", "error")
                    return redirect(request.url)

                filename = session.get('filename', '')
                df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)

            # === Aperçu et stats ===
            preview = df.head(10).to_dict('records')
            stats = {
                'rows': df.shape[0],
                'columns': list(df.columns),
                'missing': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }

            if 'analyze' not in request.form:
                return render_template('upload.html',
                                       preview=preview,
                                       stats=stats,
                                       columns=list(df.columns),
                                       filename=filename,
                                       model_options=MODEL_OPTIONS[selected_task],
                                       all_model_options=MODEL_OPTIONS,
                                       selected_task=selected_task,
                                       active_page='upload')

            # === Analyse ML ===
            target = request.form.get('target')
            model_type = request.form.get('model')
            task = request.form.get('task')

            if not target or not model_type or not task:
                flash("Tous les champs doivent être remplis pour l'analyse.", "error")
                return redirect(request.url)

            if target not in df.columns:
                flash(f"Colonne cible '{target}' introuvable.", "error")
                return redirect(request.url)

            df.dropna(inplace=True)

            if task == 'classification' and df[target].dtype == 'object':
                le = LabelEncoder()
                df[target] = le.fit_transform(df[target])

            X = pd.get_dummies(df.drop(columns=[target]))
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type not in MODEL_OPTIONS[task]:
                flash("Modèle non reconnu.", "error")
                return redirect(request.url)

            model_name, model = MODEL_OPTIONS[task][model_type]
            param_grid = PARAM_GRIDS[task].get(model_type, {})
            best_params = None

            try:
                if param_grid:
                    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy' if task == 'classification' else 'r2')
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    best_params = grid.best_params_
                else:
                    model.fit(X_train, y_train)

            except ValueError as ve:
                if "Unknown label type: continuous" in str(ve):
                    flash("❌ Erreur : vous avez choisi un modèle de **classification**, mais la variable cible contient des valeurs continues. Veuillez utiliser un modèle de **régression**.", "error")
                else:
                    flash(f"Erreur de configuration du modèle : {str(ve)}", "error")
                print("Traceback :", traceback.format_exc())
                return redirect(request.url)

            y_pred = model.predict(X_test)

            # === Résultats + Visualisations ===
            visualizations = {}
            results = {}

            if task == 'classification':
                results['accuracy'] = accuracy_score(y_test, y_pred)
                results['report'] = classification_report(y_test, y_pred, output_dict=True)

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                visualizations['confusion_matrix'] = fig_to_base64(fig)

                if len(set(y_test)) == 2 and hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.2f}")
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.legend()
                    visualizations['roc_curve'] = fig_to_base64(fig)
            else:
                results['mse'] = mean_squared_error(y_test, y_pred)
                results['r2'] = r2_score(y_test, y_pred)

                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                visualizations['regression_plot'] = fig_to_base64(fig)

            if hasattr(model, 'feature_importances_'):
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(X.columns, model.feature_importances_)
                visualizations['feature_importance'] = fig_to_base64(fig)

            analysis = Analysis(
                user_id=session['user_id'],
                filename=filename,
                model_type=model_name,
                task_type=task,
                target_column=target,
                accuracy=results.get('accuracy'),
                mse=results.get('mse'),
                r2=results.get('r2'),
                confusion_matrix=visualizations.get('confusion_matrix'),
                roc_curve=visualizations.get('roc_curve'),
                regression_plot=visualizations.get('regression_plot'),
                feature_importance=visualizations.get('feature_importance'),
                preview_data=json.dumps(preview)
            )
            db.session.add(analysis)
            db.session.commit()

            return render_template('results.html',
                                   task=task,
                                   model_name=model_name,
                                   results=results,
                                   visualizations=visualizations,
                                   preview=preview,
                                   best_params=best_params,
                                   analysis=analysis)

    except Exception as e:
        print("Erreur interne :", traceback.format_exc())
        flash("Une erreur interne est survenue : " + str(e), "error")
        return redirect(request.url)

    # GET
    return render_template('upload.html',
                           model_options=MODEL_OPTIONS[selected_task],
                           all_model_options=MODEL_OPTIONS,
                           selected_task=selected_task,
                           active_page='upload')


# Les autres routes statiques
@app.route('/contact')
def contact():
    return render_template('contact.html', active_page='contact')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/services')
def services():
    return render_template('services.html', active_page='services')

@app.route('/projects')
def projects():
    return render_template('projects.html', active_page='projects')

@app.route('/features')
def features():
    return render_template('features.html', active_page='features')

@app.route('/team')
def team():
    return render_template('team.html', active_page='team')

@app.route('/faq')
def faq():
    return render_template('faq.html', active_page='faq')

@app.route('/testimonials')
def testimonials():
    return render_template('testimonials.html', active_page='testimonials')

@app.route('/404')
def not_found():
    return render_template('404.html', active_page='not_found')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    filter_type = request.args.get('filter', 'all')
    search_term = request.args.get('search', '').lower()

    query = Analysis.query.filter_by(user_id=session['user_id'])

    if filter_type in ['classification', 'regression']:
        query = query.filter(Analysis.task_type == filter_type)

    if search_term:
        query = query.filter(Analysis.filename.ilike(f'%{search_term}%'))

    analyses = query.order_by(Analysis.created_at.desc()).all()

    return render_template('service.html', analyses=analyses, active_page='history')



@app.route('/analysis/<int:analysis_id>')
def view_analysis(analysis_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != session['user_id']:
        flash("Vous n'avez pas accès à cette analyse.")
        return redirect(url_for('history'))

    return render_template("view_analysis.html", analysis=analysis)




@app.route('/users')
def list_users():
    users = User.query.all()  # Récupère tous les utilisateurs
    return render_template('users.html', users=users)

@app.route('/compare_models', methods=['POST'])
def compare_models():
    task = request.form.get('task')
    filename = request.form.get('filename')
    target = request.form.get('target')

    if not filename or not task or not target:
        flash("Champs manquants pour la comparaison.", "error")
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        flash("Fichier introuvable pour comparaison.", "error")
        return redirect(url_for('upload'))

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
    except Exception as e:
        flash(f"Erreur lors de la lecture du fichier : {str(e)}", "error")
        return redirect(url_for('upload'))

    df.dropna(inplace=True)

    if target not in df.columns:
        flash(f"Colonne cible '{target}' introuvable.", "error")
        return redirect(url_for('upload'))

    if task == 'classification' and df[target].dtype == 'object':
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])

    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_scores = []

    for model_key, (name, model) in MODEL_OPTIONS_3[task].items():
        try:
            param_grid = PARAM_GRIDS[task].get(model_key, {})

            if param_grid:
                grid = GridSearchCV(model, param_grid, cv=3,
                                    scoring='accuracy' if task == 'classification' else 'r2')
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = accuracy_score(y_test, y_pred) if task == 'classification' else r2_score(y_test, y_pred)

            model_scores.append({'model': name, 'score': round(score, 4)})
        except Exception as e:
            model_scores.append({'model': name, 'score': None, 'error': str(e)})

    model_scores = sorted(model_scores, key=lambda x: x['score'] or 0, reverse=True)

    return render_template('compare.html',
                           task=task,
                           target=target,
                           filename=filename,
                           model_scores=model_scores)

import numpy as np
import pandas as pd
import json
from flask import jsonify, session
import requests
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    """Encodeur personnalisé pour gérer les types NumPy"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

GEMINI_CONFIG = {
    "api_key": "AIzaSyCHUlYvvCWYXCGhJ7ZHbAb7Pye4CV0YM4U",
    "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    "timeout": 30,
    "fallback_strategy": "last_reasonable_column"
}

def convert_numpy_types(data):
    """Convertit récursivement les types NumPy en types natifs Python"""
    if isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    return data

def get_fallback_target(df):
    """Détermine une colonne cible de repli intelligente"""
    for col in df.columns[::-1]:
        nunique = df[col].nunique()
        if 1 < nunique < len(df) and nunique < 100:
            return col
    return df.columns[-1]

def call_gemini_api_safe(prompt):
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_CONFIG["api_key"]}
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "safetySettings": [{
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH"
        }],
        "generationConfig": {
            "temperature": 0.3,
            "topP": 0.8,
            "maxOutputTokens": 200
        }
    }

    try:
        response = requests.post(
            GEMINI_CONFIG["endpoint"],
            headers=headers,
            params=params,
            json=payload,
            timeout=GEMINI_CONFIG["timeout"]
        )
        
        if response.status_code != 200:
            app.logger.error(f"Erreur API ({response.status_code}): {response.text[:200]}")
            return None

        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
    except Exception as e:
        app.logger.error(f"Erreur API: {str(e)}")
        return None

@app.route('/suggest_target', methods=['GET'])
def suggest_target():
    filepath = session.get('dataset_path')
    if not filepath or not os.path.exists(filepath):
        return jsonify({
            "success": False,
            "message": "Fichier non trouvé en session",
            "type": "file_error"
        })

    try:
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        df = df.dropna()
        
        if len(df.columns) == 0:
            return jsonify({
                "success": False,
                "message": "Aucune colonne valide dans le fichier",
                "type": "data_error"
            })

        # Préparation des statistiques avec conversion des types NumPy
        stats = {
            col: {
                "type": str(df[col].dtype),
                "unique": int(df[col].nunique()),
                "missing": int(df[col].isna().sum()),
                "sample": [
                    int(x) if isinstance(x, (np.integer, np.int64)) else 
                    float(x) if isinstance(x, np.floating) else 
                    str(x) 
                    for x in df[col].dropna().unique()[:3]
                ]
            }
            for col in df.columns
        }

        prompt = f"""
        [Rôle] Expert en analyse de données
        [Tâche] Identifier la meilleure colonne cible pour un modèle prédictif
        [Format] Répondre uniquement avec le nom exact de la colonne
        
        [Colonnes Disponibles]
        {json.dumps(stats, indent=2, cls=NumpyEncoder)}
        
        [Critères]
        1. Colonne avec des valeurs non aléatoires
        2. Distribution non uniforme
        3. Pas d'identifiants uniques
        4. Pertinence métier potentielle
        """

        suggestion = call_gemini_api_safe(prompt)
        valid_suggestion = None
        source = ""

        if suggestion and suggestion in df.columns:
            valid_suggestion = suggestion
            source = "gemini_api"
        else:
            if suggestion:
                app.logger.warning(f"Suggestion invalide: {suggestion}")
            valid_suggestion = get_fallback_target(df)
            source = "fallback_algorithm"

        # Préparation de la réponse avec conversion des types
        response_data = {
            "success": True,
            "suggestion": valid_suggestion,
            "source": source,
            "all_columns": list(df.columns),
            "timestamp": datetime.now().isoformat(),
            "stats": convert_numpy_types(stats)
        }

        # Retour avec sérialisation JSON sécurisée
        return app.response_class(
            response=json.dumps(response_data, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        app.logger.exception("Erreur critique dans suggest_target")
        return jsonify({
            "success": False,
            "message": str(e),
            "type": "critical_error"
        })
    # Vérification du fichier
    filepath = session.get('dataset_path')
    if not filepath or not os.path.exists(filepath):
        return jsonify({
            "success": False,
            "message": "Fichier non trouvé en session",
            "type": "file_error"
        })
    

    try:
        # Lecture des données
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        df = df.dropna()
        
        if len(df.columns) == 0:
            return jsonify({
                "success": False,
                "message": "Aucune colonne valide dans le fichier",
                "type": "data_error"
            })

        # Préparation du prompt optimisé
        prompt = f"""
        [Rôle] Expert en analyse de données
        [Tâche] Identifier la meilleure colonne cible pour un modèle prédictif
        [Format] Répondre uniquement avec le nom exact de la colonne
        
        [Colonnes Disponibles]
        {json.dumps({col: str(df[col].dtype) for col in df.columns}, indent=2)}
        
        [Statistiques Clés]
        {json.dumps({col: {
            "unique": df[col].nunique(),
            "missing": df[col].isna().sum(),
            "sample": list(df[col].dropna().unique()[:3])
        } for col in df.columns}, indent=2)}
        
        [Critères]
        1. Colonne avec des valeurs non aléatoires
        2. Distribution non uniforme
        3. Pas d'identifiants uniques
        4. Pertinence métier potentielle
        """

        # Tentative d'appel API
        suggestion = call_gemini_api_safe(prompt)
        
        # Validation de la suggestion
        valid_suggestion = None
        if suggestion and suggestion in df.columns:
            valid_suggestion = suggestion
            source = "gemini_api"
        else:
            if suggestion:
                app.logger.warning(f"Suggestion invalide: {suggestion}")
            
            # Stratégie de repli
            valid_suggestion = get_fallback_target(df)
            source = "fallback_algorithm"

        return jsonify({
            "success": True,
            "suggestion": valid_suggestion,
            "source": source,
            "all_columns": list(df.columns),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        app.logger.exception("Erreur critique dans suggest_target")
        return jsonify({
            "success": False,
            "message": str(e),
            "type": "critical_error"
        })

    filepath = session.get('dataset_path')
    if not filepath or not os.path.exists(filepath):
        return jsonify({"success": False, "message": "Fichier non trouvé"})

    try:
        # Lecture des données
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        df = df.dropna()
        
        # Préparation du prompt
        sample_data = df.head(3).to_dict(orient='records')
        column_info = {
            col: {
                "type": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "sample_values": df[col].dropna().unique().tolist()[:3]
            }
            for col in df.columns
        }

        prompt = f"""
        En tant qu'expert en machine learning, analyse ce jeu de données et suggère la colonne la plus 
        appropriée comme variable cible pour une modélisation. Réponds uniquement avec le nom de la colonne.

        Colonnes disponibles:
        {json.dumps(column_info, indent=2, ensure_ascii=False)}

        Exemple de données:
        {json.dumps(sample_data, indent=2, ensure_ascii=False)}
        """

        # Appel à l'API Gemini
        suggestion = call_gemini_api(prompt)
        
        if not suggestion:
            return jsonify({
                "success": False,
                "message": "Échec de la suggestion via API",
                "fallback": df.columns[-1]  # Colonne de repli
            })

        # Validation que la colonne suggérée existe
        if suggestion.strip() in df.columns:
            return jsonify({
                "success": True,
                "suggestion": suggestion.strip()
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Colonne suggérée invalide: {suggestion}",
                "fallback": df.columns[-1]
            })

    except Exception as e:
        app.logger.error(f"Erreur: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Erreur interne: {str(e)}"
        }) 
    filepath = session.get('dataset_path')
    filename = session.get('filename', '')
    if not filepath or not os.path.exists(filepath):
        return jsonify(success=False, message="Fichier non trouvé.")

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        df.dropna(inplace=True)

        sample_data = df.head(5).to_dict(orient='records')
        column_info = {
            col: {
                "dtype": str(df[col].dtype),
                "nunique": df[col].nunique(),
                "sample": df[col].dropna().unique().tolist()[:5]
            }
            for col in df.columns
        }

        prompt = f"""
Analyze this dataset and suggest the most appropriate target column for machine learning.
Respond with ONLY the column name, nothing else.

Column information:
{json.dumps(column_info, indent=2)}

Data sample:
{json.dumps(sample_data, indent=2)}
"""

        # Updated Gemini API call
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.5,
                "topP": 1,
                "maxOutputTokens": 256
            }
        }

        response = requests.post(gemini_url, headers=headers, json=payload)
        response.raise_for_status()
        
        try:
            suggestion = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Basic validation
            if suggestion not in df.columns:
                return jsonify(success=False, message="Invalid column suggestion from API")
                
            return jsonify(success=True, suggestion=suggestion)
            
        except (KeyError, IndexError) as e:
            return jsonify(success=False, message="Unexpected API response format")

    except requests.exceptions.RequestException as e:
        return jsonify(success=False, message=f"API request failed: {str(e)}")
    except Exception as e:
        return jsonify(success=False, message=f"An error occurred: {str(e)}")
if __name__ == '__main__':
    app.run(debug=True)