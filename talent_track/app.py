# talent_track/talent_track/app.py
import io
import base64
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, jsonify, request
from sklearn.metrics import roc_curve, auc, confusion_matrix

from talent_track.functions import (
    DataGenerator, PQModel, FeedbackTracker, ModelExplainer, 
    RecruitmentVisualizer, DriftDetector
)

# Global data store class
class DataStore:
    def __init__(self):
        self.data_generator = DataGenerator()
        self.pq_model = PQModel(n_bits=8)
        self.feedback_tracker = FeedbackTracker()
        self.model_explainer = None
        self.visualizer = None
        self.current_employees = None
        self.leads = None

def create_app(test_config=None):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    app = Flask(__name__,
                template_folder=os.path.join(base_dir, 'templates'),
                static_folder=os.path.join(base_dir, 'static'))
    
    app.data_store = DataStore()
    register_routes(app)
    
    with app.app_context():
        initialize_data(app)
    
    return app

def initialize_data(app):
    """Initialize all data and models"""
    try:
        print("Initializing data...")
        
        # Generate data
        current_employees = app.data_store.data_generator.generate_dataset(1000)
        leads = app.data_store.data_generator.generate_dataset(500)
        
        # Assign statuses
        current_employees['status'] = 'current_employee'
        leads['status'] = np.random.choice(
            ['screening', 'interview', 'offer_extended', 'rejected'],
            size=len(leads)
        )
        
        combined_data = pd.concat([current_employees, leads])
        app.data_store.pq_model.fit(combined_data)
        
        app.data_store.visualizer = RecruitmentVisualizer(
            app.data_store.pq_model, 
            app.data_store.feedback_tracker
        )
        app.data_store.model_explainer = ModelExplainer(app.data_store.pq_model)
        
        app.data_store.current_employees = current_employees
        app.data_store.leads = leads
        
        print("Initialization complete!")
        return True
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        return False

def _sanitize_records(df):
    """Convert numpy types to native Python types for JSON serialization."""
    records = df.to_dict('records')
    for record in records:
        for key, value in record.items():
            if isinstance(value, np.integer):
                record[key] = int(value)
            elif isinstance(value, np.floating):
                record[key] = float(value)
            elif isinstance(value, np.ndarray):
                record[key] = value.tolist()
    return records

def register_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html',
                               current_employees=app.data_store.current_employees,
                               leads=app.data_store.leads)

    @app.route('/talent_pool')
    def talent_pool():
        return render_template('talent_pool.html')

    @app.route('/api/talent_pool')
    def get_talent_pool():
        try:
            # Handle empty query parameters by checking for empty strings
            min_exp_param = request.args.get('min_exp', '')
            max_exp_param = request.args.get('max_exp', '')
            min_exp = float(min_exp_param) if min_exp_param.strip() != '' else 0.0
            max_exp = float(max_exp_param) if max_exp_param.strip() != '' else float('inf')
            
            filtered_leads = app.data_store.leads.copy()
            status = request.args.get('status', 'all')
            if status != 'all':
                filtered_leads = filtered_leads[filtered_leads['status'] == status]
            filtered_leads = filtered_leads[
                (filtered_leads['months_experience'] >= min_exp) &
                (filtered_leads['months_experience'] <= max_exp)
            ]
            
            fig, tsne_data = app.data_store.visualizer.create_talent_pool_visualization(
                app.data_store.current_employees, 
                filtered_leads,
                return_data=True
            )
            
            output = io.BytesIO()
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig).print_png(output)
            return jsonify({
                'image': base64.b64encode(output.getvalue()).decode('utf-8'),
                'data': tsne_data
            })
        except Exception as e:
            print(f"Error in get_talent_pool: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/model_analysis')
    def model_analysis():
        return render_template('model_analysis.html')

    @app.route('/api/candidates')
    def get_candidates():
        try:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 10))
            
            employees = app.data_store.current_employees.copy()
            employees['type'] = 'employee'
            leads = app.data_store.leads.copy()
            leads['type'] = 'lead'
            all_candidates = pd.concat([employees, leads])
            
            all_candidates['pq_score'] = app.data_store.pq_model.predict(all_candidates)
            subset = all_candidates.iloc[(page-1)*per_page:page*per_page]
            data = _sanitize_records(subset)
            
            return jsonify({
                'data': data,
                'recordsTotal': len(all_candidates),
                'recordsFiltered': len(all_candidates)
            })
        except Exception as e:
            print(f"Error in get_candidates: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/candidate_ranking')
    def candidate_ranking():
        try:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 10))
            
            employees = app.data_store.current_employees.copy()
            employees['type'] = 'employee'
            leads = app.data_store.leads.copy()
            leads['type'] = 'lead'
            all_candidates = pd.concat([employees, leads])
            
            all_candidates['pq_score'] = app.data_store.pq_model.predict(all_candidates)
            ranked_candidates = all_candidates.sort_values(by='pq_score', ascending=False)
            subset = ranked_candidates.iloc[(page-1)*per_page:page*per_page]
            data = _sanitize_records(subset)
            
            return jsonify({
                'data': data,
                'recordsTotal': len(ranked_candidates),
                'recordsFiltered': len(ranked_candidates)
            })
        except Exception as e:
            print(f"Error in candidate_ranking: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/model_shap')
    def get_model_shap():
        try:
            explanation = app.data_store.model_explainer.explain_model(
                pd.concat([app.data_store.current_employees, app.data_store.leads])
            )
            if explanation is None:
                return jsonify({'error': 'Failed to generate SHAP explanation'}), 500
                
            fig = app.data_store.model_explainer.plot_shap_summary(explanation)
            output = io.BytesIO()
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig).print_png(output)
            plt.close(fig)
            return base64.b64encode(output.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error in get_model_shap: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/candidate_shap/<candidate_id>')
    def get_candidate_shap(candidate_id):
        try:
            # Combine data from both groups
            employees = app.data_store.current_employees.copy()
            employees['type'] = 'employee'
            leads = app.data_store.leads.copy()
            leads['type'] = 'lead'
            combined = pd.concat([employees, leads])
            candidate = combined[combined['id'] == candidate_id]
            if candidate.empty:
                return jsonify({'error': 'Candidate not found'}), 404
            
            # Compute candidate-specific SHAP explanation (using the same explainer)
            explanation = app.data_store.model_explainer.explain_model(candidate)
            fig = app.data_store.model_explainer.plot_shap_summary(explanation)
            output = io.BytesIO()
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig).print_png(output)
            plt.close(fig)
            return base64.b64encode(output.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error in get_candidate_shap: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/model_monitoring')
    def model_monitoring():
        return render_template('model_monitoring.html')

    @app.route('/api/drift_detection')
    def get_drift_detection():
        try:
            drift_detector = DriftDetector(app.data_store.current_employees)
            drift_metrics = drift_detector.check_drift(app.data_store.leads)
            for feature in drift_metrics:
                drift_metrics[feature]['requires_attention'] = int(
                    drift_metrics[feature]['requires_attention']
                )
            
            fig = drift_detector.plot_drift_metrics(drift_metrics)
            output = io.BytesIO()
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig).print_png(output)
            plt.close(fig)
            return jsonify({
                'plot': base64.b64encode(output.getvalue()).decode('utf-8'),
                'metrics': drift_metrics
            })
        except Exception as e:
            print(f"Error in get_drift_detection: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/confusion_matrix')
    def get_confusion_matrix():
        try:
            y_true = app.data_store.leads['recommendation']
            scores = -app.data_store.pq_model.predict(app.data_store.leads)
            y_pred = (scores > scores.mean()).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            output = io.BytesIO()
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig).print_png(output)
            plt.close(fig)
            return base64.b64encode(output.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error in get_confusion_matrix: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/roc_curve')
    def get_roc_curve():
        try:
            y_true = app.data_store.leads['recommendation']
            scores = -app.data_store.pq_model.predict(app.data_store.leads)
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend()
            
            output = io.BytesIO()
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig).print_png(output)
            plt.close(fig)
            return base64.b64encode(output.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error in get_roc_curve: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/geography')
    def geography():
        return render_template('geography.html')

    # New endpoints for user input (feedback, add candidate)
    @app.route('/api/feedback', methods=['POST'])
    def post_feedback():
        try:
            data = request.get_json()
            candidate_id = data.get('candidate_id')
            stage = data.get('stage')
            status = data.get('status')
            notes = data.get('notes')
            if not candidate_id or not stage or not status:
                return jsonify({'error': 'Missing required fields'}), 400
            feedback = app.data_store.feedback_tracker.add_feedback(candidate_id, stage, status, notes)
            return jsonify({'message': 'Feedback received', 'feedback': feedback}), 200
        except Exception as e:
            print(f"Error in post_feedback: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/add_candidate', methods=['POST'])
    def add_candidate():
        try:
            candidate = request.get_json()
            if not candidate or 'id' not in candidate:
                return jsonify({'error': 'Candidate data with id is required'}), 400
            candidate_type = candidate.get('type', 'lead')
            if candidate_type == 'employee':
                app.data_store.current_employees = app.data_store.current_employees.append(candidate, ignore_index=True)
            else:
                app.data_store.leads = app.data_store.leads.append(candidate, ignore_index=True)
            
            combined = pd.concat([app.data_store.current_employees, app.data_store.leads])
            app.data_store.pq_model.fit(combined)
            return jsonify({'message': 'Candidate added and model updated'}), 200
        except Exception as e:
            print(f"Error in add_candidate: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # Add route for pipeline page (was missing)
    @app.route('/pipeline')
    def pipeline():
        return render_template('pipeline.html')

def main():
    app = create_app()
    print("\n=== Starting TalentTrack Server ===")
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    app.run(debug=True)

if __name__ == '__main__':
    main()
