# talent_track/talent_track/functions.py
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
import faiss
import shap
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

class DataGenerator:
    def __init__(self):
        self.fake = Faker()
        self.universities = ['MIT', 'Stanford', 'Harvard', 'Berkeley', 'Oxford']
        self.certifications = ['AWS', 'GCP', 'Azure', 'PMP', 'CISSP']
        self.courses = ['Machine Learning', 'Data Science', 'Cloud Computing']
        self.locations = ['New York', 'San Francisco', 'London', 'Berlin']
        
    def generate_sample(self):
        return {
            'id': self.fake.uuid4(),
            'name': self.fake.name(),
            'months_experience': np.random.randint(0, 240),
            'education_years': np.random.randint(12, 22),
            'university_institution': np.random.choice(self.universities),
            'certifications': np.random.choice(self.certifications, 
                                             size=np.random.randint(0, 4)).tolist(),
            'awards': np.random.randint(0, 5),
            'courses': np.random.choice(self.courses, 
                                      size=np.random.randint(0, 3)).tolist(),
            'location': np.random.choice(self.locations),
            'recommendation': np.random.choice([0, 1], p=[0.7, 0.3])
        }
    
    def generate_dataset(self, n_samples=10000):
        return pd.DataFrame([self.generate_sample() for _ in range(n_samples)])

class PQModel:
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def _preprocess_features(self, df):
        df_processed = df.copy()
        categorical_cols = ['university_institution', 'location']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder().fit(df[col].unique())
            df_processed[col] = self.label_encoders[col].transform(df[col])
        
        list_cols = ['certifications', 'courses']
        for col in list_cols:
            df_processed[f'{col}_count'] = df[col].apply(len)
        
        numerical_features = ['months_experience', 'education_years', 'awards', 
                                'certifications_count', 'courses_count',
                                'university_institution', 'location']
        if not hasattr(self, 'feature_names'):
            self.feature_names = numerical_features
            
        if not hasattr(self, 'is_fitted'):
            return self.scaler.fit_transform(df_processed[numerical_features])
        else:
            return self.scaler.transform(df_processed[numerical_features])
    
    def fit(self, df):
        X = self._preprocess_features(df)
        d = X.shape[1]
        for i in range(1, d+1):
            if d % i == 0:
                self.n_subquantizers = i
                break
        self.is_fitted = True
        self.index = faiss.IndexPQ(d, self.n_subquantizers, self.n_bits, faiss.METRIC_L2)
        self.index.train(X.astype(np.float32))
        self.index.add(X.astype(np.float32))
        return self
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            distances, _ = self.index.search(X.astype(np.float32), k=1)
            # Flatten the 2D output to a 1D array.
            return -distances.flatten()
        else:
            X_processed = self._preprocess_features(X)
            distances, _ = self.index.search(X_processed.astype(np.float32), k=1)
            return -distances.flatten()

# class ModelExplainer:
#     def __init__(self, pq_model):
#         self.pq_model = pq_model
        
#     def explain_model(self, df):
#         try:
#             X = self.pq_model._preprocess_features(df)
#             background_data = X[:100]
            
#             explainer = shap.KernelExplainer(
#                 lambda x: self.pq_model.predict(x),
#                 background_data,
#                 n_samples=50
#             )
            
#             sample_size = min(100, X.shape[0])
#             shap_values = explainer.shap_values(X[:sample_size])
            
#             if isinstance(shap_values, list):
#                 shap_values = shap_values[0]
            
#             feature_importance = {
#                 str(feature): float(importance)
#                 for feature, importance in zip(
#                     self.pq_model.feature_names,
#                     np.abs(shap_values).mean(0)
#                 )
#             }
            
#             return {
#                 'feature_importance': feature_importance,
#                 'shap_values': shap_values
#             }
#         except Exception as e:
#             print(f"Error in explain_model: {str(e)}")
#             return None
            
#     def plot_shap_summary(self, explanation):
#         if explanation is None or not explanation['shap_values']:
#             # Create empty plot if no data
#             plt.figure(figsize=(10, 6))
#             plt.text(0.5, 0.5, 'No SHAP values available', 
#                     horizontalalignment='center',
#                     verticalalignment='center')
#             return plt.gcf()
        
#         plt.figure(figsize=(10, 6))
#         try:
#             shap.summary_plot(
#                 explanation['shap_values'],
#                 pd.DataFrame(columns=self.pq_model.feature_names),
#                 show=False
#             )
#         except Exception as e:
#             plt.text(0.5, 0.5, f'Error generating SHAP plot: {str(e)}', 
#                     horizontalalignment='center',
#                     verticalalignment='center')
#         return plt.gcf()

class ModelExplainer:
    def __init__(self, pq_model):
        self.pq_model = pq_model
        
    def explain_model(self, df):
        try:
            # Preprocess the input data
            X = self.pq_model._preprocess_features(df)
            background_data = X[:100]
            explainer = shap.KernelExplainer(
                lambda x: self.pq_model.predict(x),
                background_data,
                n_samples=50
            )
            sample_size = min(100, X.shape[0])
            X_sample = X[:sample_size]
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            feature_importance = {
                str(feature): float(importance)
                for feature, importance in zip(
                    self.pq_model.feature_names,
                    np.abs(shap_values).mean(0)
                )
            }
            return {
                'feature_importance': feature_importance,
                'shap_values': shap_values,
                'X': X_sample
            }
        except Exception as e:
            print(f"Error in explain_model: {str(e)}")
            return None
            
    def plot_shap_summary(self, explanation):
        shap_values = explanation.get('shap_values', None)
        X_sample = explanation.get('X', None)
        if explanation is None or shap_values is None or (hasattr(shap_values, 'size') and shap_values.size == 0):
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No SHAP values available', 
                     horizontalalignment='center',
                     verticalalignment='center')
            return plt.gcf()
        
        # Convert X_sample to a DataFrame if it isn't one already.
        if not isinstance(X_sample, pd.DataFrame):
            X_sample = pd.DataFrame(X_sample, columns=self.pq_model.feature_names)
        
        plt.figure(figsize=(10, 6))
        try:
            # Generate the SHAP summary plot.
            shap.summary_plot(shap_values, X_sample, feature_names=self.pq_model.feature_names, show=False)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error generating SHAP plot: {str(e)}', 
                     horizontalalignment='center',
                     verticalalignment='center')
        return plt.gcf()


class FeedbackTracker:
    def __init__(self):
        self.feedback_data = []
        
    def add_feedback(self, candidate_id, stage, status, notes=None):
        feedback = {
            'candidate_id': candidate_id,
            'stage': stage,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'notes': notes
        }
        self.feedback_data.append(feedback)
        return feedback
    
    def get_candidate_history(self, candidate_id):
        return [f for f in self.feedback_data if f['candidate_id'] == candidate_id]
    
    def get_conversion_rates(self):
        stages = ['initial_screening', 'interview', 'offer', 'hired']
        conversion_rates = {}
        
        for i in range(len(stages)-1):
            current_stage = stages[i]
            next_stage = stages[i+1]
            
            current_count = len([f for f in self.feedback_data if f['stage'] == current_stage])
            next_count = len([f for f in self.feedback_data if f['stage'] == next_stage])
            
            if current_count > 0:
                conversion_rates[f'{current_stage}_to_{next_stage}'] = next_count / current_count
                
        return conversion_rates

class RecruitmentVisualizer:
    def __init__(self, pq_model, feedback_tracker):
        self.pq_model = pq_model
        self.feedback_tracker = feedback_tracker
        self.geolocator = Nominatim(user_agent="recruitment_visualizer")
        
        self.status_colors = {
            'current_employee': '#2ecc71',
            'offer_extended': '#f1c40f',
            'interview': '#e67e22',
            'screening': '#3498db',
            'rejected': '#e74c3c'
        }
        
    def create_talent_pool_visualization(self, current_employees_df, leads_df, return_data=False):
        current_employees_df['status'] = 'current_employee'
        combined_df = pd.concat([current_employees_df, leads_df])
        
        X = self.pq_model._preprocess_features(combined_df)
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        
        colors = [self.status_colors[status] for status in combined_df['status']]
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=100)
        
        plt.title('Talent Pool Visualization')
        plt.xlabel('TSNE-1')
        plt.ylabel('TSNE-2')
        
        if return_data:
            hover_text = [
                f"Name: {row['name']}\n"
                f"Experience: {row['months_experience']} months\n"
                f"Status: {row['status']}"
                for _, row in combined_df.iterrows()
            ]
            
            return plt.gcf(), {
                'x': X_tsne[:, 0].tolist(),
                'y': X_tsne[:, 1].tolist(),
                'colors': colors,
                'hover_text': hover_text
            }
        
        return plt.gcf()

class DriftDetector:
    def __init__(self, initial_data):
        self.baseline_distribution = self._calculate_distribution(initial_data)
        self.threshold = 0.1
        
    def _calculate_distribution(self, data):
        numerical_cols = ['months_experience', 'education_years', 'awards']
        return {
            col: {
                'mean': data[col].mean(), 
                'std': data[col].std()
            } for col in numerical_cols
        }
    
    def check_drift(self, new_data):
        current_distribution = self._calculate_distribution(new_data)
        drift_metrics = {}
        
        for feature in self.baseline_distribution.keys():
            baseline = self.baseline_distribution[feature]
            current = current_distribution[feature]
            
            mean_diff = abs(baseline['mean'] - current['mean']) / baseline['mean']
            std_diff = abs(baseline['std'] - current['std']) / baseline['std']
            
            drift_metrics[feature] = {
                'mean_drift': mean_diff,
                'std_drift': std_diff,
                'requires_attention': mean_diff > self.threshold or std_diff > self.threshold
            }
            
        return drift_metrics
    
    def plot_drift_metrics(self, drift_metrics):
        plt.figure(figsize=(12, 6))
        features = list(drift_metrics.keys())
        mean_drifts = [m['mean_drift'] for m in drift_metrics.values()]
        std_drifts = [m['std_drift'] for m in drift_metrics.values()]
        
        x = np.arange(len(features))
        width = 0.35
        
        plt.bar(x - width/2, mean_drifts, width, label='Mean Drift')
        plt.bar(x + width/2, std_drifts, width, label='Std Drift')
        
        plt.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Features')
        plt.ylabel('Drift Magnitude')
        plt.title('Feature Drift Analysis')
        plt.xticks(x, features)
        plt.legend()
        
        return plt.gcf()