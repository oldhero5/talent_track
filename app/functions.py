import numpy as np
import pandas as pd
from faker import Faker
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
import faiss
import shap
from datetime import datetime
import folium
from folium.plugins import HeatMap
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
        
        # Handle categorical features
        categorical_cols = ['university_institution', 'location']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder().fit(df[col].unique())
            df_processed[col] = self.label_encoders[col].transform(df[col])
        
        # Handle list features
        list_cols = ['certifications', 'courses']
        for col in list_cols:
            df_processed[f'{col}_count'] = df[col].apply(len)
        
        # Select numerical features for PQ
        numerical_features = ['months_experience', 'education_years', 'awards', 
                            'certifications_count', 'courses_count',
                            'university_institution', 'location']
        
        if not hasattr(self, 'feature_names'):
            self.feature_names = numerical_features
            
        return self.scaler.fit_transform(df_processed[numerical_features]) if not hasattr(self, 'is_fitted') else self.scaler.transform(df_processed[numerical_features])
    
    def fit(self, df):
        X = self._preprocess_features(df)
        d = X.shape[1]
        
        # Determine number of subquantizers that divides the dimension evenly
        for i in range(1, d+1):
            if d % i == 0:
                self.n_subquantizers = i
                break
                
        print(f"Using {self.n_subquantizers} subquantizers for dimension {d}")
        
        self.is_fitted = True
        self.index = faiss.IndexPQ(d, self.n_subquantizers, self.n_bits, faiss.METRIC_L2)
        self.index.train(X.astype(np.float32))
        self.index.add(X.astype(np.float32))
        
        return self
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            distances, _ = self.index.search(X.astype(np.float32), k=1)
            return -distances
        else:
            X_processed = self._preprocess_features(X)
            distances, _ = self.index.search(X_processed.astype(np.float32), k=1)
            return -distances

class ModelExplainer:
    def __init__(self, pq_model):
        self.pq_model = pq_model
        
    def explain_model(self, df):
        try:
            X = self.pq_model._preprocess_features(df)
            background_data = X[:100]
            
            explainer = shap.KernelExplainer(
                lambda x: self.pq_model.predict(x),
                background_data,
                n_samples=50
            )
            
            sample_size = min(100, X.shape[0])
            shap_values = explainer.shap_values(X[:sample_size])
            
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
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values
            }
        except Exception as e:
            print(f"Error in explain_model: {str(e)}")
            return None

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
        
    def _get_coordinates(self, location):
        max_retries = 3
        for i in range(max_retries):
            try:
                time.sleep(1)
                loc = self.geolocator.geocode(location)
                if loc:
                    return loc.latitude, loc.longitude
                return None
            except GeocoderTimedOut:
                if i == max_retries - 1:
                    return None
                continue
            
    def create_talent_pool_visualization(self, current_employees_df, leads_df):
        current_employees_df['status'] = 'current_employee'
        combined_df = pd.concat([current_employees_df, leads_df])
        
        X = self.pq_model._preprocess_features(combined_df)
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        
        mask_current = combined_df['status'] == 'current_employee'
        plt.scatter(X_tsne[mask_current, 0], X_tsne[mask_current, 1], 
                   c=self.status_colors['current_employee'], 
                   label='Current Employees', alpha=0.6, s=100)
        
        for status in ['offer_extended', 'interview', 'screening', 'rejected']:
            mask = combined_df['status'] == status
            if mask.any():
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                          c=self.status_colors[status],
                          label=status.replace('_', ' ').title(),
                          alpha=0.6, s=100)
        
        plt.title('Talent Pool Visualization')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_geographical_heatmap(self, employees_df):
        quality_scores = self.pq_model.predict(employees_df)
        employees_df['quality_score'] = -quality_scores
        
        location_data = []
        unique_locations = employees_df['location'].unique()
        
        for location in unique_locations:
            coords = self._get_coordinates(location)
            if coords:
                location_score = employees_df[
                    employees_df['location'] == location
                ]['quality_score'].mean()
                
                location_data.append({
                    'location': location,
                    'lat': coords[0],
                    'lon': coords[1],
                    'score': float(location_score),
                    'count': int(employees_df[
                        employees_df['location'] == location
                    ].shape[0])
                })
        
        location_df = pd.DataFrame(location_data)
        center_lat = location_df['lat'].mean()
        center_lon = location_df['lon'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=3)
        
        heat_data = [[row['lat'], row['lon'], row['score'] * row['count']] 
                    for _, row in location_df.iterrows()]
        HeatMap(heat_data).add_to(m)
        
        for _, row in location_df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=10,
                popup=f"""
                <b>{row['location']}</b><br>
                Employees: {row['count']}<br>
                Avg Quality: {row['score']:.2f}
                """,
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(m)
        
        return m
    
    def create_recruitment_funnel(self):
        stages = ['screening', 'interview', 'offer_extended', 'hired']
        stage_counts = {}
        
        for stage in stages:
            stage_counts[stage] = len([
                f for f in self.feedback_tracker.feedback_data 
                if f['stage'] == stage
            ])
        
        plt.figure(figsize=(10, 6))
        y_pos = range(len(stages))
        counts = [stage_counts[stage] for stage in stages]
        
        plt.barh(y_pos, counts)
        plt.yticks(y_pos, [s.replace('_', ' ').title() for s in stages])
        plt.xlabel('Number of Candidates')
        plt.title('Recruitment Funnel')
        
        for i in range(len(stages)-1):
            if counts[i] > 0:
                conversion = (counts[i+1] / counts[i]) * 100
                plt.text(counts[i], i, f'{conversion:.1f}%', 
                        va='center', ha='left', color='red')
        
        plt.tight_layout()
        return plt.gcf()