# talent_track/talent_track/db.py
from pymongo import MongoClient
import os
import numpy as np
import pickle
from datetime import datetime

class MongoDB:
    def __init__(self):
        self.client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        self.db = self.client.talent_track

    def save_model(self, model_name, model_data, metadata=None):
        """Save a model and its metadata to MongoDB"""
        collection = self.db.models
        
        # Serialize the model
        model_binary = pickle.dumps(model_data)
        
        # Prepare the document
        document = {
            'name': model_name,
            'model_data': model_binary,
            'metadata': metadata or {},
            'created_at': datetime.utcnow()
        }
        
        # Update or insert the model
        collection.update_one(
            {'name': model_name},
            {'$set': document},
            upsert=True
        )

    def load_model(self, model_name):
        """Load a model from MongoDB"""
        collection = self.db.models
        document = collection.find_one({'name': model_name})
        
        if document:
            return pickle.loads(document['model_data'])
        return None

    def save_candidates(self, candidates_df):
        """Save candidates data to MongoDB"""
        collection = self.db.candidates
        
        # Convert DataFrame to list of dictionaries
        records = candidates_df.to_dict('records')
        
        # Convert numpy types to Python native types
        for record in records:
            for key, value in record.items():
                if isinstance(value, np.integer):
                    record[key] = int(value)
                elif isinstance(value, np.floating):
                    record[key] = float(value)
                elif isinstance(value, np.ndarray):
                    record[key] = value.tolist()

        # Clear existing records and insert new ones
        collection.delete_many({})
        collection.insert_many(records)

    def load_candidates(self):
        """Load candidates data from MongoDB"""
        collection = self.db.candidates
        return list(collection.find({}, {'_id': 0}))

    def save_feedback(self, feedback_data):
        """Save feedback data to MongoDB"""
        collection = self.db.feedback
        collection.insert_one(feedback_data)

    def get_feedback_history(self, candidate_id):
        """Get feedback history for a candidate"""
        collection = self.db.feedback
        return list(collection.find({'candidate_id': candidate_id}, {'_id': 0}))

    def save_model_metrics(self, metrics_data):
        """Save model monitoring metrics"""
        collection = self.db.model_metrics
        metrics_data['timestamp'] = datetime.utcnow()
        collection.insert_one(metrics_data)

    def get_model_metrics_history(self, start_date=None, end_date=None):
        """Get model metrics history within a date range"""
        collection = self.db.model_metrics
        query = {}
        if start_date or end_date:
            query['timestamp'] = {}
            if start_date:
                query['timestamp']['$gte'] = start_date
            if end_date:
                query['timestamp']['$lte'] = end_date
        return list(collection.find(query, {'_id': 0}))