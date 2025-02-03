# TalentTrack

TalentTrack is a recruitment analysis tool that uses Product Quantization (PQ) for candidate similarity matching and provides visualizations for talent pool analysis, model monitoring, and recruitment pipeline tracking.

## Features

- Talent Pool Visualization with TSNE clustering
- Candidate similarity scoring using Product Quantization
- Model explanability with SHAP values
- Geographic distribution analysis
- Model drift detection
- Recruitment pipeline monitoring
- Interactive data tables and filtering

## Installation

1. Clone the repository:
```bash
git clone https://github.com/oldhero5/talent_track.git
cd talent_track
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Project Structure
```
talent_track/
├── talent_track/
│   ├── __init__.py
│   ├── app.py
│   ├── functions.py
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       ├── talent_pool.js
│   │       ├── model_analysis.js
│   │       └── model_monitoring.js
│   └── templates/
│       ├── index.html
│       ├── talent_pool.html
│       ├── model_analysis.html
│       └── model_monitoring.html
├── pyproject.toml
└── requirements.txt
```

## Running the Application

1. Make sure you're in the project root directory and your virtual environment is activated

2. Run the Flask application:
```bash
python -m talent_track.app
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Features in Detail

### Talent Pool Visualization
- Interactive scatter plot showing employee and candidate clustering
- Filter by status and experience level
- Hover for detailed information
- Click for candidate details

### Model Analysis
- Candidate similarity scoring
- SHAP value explanations
- Feature importance visualization
- Individual candidate analysis

### Model Monitoring
- Data drift detection
- Confusion matrix
- ROC curve analysis
- Performance metrics tracking

## Development

### Prerequisites
- Python 3.8+
- uv package manager
- Node.js (for DataTables)

### Setup Development Environment
```bash
# Clone repository
git clone https://github.com/yourusername/talent_track.git
cd talent_track

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Running Tests
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- FAISS library by Facebook Research
- SHAP (SHapley Additive exPlanations)
- Flask web framework
- DataTables for interactive tables
- Plotly for interactive visualizations

## Troubleshooting

### Common Issues

1. Template Not Found
```bash
# Make sure your template folder path is correct in app.py
app = Flask(__name__,
            template_folder='/path/to/your/templates',
            static_folder='static')
```

2. Data Initialization Errors
```bash
# Check the console output for initialization errors
# Make sure all dependencies are installed
uv pip install -r requirements.txt
```

3. Visualization Errors
```bash
# If visualizations aren't showing:
# - Check browser console for JavaScript errors
# - Verify that static files are being served correctly
# - Check that matplotlib backend is set to 'Agg'
```

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.