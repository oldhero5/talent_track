function loadMonitoringData() {
    // Load drift detection
    fetch('/api/drift_detection')
        .then(response => response.json())
        .then(data => {
            document.getElementById('drift-viz').src = 'data:image/png;base64,' + data.plot;
            
            // Display metrics
            const metricsDiv = document.getElementById('drift-metrics');
            metricsDiv.innerHTML = '<h3>Drift Metrics</h3>';
            Object.entries(data.metrics).forEach(([feature, metrics]) => {
                metricsDiv.innerHTML += `
                    <div class="metric-group">
                        <h4>${feature}</h4>
                        <p>Mean drift: ${metrics.mean_drift.toFixed(4)}</p>
                        <p>Std drift: ${metrics.std_drift.toFixed(4)}</p>
                        <p class="${metrics.requires_attention ? 'alert' : ''}">
                            ${metrics.requires_attention ? 'Requires Attention' : 'Stable'}
                        </p>
                    </div>
                `;
            });
        });

    // Load confusion matrix
    fetch('/api/confusion_matrix')
        .then(response => response.text())
        .then(data => {
            document.getElementById('confusion-matrix-viz').src = 'data:image/png;base64,' + data;
        });

    // Load ROC curve
    fetch('/api/roc_curve')
        .then(response => response.text())
        .then(data => {
            document.getElementById('roc-curve-viz').src = 'data:image/png;base64,' + data;
        });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', loadMonitoringData);