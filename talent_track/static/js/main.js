// static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
    console.log('TalentTrack initialized');
});

// Function to load and display the Geography visualization.
// For example, this code initializes a basic Leaflet map.
// Make sure to include the Leaflet library (CSS and JS) in your geography page.
function loadGeography() {
    console.log("Loading geography visualization...");

    // Check if Leaflet (L) is available.
    if (typeof L !== 'undefined') {
        // Create a map in the 'geography-viz' container.
        var map = L.map('geography-viz').setView([51.505, -0.09], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }).addTo(map);
    } else {
        console.warn("Leaflet is not loaded. Please include the Leaflet library in your geography page.");
        // As a fallback, you could display a message.
        document.getElementById('geography-viz').innerHTML = "<p>Map cannot be loaded because Leaflet is missing.</p>";
    }
}

// Function to load and display the Pipeline visualization.
// In this example, we simply put a placeholder message.
// Replace this with actual AJAX calls or visualization code as needed.
function loadPipeline() {
    console.log("Loading pipeline visualization...");
    var pipelineContainer = document.getElementById('pipeline-viz');
    if (pipelineContainer) {
        pipelineContainer.innerHTML = "<p>Pipeline visualization coming soon!</p>";
    }
}
