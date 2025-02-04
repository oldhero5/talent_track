function loadGeography() {
    console.log("Loading geography visualization...");
    
    // Initialize the map
    const map = L.map('geography-viz').setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    // Fetch geography data
    fetch('/api/geography_data')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error loading geography data:', data.error);
                return;
            }
            
            // Prepare heatmap data
            const heatData = data.locations.map(loc => {
                const coords = loc.coordinates;
                // Normalize the intensity based on both count and quality
                const intensity = (loc.candidate_count / data.max_count + 
                                 loc.avg_quality / data.max_quality) / 2;
                return [coords.lat, coords.lng, intensity];
            });
            
            // Create and add the heatmap layer
            const heat = L.heatLayer(heatData, {
                radius: 25,
                blur: 15,
                maxZoom: 10,
                max: 1.0,
                gradient: {
                    0.4: 'blue',
                    0.65: 'lime',
                    1: 'red'
                }
            }).addTo(map);
            
            // Add a legend
            const legend = L.control({ position: 'bottomright' });
            legend.onAdd = function(map) {
                const div = L.DomUtil.create('div', 'legend');
                div.innerHTML = `
                    <h4>Candidate Density</h4>
                    <div class="legend-item">
                        <span class="legend-color" style="background: red"></span>High
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background: lime"></span>Medium
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background: blue"></span>Low
                    </div>
                `;
                return div;
            };
            legend.addTo(map);
            
            // Fit the map to the data points
            if (heatData.length > 0) {
                const bounds = L.latLngBounds(heatData.map(point => [point[0], point[1]]));
                map.fitBounds(bounds);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('geography-viz').innerHTML = 
                '<p class="error">Error loading geography data</p>';
        });
}