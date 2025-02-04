// static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
    console.log('TalentTrack initialized');
});

function loadGeography() {
    console.log("Loading geography visualization...");
    
    if (!document.getElementById('geography-viz')) {
        console.error('Geography container not found!');
        return;
    }

    let map;
    try {
        const container = document.getElementById('geography-viz');
        container.innerHTML = '';
        
        map = L.map('geography-viz', {
            minZoom: 2,
            maxZoom: 18,
            worldCopyJump: true
        }).setView([20, 0], 2);
        
        console.log('Map initialized successfully');

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        fetch('/api/geography_data')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error('Error loading geography data:', data.error);
                    return;
                }
                
                map.eachLayer((layer) => {
                    if (layer instanceof L.HeatLayer) {
                        map.removeLayer(layer);
                    }
                });
                
                const heatData = data.locations.map(loc => {
                    const coords = loc.coordinates;
                    const intensity = (loc.candidate_count / data.max_count + 
                                    loc.avg_quality / data.max_quality) / 2;
                    return [coords.lat, coords.lng, intensity];
                });
                
                const heat = L.heatLayer(heatData, {
                    radius: 35,
                    blur: 20,
                    maxZoom: 10,
                    max: 1.0,
                    minOpacity: 0.4,
                    gradient: {
                        0.2: '#0000ff',
                        0.5: '#00ff00',
                        0.8: '#ff0000'
                    }
                }).addTo(map);
                
                const existingLegends = document.querySelectorAll('.legend');
                existingLegends.forEach(el => el.remove());
                
                const legend = L.control({ position: 'bottomright' });
                legend.onAdd = function(map) {
                    const div = L.DomUtil.create('div', 'legend');
                    div.innerHTML = `
                        <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 5px rgba(0,0,0,0.4);">
                            <h4 style="margin: 0 0 8px 0; font-weight: bold;">Talent Density</h4>
                            <div class="legend-item" style="margin: 5px 0;">
                                <span style="display: inline-block; width: 20px; height: 20px; background: #ff0000; margin-right: 8px;"></span>
                                High concentration
                            </div>
                            <div class="legend-item" style="margin: 5px 0;">
                                <span style="display: inline-block; width: 20px; height: 20px; background: #00ff00; margin-right: 8px;"></span>
                                Medium concentration
                            </div>
                            <div class="legend-item" style="margin: 5px 0;">
                                <span style="display: inline-block; width: 20px; height: 20px; background: #0000ff; margin-right: 8px;"></span>
                                Low concentration
                            </div>
                        </div>
                    `;
                    return div;
                };
                legend.addTo(map);
                
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
    } catch (error) {
        console.error('Error initializing map:', error);
        document.getElementById('geography-viz').innerHTML = 
            '<p class="error">Error initializing map</p>';
    }
}

// Function to load and display the Pipeline visualization
function loadPipeline() {
    console.log("Loading pipeline visualization...");
    var pipelineContainer = document.getElementById('pipeline-viz');
    if (pipelineContainer) {
        pipelineContainer.innerHTML = "<p>Pipeline visualization coming soon!</p>";
    }
}