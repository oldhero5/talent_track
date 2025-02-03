// talent_track/static/js/talent_pool.js
function updateVisualization() {
    const status = document.getElementById('status').value;
    const minExp = document.getElementById('min_exp').value;
    const maxExp = document.getElementById('max_exp').value;
    
    fetch(`/api/talent_pool?status=${status}&min_exp=${minExp}&max_exp=${maxExp}`)
        .then(response => response.json())
        .then(data => {
            const scatter = {
                mode: 'markers',
                type: 'scatter',
                x: data.data.x,
                y: data.data.y,
                text: data.data.hover_text,
                marker: {
                    size: 10,
                    color: data.data.colors
                },
                hoverinfo: 'text'
            };

            const layout = {
                title: 'Talent Pool',
                showlegend: false,
                hovermode: 'closest',
                xaxis: {
                    title: 'TSNE-1',
                    zeroline: false
                },
                yaxis: {
                    title: 'TSNE-2',
                    zeroline: false
                }
            };

            Plotly.newPlot('talent-pool-viz', [scatter], layout);

            // Add click handler for detailed view
            document.getElementById('talent-pool-viz').on('plotly_click', (data) => {
                const candidateInfo = data.points[0];
                showCandidateDetails(candidateInfo.customdata);
            });
        });
}

function showCandidateDetails(candidateId) {
    fetch(`/api/candidate_details/${candidateId}`)
        .then(response => response.json())
        .then(data => {
            const detailsContent = document.getElementById('details-content');
            detailsContent.innerHTML = `
                <p><strong>Name:</strong> ${data.name}</p>
                <p><strong>Experience:</strong> ${data.months_experience} months</p>
                <p><strong>Education:</strong> ${data.education_years} years</p>
                <p><strong>PQ Score:</strong> ${data.pq_score.toFixed(2)}</p>
                <p><strong>Status:</strong> ${data.status}</p>
            `;
        });
}

// Initialize visualization on page load
document.addEventListener('DOMContentLoaded', updateVisualization);