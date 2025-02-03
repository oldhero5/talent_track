let candidatesTable;

function initializeCandidatesTable() {
    candidatesTable = $('#candidates-table').DataTable({
        serverSide: true,
        ajax: {
            url: '/api/candidates',
            data: function(d) {
                return {
                    page: (d.start / d.length) + 1,
                    per_page: d.length
                };
            }
        },
        columns: [
            { data: 'name' },
            { data: 'type' },
            { data: 'months_experience' },
            { 
                data: 'pq_score',
                render: function(data) {
                    return data.toFixed(2);
                }
            },
            { data: 'status' },
            {
                data: 'id',
                render: function(data) {
                    return `<button onclick="showCandidateShap('${data}')">View SHAP</button>`;
                }
            }
        ],
        pageLength: 10
    });
}

function loadModelShap() {
    fetch('/api/model_shap')
        .then(response => response.text())
        .then(data => {
            document.getElementById('model-shap-viz').src = 'data:image/png;base64,' + data;
        });
}

function showCandidateShap(candidateId) {
    fetch(`/api/candidate_shap/${candidateId}`)
        .then(response => response.text())
        .then(data => {
            const content = document.getElementById('candidate-shap-content');
            content.innerHTML = `<img src="data:image/png;base64,${data}" alt="Candidate SHAP Values">`;
        });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeCandidatesTable();
    loadModelShap();
});
