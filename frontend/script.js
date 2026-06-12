function analyzeLogs() {
    const fileInput = document.getElementById('logFileInput');
    if (fileInput.files.length === 0) {
        alert("Please select a log file to analyze.");
        return;
    }

    // Show loading state
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('resultsTableBody').innerHTML = ''; // Clear previous results

    // Simulate analysis delay
    setTimeout(() => {
        document.getElementById('loading').classList.add('hidden');
        displayMockResults();
    }, 2000);
}

function displayMockResults() {
    // Reveal the results section
    document.getElementById('resultsSection').classList.remove('hidden');

    // Generate mock data representing an analysis of the uploaded log file
    const totalLines = 500;
    const mockAnomalies = [
        { line: 42, content: "ERROR: Connection timeout connecting to database 10.0.0.5", score: "15.4", prediction: "Anomaly" },
        { line: 128, content: "WARN: Unrecognized token 'admin' attempted login", score: "12.8", prediction: "Anomaly" },
        { line: 310, content: "FATAL: OutOfMemoryException in Thread-4", score: "22.1", prediction: "Anomaly" },
        { line: 499, content: "INFO: Failed to load module 'secure_auth.so'", score: "9.5", prediction: "Anomaly" }
    ];

    document.getElementById('totalLogs').textContent = totalLines;
    document.getElementById('anomaliesDetected').textContent = mockAnomalies.length;

    const tbody = document.getElementById('resultsTableBody');

    mockAnomalies.forEach(anomaly => {
        const row = document.createElement('tr');
        row.className = 'anomaly-row';

        row.innerHTML = `
            <td>${anomaly.line}</td>
            <td><code>${anomaly.content}</code></td>
            <td>${anomaly.score}</td>
            <td>${anomaly.prediction}</td>
        `;

        tbody.appendChild(row);
    });
}