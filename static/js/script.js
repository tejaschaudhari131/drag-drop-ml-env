let preprocessingSteps = [];
let selectedModel = '';

function addPreprocessingStep(step) {
    preprocessingSteps.push(step);
    document.getElementById('steps').innerHTML += `<p>${step}</p>`;
}

function selectModel(model) {
    selectedModel = model;
    document.getElementById('steps').innerHTML += `<p>Model: ${model}</p>`;
}

function trainModel() {
    const dataset = { /* dataset from user */ };
    fetch('/train_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset, model_type: selectedModel })
    })
    .then(response => response.json())
    .then(data => alert(`Model trained! Accuracy: ${data.accuracy}`));
}

function exportModel() {
    const format = 'pickle';  // Example export format
    fetch('/export_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: selectedModel, format })
    })
    .then(response => response.json())
    .then(data => alert(`Model exported to: ${data.model_path}`));
}
