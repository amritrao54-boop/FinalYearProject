import React, { useState } from 'react';
import axios from 'axios';

const AVAILABLE_SYMPTOMS = [
  { id: 'fever', label: 'Fever / High Temperature' },
  { id: 'diarrhoea', label: 'Diarrhoea / Loose Motion' },
  { id: 'coughing', label: 'Coughing' },
  { id: 'loss_of_appetite', label: 'Loss of Appetite / Not Eating' },
  { id: 'lameness', label: 'Lameness / Limping' },
  { id: 'swelling', label: 'Swelling / Udder Inflammation' },
  { id: 'salivation', label: 'Salivation / Excessive Drooling' },
  { id: 'depression', label: 'Depression / Lethargy' },
  { id: 'weight_loss', label: 'Weight Loss' }
];

function App() {
  const [animalId, setAnimalId] = useState('COW-101');
  const [query, setQuery] = useState('Loose motion , fever');
  const [selectedSymptoms, setSelectedSymptoms] = useState(['diarrhoea', 'fever']);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trace, setTrace] = useState([]);

  const toggleSymptom = (id) => {
    if (selectedSymptoms.includes(id)) {
      setSelectedSymptoms(selectedSymptoms.filter(s => s !== id));
    } else {
      setSelectedSymptoms([...selectedSymptoms, id]);
    }
  };

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleDiagnose = async () => {
    setLoading(true);
    setTrace([
      '📥 Received user request and input data...',
      '🔍 Analyzing symptoms and extracting medical entities...',
      '🤖 Invoking Machine Learning classification models...',
      '📚 Searching Veterinary Knowledge Base (RAG)...',
      '⚖️ Multimodal Fusion & Generating Grounded Diagnostic Assessment...'
    ]);

    const formData = new FormData();
    formData.append('user_query', query);
    formData.append('animal_id', animalId);
    
    selectedSymptoms.forEach(sym => {
      formData.append('symptoms', sym);
    });

    if (imageFile) {
      formData.append('image', imageFile);
    }

    try {
      const response = await axios.post('http://localhost:8000/api/agent/diagnose', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (error) {
      console.error(error);
      setResult({
        assessment: {
          predicted_condition: 'Connection Error',
          confidence: '0%'
        },
        warning: 'Could not connect to FastAPI Agent backend on port 8000. Please ensure the backend is running.',
        evidence: ['API connection failed.'],
        recommendations: ['Check if backend is started with: python -m uvicorn backend.app.main:app --port 8000']
      });
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', padding: '2rem', fontFamily: 'Segoe UI, Roboto, Helvetica, Arial, sans-serif', color: '#1a202c' }}>
      
      {/* Header */}
      <header style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '1rem', marginBottom: '2rem' }}>
        <h1 style={{ margin: 0, color: '#2b6cb0', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span>🐄</span> Multimodal Agentic AI Cattle Health Diagnosis
        </h1>
        <p style={{ margin: '0.5rem 0 0 0', color: '#718096' }}>
          AI-Powered Diagnostic Orchestrator with Machine Learning, CNN Computer Vision, RAG Knowledge Retrieval & Explainable AI.
        </p>
      </header>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        
        {/* Input Panel */}
        <div style={{ background: '#f7fafc', padding: '1.5rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
          <h2 style={{ fontSize: '1.25rem', marginTop: 0, color: '#2d3748' }}>📝 Patient & Symptom Input</h2>
          
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.25rem' }}>Animal Identifier / Tag ID:</label>
            <input 
              type="text" 
              value={animalId} 
              onChange={(e) => setAnimalId(e.target.value)}
              style={{ width: '100%', padding: '0.5rem', borderRadius: '6px', border: '1px solid #cbd5e0' }}
            />
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.25rem' }}>Observed Clinical Symptoms:</label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', maxHeight: '180px', overflowY: 'auto', background: '#fff', padding: '0.75rem', borderRadius: '6px', border: '1px solid #cbd5e0' }}>
              {AVAILABLE_SYMPTOMS.map((sym) => (
                <label key={sym.id} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.875rem', cursor: 'pointer' }}>
                  <input 
                    type="checkbox" 
                    checked={selectedSymptoms.includes(sym.id)} 
                    onChange={() => toggleSymptom(sym.id)}
                  />
                  {sym.label}
                </label>
              ))}
            </div>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.25rem' }}>Additional Notes / Farmer Query:</label>
            <textarea 
              rows="3" 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. Cow has high fever, skin lesions, and refuses feed since yesterday..."
              style={{ width: '100%', padding: '0.5rem', borderRadius: '6px', border: '1px solid #cbd5e0', boxSizing: 'border-box' }}
            />
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.25rem' }}>Upload Cattle Lesion / Udder Image:</label>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageChange} 
              style={{ width: '100%', fontSize: '0.875rem' }}
            />
            {imagePreview && (
              <div style={{ marginTop: '0.75rem' }}>
                <img src={imagePreview} alt="Preview" style={{ width: '100%', maxHeight: '150px', objectFit: 'cover', borderRadius: '6px' }} />
              </div>
            )}
          </div>

          <button 
            onClick={handleDiagnose} 
            disabled={loading}
            style={{ 
              width: '100%', 
              padding: '0.75rem', 
              background: loading ? '#a0aec0' : '#3182ce', 
              color: '#fff', 
              fontWeight: 'bold', 
              border: 'none', 
              borderRadius: '8px', 
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '1rem'
            }}
          >
            {loading ? '⚙️ Agent Reasoning in Progress...' : '🚀 Run Agent Diagnostic Analysis'}
          </button>
        </div>

        {/* Results & Trace Panel */}
        <div>
          {loading && (
            <div style={{ background: '#ebf8ff', border: '1px solid #bee3f8', padding: '1.5rem', borderRadius: '12px', marginBottom: '1rem' }}>
              <h3 style={{ marginTop: 0, color: '#2b6cb0' }}>🤖 Agent Execution Trace</h3>
              <ul style={{ margin: 0, paddingLeft: '1.25rem', color: '#2c5282' }}>
                {trace.map((step, idx) => (
                  <li key={idx} style={{ marginBottom: '0.5rem' }}>{step}</li>
                ))}
              </ul>
            </div>
          )}

          {result && (
            <div style={{ background: '#fff', padding: '1.5rem', borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #edf2f7', paddingBottom: '0.75rem' }}>
                <h2 style={{ margin: 0, fontSize: '1.3rem', color: '#2d3748' }}>Diagnostic Assessment</h2>
                <span style={{ background: '#c6f6d5', color: '#22543d', padding: '0.25rem 0.75rem', borderRadius: '9999px', fontSize: '0.85rem', fontWeight: 'bold' }}>
                  {result.assessment?.certainty_level || 'Preliminary'}
                </span>
              </div>

              <div style={{ marginTop: '1rem' }}>
                <p style={{ fontSize: '1.15rem', margin: '0.5rem 0' }}>
                  <strong>Most Likely Condition:</strong>{' '}
                  <span style={{ color: '#c53030', fontWeight: 'bold' }}>
                    {result.assessment?.predicted_condition?.replace(/_/g, ' ').toUpperCase()}
                  </span>
                </p>
                <p style={{ margin: '0.5rem 0', color: '#4a5568' }}>
                  <strong>Confidence:</strong> {result.assessment?.confidence}
                </p>
              </div>

              {/* Supporting Evidence */}
              {result.evidence && result.evidence.length > 0 && (
                <div style={{ marginTop: '1rem' }}>
                  <h4 style={{ margin: '0.5rem 0', color: '#2d3748' }}>📋 Supporting Clinical Evidence:</h4>
                  <ul style={{ margin: 0, paddingLeft: '1.25rem', color: '#4a5568', fontSize: '0.9rem' }}>
                    {result.evidence.map((ev, i) => (
                      <li key={i}>{ev}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Model Breakdown */}
              {result.model_results?.symptom_model?.probabilities && (
                <div style={{ marginTop: '1rem', background: '#f7fafc', padding: '0.75rem', borderRadius: '6px' }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem', color: '#4a5568' }}>📊 Ensemble Model Breakdown:</h4>
                  <div style={{ fontSize: '0.85rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    {Object.entries(result.model_results.symptom_model.probabilities).map(([modelName, pred]) => (
                      <span key={modelName} style={{ background: '#edf2f7', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                        <strong>{modelName}:</strong> {pred}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* RAG Veterinary Sources */}
              {result.knowledge_sources && result.knowledge_sources.length > 0 && (
                <div style={{ marginTop: '1rem' }}>
                  <h4 style={{ margin: '0.5rem 0', color: '#2d3748' }}>📖 Veterinary Knowledge Base (RAG):</h4>
                  <div style={{ background: '#edf2f7', padding: '0.75rem', borderRadius: '6px', fontSize: '0.85rem', color: '#2d3748' }}>
                    {result.knowledge_sources.map((src, i) => (
                      <p key={i} style={{ margin: '0.25rem 0' }}>{src}</p>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommended Next Steps */}
              {result.recommendations && (
                <div style={{ marginTop: '1rem' }}>
                  <h4 style={{ margin: '0.5rem 0', color: '#2d3748' }}>💡 Recommended Action Plan:</h4>
                  <ul style={{ margin: 0, paddingLeft: '1.25rem', color: '#2f855a', fontSize: '0.9rem' }}>
                    {result.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Warning Alert */}
              <div style={{ marginTop: '1.25rem', background: '#fffaf0', borderLeft: '4px solid #dd6b20', padding: '0.75rem', borderRadius: '4px' }}>
                <p style={{ margin: 0, fontSize: '0.8rem', color: '#7b341e' }}>
                  ⚠️ <strong>Disclaimer:</strong> {result.warning}
                </p>
              </div>

            </div>
          )}

          {!loading && !result && (
            <div style={{ border: '2px dashed #cbd5e0', borderRadius: '12px', padding: '3rem', textAlign: 'center', color: '#a0aec0' }}>
              <span style={{ fontSize: '3rem' }}>🔬</span>
              <p style={{ marginTop: '1rem' }}>Select symptoms or upload a photo on the left, then click <strong>Run Agent Diagnostic Analysis</strong> to view the results.</p>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default App;
