import './App.css';
import { Editor } from 'ketcher-react';
import { StandaloneStructServiceProvider } from 'ketcher-standalone';
import "ketcher-react/dist/index.css";
import { useEffect, useState, useRef, useCallback } from 'react';

const structServiceProvider = new StandaloneStructServiceProvider();
const apiBase = import.meta.env.VITE_API_URL || '';

function App() {
  const editorRef = useRef(null);
  const [molecules, setMolecules] = useState([]);
  const [solvents, setSolvents] = useState([]);
  const [selectedMolId, setSelectedMolId] = useState(null);
  const [selectedSolvent, setSelectedSolvent] = useState('');  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [currentMolecule, setCurrentMolecule] = useState(null);
  const [moleculeUpdated, setMoleculeUpdated] = useState(false);
  const [moleculeImage, setMoleculeImage] = useState(null);
  const [isDropping, setIsDropping] = useState(false);
  const [showSplash, setShowSplash] = useState(false);
  const [moleculeVanished, setMoleculeVanished] = useState(false);  const [moleculeReappearing, setMoleculeReappearing] = useState(false);
  const [isCheckingMolecule, setIsCheckingMolecule] = useState(false);
  const [showResultModal, setShowResultModal] = useState(false);
  const [includeUncertainty, setIncludeUncertainty] = useState(true);
  const [includeExplanations, setIncludeExplanations] = useState(false);
  const [useEnhancedApi, setUseEnhancedApi] = useState(true);
  const [includeMolt5, setIncludeMolt5] = useState(true); // Enable by default
  useEffect(() => {
    fetch(`${apiBase}/molecules`)
      .then(res => res.json())
      .then(setMolecules)
      .catch(() => {});
    fetch(`${apiBase}/solvents`)
      .then(res => res.json())
      .then(setSolvents)
      .catch(() => {});
  }, []);  // Function to update molecule preview
  const updateMoleculePreview = useCallback(async () => {
    try {
      if (editorRef.current) {
        const smiles = await editorRef.current.getSmiles();
        const molfile = await editorRef.current.getMolfile();
        const newMolecule = { smiles, molfile };

        // Generate molecule image if there's a structure
        if (smiles && smiles.trim() && smiles !== '') {
          try {
            const imageBlob = await editorRef.current.generateImage(molfile, {
              outputFormat: 'svg'
            });
            
            if (imageBlob) {
              const imageUrl = URL.createObjectURL(imageBlob);
              setMoleculeImage(imageUrl);
            }          } catch {
            // Fallback: try PNG format
            try {
              const pngBlob = await editorRef.current.generateImage(molfile, {
                outputFormat: 'png'
              });
              
              if (pngBlob) {
                const imageUrl = URL.createObjectURL(pngBlob);
                setMoleculeImage(imageUrl);
              } else {
                setMoleculeImage(null);
              }            } catch {
              setMoleculeImage(null);
            }
          }
        } else {
          setMoleculeImage(null);
        }

        // Check if molecule actually changed
        const moleculeChanged = currentMolecule && currentMolecule.smiles !== smiles;
        const isSignificantChange = moleculeChanged && smiles && smiles.trim() !== '';
        
        if (isSignificantChange && !isCheckingMolecule) {
          // Only trigger reappearing if molecule was vanished
          if (moleculeVanished) {
            setMoleculeVanished(false);
            setMoleculeReappearing(true);
            setTimeout(() => setMoleculeReappearing(false), 800);
          } else if (!moleculeVanished) {
            // Show update animation if not vanished
            setMoleculeUpdated(true);
            setTimeout(() => setMoleculeUpdated(false), 500);
          }
        }

        setCurrentMolecule(newMolecule);
      }
    } catch (error) {
      console.error('Error updating molecule preview:', error);
    }
  }, [currentMolecule, moleculeVanished, isCheckingMolecule]);

  // Handle pre-existing molecule selection
  useEffect(() => {
    if (selectedMolId && editorRef.current) {
      // When a pre-existing molecule is selected, trigger a preview update
      setTimeout(() => {
        updateMoleculePreview();
      }, 500);
    }
  }, [selectedMolId, updateMoleculePreview]);

  // Handle solvent changes - trigger reappearing if molecule was vanished
  useEffect(() => {
    if (moleculeVanished && selectedSolvent) {
      setMoleculeVanished(false);
      setMoleculeReappearing(true);
      setTimeout(() => setMoleculeReappearing(false), 800);
    }
  }, [selectedSolvent, moleculeVanished]);

  // Handle molecule selection changes - trigger reappearing if molecule was vanished
  useEffect(() => {
    if (moleculeVanished && selectedMolId) {
      setMoleculeVanished(false);
      setMoleculeReappearing(true);
      setTimeout(() => setMoleculeReappearing(false), 800);
    }
  }, [selectedMolId, moleculeVanished]);

  // Close modal on Escape key
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && showResultModal) {
        setShowResultModal(false);
      }
    };
    
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [showResultModal]);

  const handleCheck = async () => {
    setResult(null);
    setError('');
    setLoading(true);
    setIsCheckingMolecule(true); // Flag that we're checking
    
    // Start drop animation if we have a molecule and solvent
    if ((currentMolecule?.smiles || selectedMolId) && selectedSolvent) {
      setIsDropping(true);
      
      // Wait for drop animation to complete before showing splash
      setTimeout(() => {
        setShowSplash(true);
        // Hide splash and set molecule as vanished
        setTimeout(() => {
          setShowSplash(false);
          setIsDropping(false);
          setMoleculeVanished(true);
        }, 1000);
      }, 1500); // Drop animation duration
    }
      try {
      let endpoint, payload;
      
      if (useEnhancedApi) {
        // Use the new enhanced prediction API
        endpoint = `${apiBase}/predict/enhanced`;
        
        if (selectedMolId) {
          // For existing molecules, we need to get the SMILES first
          const selectedMol = molecules.find(m => m.id === selectedMolId);
          payload = {
            smiles: selectedMol?.smiles || '',
            include_uncertainty: includeUncertainty,
            include_explanations: includeExplanations,
            include_molt5: includeMolt5  // Add this line
          };
        } else {
          // For drawn molecules, extract SMILES from editor
          const smiles = await editorRef.current?.getSmiles();
          payload = {
            smiles: smiles || '',
            include_uncertainty: includeUncertainty,
            include_explanations: includeExplanations,
            include_molt5: includeMolt5  // Add this line
          };
        }
      } else {
        // Use the old solubility API (backward compatibility)
        endpoint = `${apiBase}/solubility`;
        payload = { solvent: selectedSolvent };
        
        if (selectedMolId) {
          payload.molecule_id = selectedMolId;
        } else {
          const smiles = await editorRef.current?.getSmiles();
          payload.smiles = smiles;
        }
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Error');
      setResult(data);
      setShowResultModal(true); // Show the popup modal
    } catch (e) {
      setError(e.message);
      // Reset animation states on error
      setIsDropping(false);
      setShowSplash(false);
      setMoleculeVanished(false);
    } finally {
      setLoading(false);
      setIsCheckingMolecule(false); // Clear the flag
    }
  };
  return (
    <div className="app-container">
      {/* Left Panel - Molecule Editor */}
      <div className="left-panel">
        <h1 className="panel-title">Molecule Designer</h1>        <div className="controls-section">
          <div className="control-group">
            <label>Select existing molecule or draw new:</label>
            <select value={selectedMolId || ''} onChange={e => setSelectedMolId(e.target.value || null)}>
              <option value="">-- Draw New --</option>
              {molecules.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </div>

          <div className="control-group">
            <label>Prediction Options:</label>
            <div className="checkbox-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={useEnhancedApi}
                  onChange={e => setUseEnhancedApi(e.target.checked)}
                />
                Use Enhanced MEGAN Predictions
              </label>
              
              {useEnhancedApi && (
                <>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={includeUncertainty}
                      onChange={e => setIncludeUncertainty(e.target.checked)}
                    />
                    Include Uncertainty Analysis
                  </label>
                  
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={includeExplanations}
                      onChange={e => setIncludeExplanations(e.target.checked)}
                    />
                    Include Molecular Explanations
                  </label>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={includeMolt5}
                      onChange={e => setIncludeMolt5(e.target.checked)}
                    />
                    Include MolT5 AI Analysis 🧠
                  </label>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="editor-container">
          <Editor
            structServiceProvider={structServiceProvider}
            onInit={ketcher => { 
              editorRef.current = ketcher;
              
              // Subscribe to Ketcher's change events for automatic updates
              ketcher.editor.subscribe('change', () => {
                updateMoleculePreview();   
              });
            }}
          />
        </div>        <div className="controls-section">
          {error && <div className="error">{error}</div>}
        </div>
      </div>

      {/* Right Panel - Laboratory Visualization */}
      <div className="right-panel">
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          width: '100%',
          marginBottom: '2rem'
        }}>
          <div className="control-group" style={{flex: '0 0 auto'}}>
            <label style={{color: 'white'}}>Select solvent:</label>
            <select value={selectedSolvent} onChange={e => setSelectedSolvent(e.target.value)}>
              <option value="">-- Choose Solvent --</option>
              {solvents.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>          <button className="check-button" onClick={handleCheck} disabled={loading || (!useEnhancedApi && !selectedSolvent) || (useEnhancedApi && !currentMolecule?.smiles && !selectedMolId)} style={{marginTop: 0}}>
            {loading ? 'Analyzing...' : useEnhancedApi ? 'Predict Solubility' : 'Check Solubility'}
          </button>
        </div>

        <div className="lab-container">
          {/* Molecule Preview */}
          <div className={`molecule-preview ${moleculeUpdated ? 'updated' : ''} ${moleculeImage ? 'has-image' : 'has-fallback'} ${isDropping ? 'dropping' : ''} ${moleculeVanished ? 'vanished' : ''} ${moleculeReappearing ? 'reappearing' : ''}`}>
            {moleculeImage ? (
              <img 
                src={moleculeImage} 
                alt="Molecule structure" 
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',                  
                  background: 'transparent'
                }}
                onError={() => {
                  setMoleculeImage(null);
                }}
              />
            ) : currentMolecule?.smiles && currentMolecule.smiles.trim() ? (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                height: '100%',
                fontSize: '0.8rem',
                color: '#333',
                textAlign: 'center',
                padding: '5px'
              }}>
                <div style={{ fontSize: '1.5rem', marginBottom: '5px' }}>🧬</div>
                <div>Molecule Detected</div>
                <div style={{ fontSize: '0.6rem', opacity: 0.7 }}>
                  {currentMolecule.smiles.length > 10 
                    ? currentMolecule.smiles.substring(0, 10) + '...' 
                    : currentMolecule.smiles}
                </div>
              </div>
            ) : (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                height: '100%',
                fontSize: '1rem',
                color: '#666'
              }}>
                Draw a molecule
              </div>
            )}
          </div>

          {/* Laboratory Flask */}
          <div className="flask-container">
            <div className="flask">
              {selectedSolvent && (
                <div className={`solvent ${selectedSolvent.toLowerCase()} ${showSplash ? 'splash' : ''}`}></div>
              )}
              {showSplash && (
                <>
                  <div className="splash-particle splash-1"></div>
                  <div className="splash-particle splash-2"></div>
                  <div className="splash-particle splash-3"></div>
                  <div className="splash-particle splash-4"></div>
                </>
              )}
            </div>
            {selectedSolvent && (
              <div className="flask-label">
                {selectedSolvent.charAt(0).toUpperCase() + selectedSolvent.slice(1)}
              </div>
            )}
          </div>

          {!selectedSolvent && (
            <div className="no-solvent-message">
              Select a solvent to see the laboratory setup
            </div>          )}
        </div>
      </div>

      {/* Results Modal */}
      {showResultModal && (
        <div className="modal-backdrop">
          <div className="modal-content">
            <div className="modal-header">
              <h2>Solubility Analysis Results</h2>
              <button className="close-button" onClick={() => setShowResultModal(false)}>×</button>
            </div>

            <div className="modal-body">
              <div className="result-section">
                <div className="result-item">
                  <label>SMILES:</label>
                  <span className="smiles-value">{result.smiles}</span>
                </div>

                <div className="result-item">
                  <label>Solubility Prediction:</label>
                  <span className="prediction-value">
                    {result.prediction.value.toFixed(3)} {result.prediction.unit}
                  </span>
                </div>
                
                <div className="result-item">
                  <label>Model Confidence:</label>
                  <span className="confidence-value">
                    {(result.prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                {result.prediction.uncertainty && (
                  <div className="uncertainty-section">
                    <h3>Uncertainty Analysis</h3>
                    <div className="uncertainty-grid">
                      <div className="uncertainty-item">
                        <label>Prediction Std:</label>
                        <span>{result.prediction.uncertainty.prediction_std.toFixed(4)}</span>
                      </div>
                      <div className="uncertainty-item">
                        <label>UAA Score:</label>
                        <span>{result.prediction.uncertainty.uaa_score.toFixed(4)}</span>
                      </div>
                      {result.prediction.uncertainty.aau_scores && (
                        <div className="uncertainty-item">
                          <label>AAU Scores:</label>
                          <span>{result.prediction.uncertainty.aau_scores.map(s => s.toFixed(3)).join(', ')}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {result.prediction.explanations && (
                  <div className="explanations-section">
                    <h3>Molecular Explanations</h3>
                    <div className="explanation-content">
                      {/* Remove the MEGAN interpretation display */}
                      {/* {result.prediction.explanations.interpretation && (
                        <div className="explanation-item">
                          <label>MEGAN Analysis:</label>
                          <p>{result.prediction.explanations.interpretation}</p>
                        </div>
                      )} */}
                      
                      {/* Keep only MolT5 Analysis Display */}
                      {result.prediction.explanations.molt5_interpretation && (
                        <div className="explanation-item molt5-analysis">
                          <label>🧠 MolT5 AI Analysis:</label>
                          <div className="molt5-content">
                            <pre className="molt5-analysis-text">{result.prediction.explanations.molt5_interpretation.analysis}</pre>
                            <div className="molt5-meta">
                              <small>
                                Confidence: {(result.prediction.explanations.molt5_interpretation.confidence * 100).toFixed(1)}% | 
                                Model: {result.prediction.explanations.molt5_interpretation.model_version || 'MolT5'}
                              </small>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {/* Standalone MolT5 analysis (when explanations are disabled but MolT5 is enabled) */}
                      {!result.prediction.explanations.molt5_interpretation && result.prediction.molt5_analysis && (
                        <div className="explanation-item molt5-analysis">
                          <label>🧠 MolT5 AI Analysis:</label>
                          <div className="molt5-content">
                            <pre className="molt5-analysis-text">{result.prediction.molt5_analysis.analysis}</pre>
                            <div className="molt5-meta">
                              <small>
                                Confidence: {(result.prediction.molt5_analysis.confidence * 100).toFixed(1)}% | 
                                Model: {result.prediction.molt5_analysis.model_version || 'MolT5'}
                              </small>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="modal-footer">
              <button className="modal-button" onClick={() => setShowResultModal(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
