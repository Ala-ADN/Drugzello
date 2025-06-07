import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Editor } from 'ketcher-react';
import { StandaloneStructServiceProvider } from 'ketcher-standalone';
import "ketcher-react/dist/index.css";
import './EditorComponent.css'; // Assuming you have a CSS file for styles
const structServiceProvider = new StandaloneStructServiceProvider();
const apiBase = import.meta.env.VITE_API_URL || '';

const EditorComponent = () => {
  const editorRef = useRef(null);
  const [molecules, setMolecules] = useState([]);
  const [solvents, setSolvents] = useState([]);
  const [selectedMolId, setSelectedMolId] = useState(null);
  const [selectedSolvent, setSelectedSolvent] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [currentMolecule, setCurrentMolecule] = useState(null);
  const [moleculeUpdated, setMoleculeUpdated] = useState(false);
  const [moleculeImage, setMoleculeImage] = useState(null);
  const [isDropping, setIsDropping] = useState(false);
  const [showSplash, setShowSplash] = useState(false);
  const [moleculeVanished, setMoleculeVanished] = useState(false);
  const [moleculeReappearing, setMoleculeReappearing] = useState(false);
  const [isCheckingMolecule, setIsCheckingMolecule] = useState(false);
  const [showResultModal, setShowResultModal] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/molecules`)
      .then(res => res.json())
      .then(setMolecules)
      .catch(() => {});
    fetch(`${apiBase}/solvents`)
      .then(res => res.json())
      .then(setSolvents)
      .catch(() => {});
  }, []);

  const updateMoleculePreview = useCallback(async () => {
    try {
      if (editorRef.current) {
        const smiles = await editorRef.current.getSmiles();
        const molfile = await editorRef.current.getMolfile();
        const newMolecule = { smiles, molfile };

        if (smiles && smiles.trim() && smiles !== '') {
          try {
            const imageBlob = await editorRef.current.generateImage(molfile, {
              outputFormat: 'svg'
            });

            if (imageBlob) {
              const imageUrl = URL.createObjectURL(imageBlob);
              setMoleculeImage(imageUrl);
            }
          } catch {
            try {
              const pngBlob = await editorRef.current.generateImage(molfile, {
                outputFormat: 'png'
              });

              if (pngBlob) {
                const imageUrl = URL.createObjectURL(pngBlob);
                setMoleculeImage(imageUrl);
              } else {
                setMoleculeImage(null);
              }
            } catch {
              setMoleculeImage(null);
            }
          }
        } else {
          setMoleculeImage(null);
        }

        const moleculeChanged = currentMolecule && currentMolecule.smiles !== smiles;
        const isSignificantChange = moleculeChanged && smiles && smiles.trim() !== '';

        if (isSignificantChange && !isCheckingMolecule) {
          if (moleculeVanished) {
            setMoleculeVanished(false);
            setMoleculeReappearing(true);
            setTimeout(() => setMoleculeReappearing(false), 800);
          } else if (!moleculeVanished) {
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

  useEffect(() => {
    if (selectedMolId && editorRef.current) {
      setTimeout(() => {
        updateMoleculePreview();
      }, 500);
    }
  }, [selectedMolId, updateMoleculePreview]);

  useEffect(() => {
    if (moleculeVanished && selectedSolvent) {
      setMoleculeVanished(false);
      setMoleculeReappearing(true);
      setTimeout(() => setMoleculeReappearing(false), 800);
    }
  }, [selectedSolvent, moleculeVanished]);

  useEffect(() => {
    if (moleculeVanished && selectedMolId) {
      setMoleculeVanished(false);
      setMoleculeReappearing(true);
      setTimeout(() => setMoleculeReappearing(false), 800);
    }
  }, [selectedMolId, moleculeVanished]);

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
    setIsCheckingMolecule(true);

    if ((currentMolecule?.smiles || selectedMolId) && selectedSolvent) {
      setIsDropping(true);

      setTimeout(() => {
        setShowSplash(true);
        setTimeout(() => {
          setShowSplash(false);
          setIsDropping(false);
          setMoleculeVanished(true);
        }, 1000);
      }, 1500);
    }

    try {
      const payload = { solvent: selectedSolvent };
      if (selectedMolId) {
        payload.molecule_id = selectedMolId;
      } else {
        const smiles = await editorRef.current?.getSmiles();
        payload.smiles = smiles;
      }
      const res = await fetch(`${apiBase}/solubility`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Error');
      setResult(data);
      setShowResultModal(true);
    } catch (e) {
      setError(e.message);
      setIsDropping(false);
      setShowSplash(false);
      setMoleculeVanished(false);
    } finally {
      setLoading(false);
      setIsCheckingMolecule(false);
    }
  };
  return (
    <div className="editor-container">
      {/* Animated background particles */}
      <div className="particles-bg">
        {[...Array(20)].map((_, i) => (
          <div key={i} className={`particle particle-${i % 4}`}></div>
        ))}
      </div>
      
      <div className="left-panel">
        <div className="panel-header">
          <div className="logo-container">
            <div className="dna-helix"></div>
            <h1 className="panel-title">
              <span className="title-gradient">Molecular</span>
              <span className="title-accent">Designer</span>
            </h1>
          </div>
          <p className="panel-subtitle">Design and analyze molecular structures with precision</p>
        </div>

        <div className="controls-section">
          <div className="control-group">
            <label className="modern-label">
              <span className="label-icon">🧪</span>
              Select existing molecule or draw new:
            </label>
            <div className="select-wrapper">
              <select value={selectedMolId || ''} onChange={e => setSelectedMolId(e.target.value || null)}>
                <option value="">✨ Draw New Molecule</option>
                {molecules.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
              <div className="select-arrow"></div>
            </div>
          </div>        </div>
        
        <div className="editor-wrapper">
          <div className="editor-header">
            <h3>Molecular Structure Editor</h3>
            <div className="editor-status">
              {currentMolecule?.smiles ? (
                <span className="status-active">
                  <span className="status-dot"></span>
                  Molecule detected
                </span>
              ) : (
                <span className="status-waiting">
                  <span className="status-dot"></span>
                  Draw structure
                </span>
              )}
            </div>
          </div>
          <div className="editor-frame">
            <Editor
              structServiceProvider={structServiceProvider}
              onInit={ketcher => {
                editorRef.current = ketcher;
                ketcher.editor.subscribe('change', () => {
                  updateMoleculePreview();
                });
              }}
            />
          </div>
        </div>
        <div className="controls-section">
          {error && <div className="error">{error}</div>}
        </div>
      </div>      <div className="right-panel">
        <div className="panel-header">
          <h2 className="analysis-title">
            <span className="title-gradient">Solubility</span>
            <span className="title-accent">Analysis</span>
          </h2>
          <p className="panel-subtitle">Predict molecular solubility in various solvents</p>
        </div>

        <div className="analysis-controls">
          <div className="control-group solvent-selector">
            <label className="modern-label">
              <span className="label-icon">⚗️</span>
              Select solvent:
            </label>
            <div className="select-wrapper">
              <select value={selectedSolvent} onChange={e => setSelectedSolvent(e.target.value)}>
                <option value="">🔬 Choose Solvent</option>
                {solvents.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
              <div className="select-arrow"></div>
            </div>
          </div>

          <button className="predict-button" onClick={handleCheck} disabled={loading || !selectedSolvent}>
            <span className="button-icon">🧬</span>
            <span className="button-text">
              {loading ? 'Analyzing...' : 'Predict Solubility'}
            </span>
            {loading && <div className="loading-spinner"></div>}
          </button>
        </div>        <div className="lab-container">
          <div className="lab-title">
            <h3>Virtual Laboratory</h3>
            <p>Watch the molecular interaction in real-time</p>
          </div>
          
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
              <div className="molecule-placeholder">
                <div className="molecule-icon">🧬</div>
                <div className="molecule-status">Molecule Ready</div>
                <div className="molecule-smiles">
                  {currentMolecule.smiles.length > 12
                    ? currentMolecule.smiles.substring(0, 12) + '...'
                    : currentMolecule.smiles}
                </div>
              </div>
            ) : (
              <div className="molecule-placeholder empty">
                <div className="molecule-icon">⚛️</div>
                <div className="molecule-status">Draw a molecule</div>
                <div className="molecule-hint">Use the editor above</div>
              </div>
            )}
          </div>

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
            <div className="setup-instructions">
              <div className="instruction-icon">🔬</div>
              <h4>Ready to Analyze</h4>
              <p>Select a solvent above to begin the molecular interaction simulation</p>
            </div>
          )}
        </div>
      </div>      {showResultModal && result && (
        <div className="modal-overlay" onClick={() => setShowResultModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-section">
                <div className="modal-icon">🎯</div>
                <div>
                  <h2>Analysis Complete</h2>
                  <p>Molecular solubility prediction results</p>
                </div>
              </div>
              <button className="modal-close" onClick={() => setShowResultModal(false)}>
                <span>×</span>
              </button>
            </div>
            <div className="modal-body">
              <div className="result-card">
                <div className="result-header">
                  <span className="result-label">Predicted Solubility</span>
                  <div className="result-badge">
                    <span className="solubility-value">
                      {result.solubility}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="explanation-card">
                <div className="explanation-header">
                  <span className="explanation-icon">🧠</span>
                  <span className="explanation-label">AI Analysis</span>
                </div>
                <div className="explanation-content">
                  <p>{result.explanation}</p>
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="modal-button primary" onClick={() => setShowResultModal(false)}>
                <span className="button-icon">✅</span>
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EditorComponent;