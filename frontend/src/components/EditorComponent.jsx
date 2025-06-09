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
  const [moleculeUpdated, setMoleculeUpdated] = useState(false);  const [moleculeImage, setMoleculeImage] = useState(null);
  const [isDropping, setIsDropping] = useState(false);
  const [moleculeVanished, setMoleculeVanished] = useState(false);
  const [moleculeReappearing, setMoleculeReappearing] = useState(false);
  const [isCheckingMolecule, setIsCheckingMolecule] = useState(false);
  const [showResultModal, setShowResultModal] = useState(false);  const [isSplashing, setIsSplashing] = useState(false);
  const [isDissolving, setIsDissolving] = useState(false);

  // Animation reset function
  const resetAllAnimations = useCallback(() => {
    setIsDropping(false);
    setMoleculeVanished(false);
    setMoleculeReappearing(false);
    setIsSplashing(false);
    setIsDissolving(false);
    setIsCheckingMolecule(false);
  }, []);

  // Molecule reappearance function
  const triggerMoleculeReappearance = useCallback(() => {
    setMoleculeReappearing(true);
    setTimeout(() => {
      setMoleculeReappearing(false);
    }, 800); // Match animation duration
  }, []);

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
        // Reset all animations and trigger molecule reappearance after modal closes
        setTimeout(() => {
          resetAllAnimations();
          if (currentMolecule?.smiles || selectedMolId) {
            triggerMoleculeReappearance();
          }
        }, 100);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [showResultModal, resetAllAnimations, triggerMoleculeReappearance, currentMolecule?.smiles, selectedMolId]);  const handleCheck = async () => {    setResult(null);
    setError('');
    setLoading(true);
    setIsCheckingMolecule(true);    if ((currentMolecule?.smiles || selectedMolId) && selectedSolvent) {
      // Start the drop animation
      setIsDropping(true);

      // After molecule reaches flask, create splash (molecule still visible)
      setTimeout(() => {
        setIsSplashing(true);
      }, 1000); // Molecule reaches flask

      // After splash completes, start dissolving and vanish molecule
      setTimeout(() => {
        setIsSplashing(false);
        setIsDissolving(true);
        setMoleculeVanished(true);
      }, 2200); // Give splash time to complete (0.8s) + buffer

      // End dissolving animation
      setTimeout(() => {
        setIsDissolving(false);
      }, 4800); // Allow dissolve to run for 1.6s
    }

    // Store API result to show after animation completes
    let apiResult = null;
    let apiError = null;

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
      });      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Error');
      apiResult = data;
        // If animation is running, delay showing results until animation completes
      if ((currentMolecule?.smiles || selectedMolId) && selectedSolvent) {
        setTimeout(() => {
          setResult(apiResult);
          setShowResultModal(true);
          setLoading(false);
          setIsCheckingMolecule(false);
        }, 2500); // Show modal 200ms after animation ends (4.8s + 0.2s)
      } else {
        // No animation, show results immediately
        setResult(apiResult);
        setShowResultModal(true);
        setLoading(false);
        setIsCheckingMolecule(false);
      }
    } catch (e) {
      apiError = e.message;
      setError(apiError);
      setIsDropping(false);
      setMoleculeVanished(false);
      setIsSplashing(false);
      setIsDissolving(false);
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
            <h1 className="panel-title">
              <span className="title-gradient">Molecular</span>
              <span className="title-accent">Designer</span>
            </h1>
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
              buttons={{
                layout: { hidden: true },
                clean: { hidden: true },
                arom: { hidden: true },
                dearom: { hidden: true },
                cip: { hidden: true },
                check: { hidden: true },
                analyse: { hidden: true },
                recognize: { hidden: true },
                miew: { hidden: true },
                save: { hidden: true },
                load: { hidden: true },
                open: { hidden: true },
                copy: { hidden: true },
                paste: { hidden: true },
                cut: { hidden: true },
                about: { hidden: true },
                help: { hidden: true },
                undo: { hidden: true },
                redo: { hidden: true },
                settings: { hidden: true },
                text: { hidden: true },
                shape: { hidden: true },
                'shape-ellipse': { hidden: true },
                'shape-rectangle': { hidden: true },
                'shape-line': { hidden: true },
                arrows: { hidden: true },
                'reaction-mapping-tools': { hidden: true },
                'reaction-plus': { hidden: true },
              }}
            />
          </div>
        </div>
        <div className="controls-section">
          {error && <div className="error">{error}</div>}
        </div>
      </div>      
      <div className="right-panel">
        <div className="panel-header">
          <h1 className="analysis-title">
              <span className="title-gradient">Solubility</span>
              <span className="title-accent">Analysis</span>
          </h1>
        </div>        
        <div className="analysis-controls">
          <div className="control-group solvent-selector">
            <label className="modern-label">
              <span className="label-icon">⚗️</span>
              Select solvent:
            </label>
            <div className="horizontal-controls">
              <div className="select-wrapper">
                <select value={selectedSolvent} onChange={e => setSelectedSolvent(e.target.value)}>
                  <option value="">🔬 Choose Solvent</option>
                  {solvents.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
                <div className="select-arrow"></div>
              </div>
              <button className="predict-button" onClick={handleCheck} disabled={loading || !selectedSolvent}>
                <span className="button-text">
                  {loading ? 'Analyzing...' : '🧬 Predict Solubility'}
                </span>
                {loading && <div className="loading-spinner"></div>}
              </button>
            </div>
          </div>
        </div>
          <div className="lab-container">
          <div className="lab-title">
            <h3>Virtual Laboratory</h3>
            <p>Watch the molecular interaction in real-time</p>
          </div>
            <div className="lab-content">            <div className={`molecule-preview ${moleculeUpdated ? 'updated' : ''} ${moleculeImage ? 'has-image showing-molecule' : 'has-fallback'} ${isDropping ? 'dropping' : ''} ${moleculeVanished ? 'vanished' : ''} ${moleculeReappearing ? 'reappearing' : ''}`}>
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
            </div>            {/* U-shaped Flask - Always visible */}
            <div className={`flask-container ${selectedSolvent ? 'has-solvent' : ''} ${selectedSolvent ? `solvent-${selectedSolvent.toLowerCase()}` : ''} ${isSplashing ? 'splashing' : ''} ${isDissolving ? 'dissolving' : ''}`}>
              <div className="u-shaped-flask">
                <div className="flask-base">
                  <div className="solvent-level"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>      {showResultModal && result && (
        <div className="modal-overlay" onClick={() => {
          setShowResultModal(false);
          // Reset all animations and trigger molecule reappearance after modal closes
          setTimeout(() => {
            resetAllAnimations();
            if (currentMolecule?.smiles || selectedMolId) {
              triggerMoleculeReappearance();
            }
          }, 100);
        }}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-section">
                <div className="modal-icon">🎯</div>
                <div>
                  <h2>Analysis Complete</h2>
                  <p>Molecular solubility prediction results</p>
                </div>
              </div>
              <button className="modal-close" onClick={() => {
                setShowResultModal(false);
                // Reset all animations and trigger molecule reappearance after modal closes
                setTimeout(() => {
                  resetAllAnimations();
                  if (currentMolecule?.smiles || selectedMolId) {
                    triggerMoleculeReappearance();
                  }
                }, 100);
              }}>
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

              {/* MEGAN XAI SVG Visualizations */}
              <div className="attributions-card">
                <div className="attributions-header">
                  <span className="attributions-icon">🔍</span>
                  <span className="attributions-label">Model Attributions</span>
                </div>
                <div className="attributions-content">
                  <div className="svg-section">
                    <div className="svg-label">ML Attributions</div>
                    <div className="svg-container" dangerouslySetInnerHTML={{ __html: result.svg_ml }} />
                  </div>
                  <div className="svg-section">
                    <div className="svg-label">Atomic Attributions</div>
                    <div className="svg-container" dangerouslySetInnerHTML={{ __html: result.svg_atomic }} />
                  </div>
                  <div className="svg-section">
                    <div className="svg-label">FPA Attributions</div>
                    <div className="svg-container" dangerouslySetInnerHTML={{ __html: result.svg_fpa }} />
                  </div>
                  <div className="svg-section">
                    <div className="svg-label">UAA Attributions</div>
                    <div className="svg-container" dangerouslySetInnerHTML={{ __html: result.svg_uaa }} />
                  </div>
                  <div className="svg-section">
                    <div className="svg-label">AAU Attributions</div>
                    <div className="svg-container" dangerouslySetInnerHTML={{ __html: result.svg_aau }} />
                  </div>
                </div>
              </div>
            </div>            <div className="modal-footer">
              <button className="modal-button primary" onClick={() => {
                setShowResultModal(false);
                // Reset all animations and trigger molecule reappearance after modal closes
                setTimeout(() => {
                  resetAllAnimations();
                  if (currentMolecule?.smiles || selectedMolId) {
                    triggerMoleculeReappearance();
                  }
                }, 100);
              }}>
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