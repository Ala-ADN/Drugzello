import './App.css';
import { Editor } from 'ketcher-react';
import { StandaloneStructServiceProvider } from 'ketcher-standalone';
import "ketcher-react/dist/index.css";
import { useEffect, useState, useRef } from 'react';

const structServiceProvider = new StandaloneStructServiceProvider();
const apiBase = import.meta.env.VITE_API_URL || '';

function App() {
  const editorRef = useRef(null);
  const [molecules, setMolecules] = useState([]);
  const [solvents, setSolvents] = useState([]);
  const [selectedMolId, setSelectedMolId] = useState(null);
  const [selectedSolvent, setSelectedSolvent] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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

  const handleCheck = async () => {
    setResult(null);
    setError('');
    setLoading(true);
    try {
      const payload = { solvent: selectedSolvent };
      if (selectedMolId) {
        payload.molecule_id = selectedMolId;
      } else {
        // extract SMILES from Ketcher editor
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
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="landing-container">
      <h1>Drugzello Solubility Checker</h1>

      <label>Select existing molecule or draw new:</label>
      <select value={selectedMolId || ''} onChange={e => setSelectedMolId(e.target.value || null)}>
        <option value="">-- Draw New --</option>
        {molecules.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
      </select>

      <div style={{ margin: '2rem 0', border: '1px solid #ccc', borderRadius: 8, overflow: 'hidden', width: '100%', height: 400 }}>
        <Editor
          structServiceProvider={structServiceProvider}
          onInit={editor => { editorRef.current = editor; }}
        />
      </div>

      <label>Select solvent:</label>
      <select value={selectedSolvent} onChange={e => setSelectedSolvent(e.target.value)}>
        <option value="">-- Choose Solvent --</option>
        {solvents.map(s => <option key={s} value={s}>{s}</option>)}
      </select>

      <button onClick={handleCheck} disabled={loading || !selectedSolvent}>
        {loading ? 'Checking...' : 'Check Solubility'}
      </button>

      {error && <p className="error">{error}</p>}
      {result && (
        <div className="result">
          <h2>Solubility: {result.solubility}</h2>
          <p>{result.explanation}</p>
        </div>
      )}
    </div>
  );
}

export default App;
