import './App.css';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { useEffect } from 'react';
import EditorComponent from './components/EditorComponent';
import LandingPage from './components/LandingPage';

function AppContent() {
  const location = useLocation();
    useEffect(() => {
    // Add/remove body classes based on current route
    if (location.pathname === '/editor') {
      document.body.classList.add('editor-page-body');
      document.body.classList.remove('landing-page-body');
    } else {
      document.body.classList.add('landing-page-body');
      document.body.classList.remove('editor-page-body');
    }
  }, [location]);

  return (
    <div className="app-container">
      <Routes>
        <Route
          path="/"
          element={<div className="landing-page"><LandingPage /></div>}
        />
        <Route 
          path="/editor" 
          element={<div className="editor-page"><EditorComponent /></div>} 
        />
      </Routes>
    </div>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
