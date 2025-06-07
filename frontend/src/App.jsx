import './App.css';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import EditorComponent from './components/EditorComponent';

function App() {
  return (
    <Router>
      <div className="app-container">
        <Routes>
          <Route
            path="/"
            element={
              <div className="landing-page">
                <h1>Welcome to Drugzello</h1>
                <Link to="/editor">
                  <button className="get-started-button">Get Started</button>
                </Link>
              </div>
            }
          />
          <Route path="/editor" element={<EditorComponent />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
