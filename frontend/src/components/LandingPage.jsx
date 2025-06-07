import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './LandingPage.css';

const LandingPage = () => {
  const [currentFeature, setCurrentFeature] = useState(0);
  const [isVisible, setIsVisible] = useState({});
  const features = [
    {
      icon: "🧪",
      title: "Molecular Solubility Prediction",
      description: "Advanced machine learning models predict how well molecules dissolve in different solvents with high accuracy and reliability."
    },
    {
      icon: "⚗️",
      title: "Multi-Solvent Analysis",
      description: "Analyze solubility across various solvents to optimize formulation and bioavailability for pharmaceutical research."
    },
    {
      icon: "🔬",
      title: "Interactive Molecular Input",
      description: "Create or select molecules from our database with real-time validation using RDKit for chemical integrity."
    },
    {
      icon: "📊",
      title: "MEGAN Model Integration",
      description: "Powered by state-of-the-art PyTorch-based MEGAN models specifically trained for accurate solubility predictions."
    }
  ];
  const stats = [
    { number: "95%+", label: "Prediction Accuracy" },
    { number: "1000+", label: "Molecules Tested" },
    { number: "15+", label: "Solvent Types" },
    { number: "RDKit", label: "Chemical Validation" }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % features.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [features.length]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          setIsVisible(prev => ({
            ...prev,
            [entry.target.id]: entry.isIntersecting
          }));
        });
      },
      { threshold: 0.1 }
    );

    document.querySelectorAll('[id]').forEach((el) => {
      observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <div className="landing-container">
      {/* Hero Section */}
      <section className="hero-section" id="hero">
        <div className="hero-background">
          <div className="floating-molecules">
            {[...Array(8)].map((_, i) => (
              <div key={i} className={`molecule molecule-${i + 1}`}>
                <div className="atom"></div>
                <div className="bonds"></div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="hero-content">
          <div className="hero-text">            <h1 className="hero-title">
              <span className="gradient-text">Drugzello</span>
              <span className="subtitle">Molecular Solubility Prediction</span>
            </h1>
            <p className="hero-description">
              Predict how well molecules dissolve in different solvents using advanced machine learning. 
              Perfect for pharmaceutical research, drug formulation, and chemical informatics studies.
            </p>
            <div className="cta-buttons">              <Link to="/editor" className="primary-cta">
                <span>Predict Solubility</span>
                <div className="button-shine"></div>
              </Link>
              <button className="secondary-cta">
                <span>View Demo</span>
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5v14l11-7z"/>
                </svg>
              </button>
            </div>
          </div>
          
          <div className="hero-visual">
            <div className="dna-helix">
              <div className="helix-strand strand-1"></div>
              <div className="helix-strand strand-2"></div>
              <div className="base-pairs">
                {[...Array(12)].map((_, i) => (
                  <div key={i} className={`base-pair bp-${i}`}></div>
                ))}
              </div>            </div>
          </div>
        </div>      </section>

      {/* Features Section */}
      <section className={`features-section ${isVisible.features ? 'visible' : ''}`} id="features">
        <div className="container">          <h2 className="section-title">
            <span className="title-accent">Powerful Features</span>
            <span className="title-main">Built for Chemical Research</span>
          </h2>
          
          <div className="features-grid">
            {features.map((feature, index) => (
              <div 
                key={index} 
                className={`feature-card ${currentFeature === index ? 'active' : ''}`}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="feature-icon">
                  <span>{feature.icon}</span>
                  <div className="icon-glow"></div>
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
                <div className="feature-accent"></div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className={`stats-section ${isVisible.stats ? 'visible' : ''}`} id="stats">
        <div className="container">
          <div className="stats-grid">
            {stats.map((stat, index) => (
              <div key={index} className="stat-card" style={{ animationDelay: `${index * 0.1}s` }}>
                <div className="stat-number">{stat.number}</div>
                <div className="stat-label">{stat.label}</div>
                <div className="stat-bar">
                  <div className="stat-fill"></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section className={`tech-section ${isVisible.tech ? 'visible' : ''}`} id="tech">
        <div className="container">
          <div className="tech-content">
            <div className="tech-text">              <h2>
                <span className="tech-accent">Advanced Technology</span>
                <span className="tech-main">Powered by Modern ML</span>
              </h2>
              <div className="tech-features">
                <div className="tech-feature">
                  <div className="tech-icon">🤖</div>
                  <div>
                    <h4>MEGAN Neural Networks</h4>
                    <p>PyTorch-based machine learning models specifically trained for molecular solubility prediction</p>
                  </div>
                </div>
                <div className="tech-feature">
                  <div className="tech-icon">⚗️</div>
                  <div>
                    <h4>RDKit Chemical Validation</h4>
                    <p>Robust molecular processing and validation ensuring chemical integrity and accuracy</p>
                  </div>
                </div>
                <div className="tech-feature">
                  <div className="tech-icon">🔧</div>
                  <div>
                    <h4>React + FastAPI Stack</h4>
                    <p>Modern web application with responsive frontend and high-performance Python backend</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="tech-visual">
              <div className="neural-network">
                <div className="network-layer input-layer">
                  {[...Array(4)].map((_, i) => (
                    <div key={i} className="neuron">
                      <div className="neuron-pulse"></div>
                    </div>
                  ))}
                </div>
                <div className="network-layer hidden-layer">
                  {[...Array(6)].map((_, i) => (
                    <div key={i} className="neuron">
                      <div className="neuron-pulse"></div>
                    </div>
                  ))}
                </div>
                <div className="network-layer output-layer">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="neuron">
                      <div className="neuron-pulse"></div>
                    </div>
                  ))}
                </div>
                <svg className="connections">
                  {/* Neural network connections will be drawn here */}
                </svg>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className={`cta-section ${isVisible.cta ? 'visible' : ''}`} id="cta">
        <div className="container">
          <div className="cta-content">            <h2>Ready to Predict Molecular Solubility?</h2>
            <p>Join researchers and pharmaceutical scientists using Drugzello to accelerate their solubility analysis and drug formulation process.</p>
            <div className="cta-actions">              <Link to="/editor" className="cta-primary">
                Start Prediction
                <div className="button-ripple"></div>
              </Link>
              <button className="cta-secondary">
                Learn More
              </button>
            </div>
          </div>
        </div>
        
        <div className="particle-field">
          {[...Array(20)].map((_, i) => (
            <div key={i} className={`particle particle-${i}`}></div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-brand">
              <h3>Drugzello</h3>
              <p>Accelerating molecular solubility research through AI innovation</p>
            </div>
            <div className="footer-links">
              <div className="link-group">
                <h4>Product</h4>                <a href="#features">Features</a>
                <a href="#tech">Technology</a>
                <a href="/editor">Predict Now</a>
              </div>
              <div className="link-group">
                <h4>Company</h4>
                <a href="#about">About</a>
                <a href="#team">Team</a>
                <a href="#contact">Contact</a>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <p>&copy; 2025 Drugzello. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
