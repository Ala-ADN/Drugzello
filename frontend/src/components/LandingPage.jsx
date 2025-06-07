import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './LandingPage.css';

const LandingPage = () => {
  const [currentFeature, setCurrentFeature] = useState(0);
  const [isVisible, setIsVisible] = useState({});

  const features = [
    {
      icon: "🧪",
      title: "AI-Powered Molecular Analysis",
      description: "Advanced machine learning algorithms analyze molecular structures to predict drug properties with unprecedented accuracy."
    },
    {
      icon: "⚗️",
      title: "Solubility Prediction",
      description: "Instantly predict how well compounds dissolve in various solvents, crucial for drug formulation and bioavailability."
    },
    {
      icon: "🔬",
      title: "Interactive Visualization",
      description: "Real-time 3D molecular visualization and interactive laboratory simulations for intuitive understanding."
    },
    {
      icon: "📊",
      title: "Data-Driven Insights",
      description: "Comprehensive analytics and reporting tools to guide your drug discovery and development process."
    }
  ];

  const stats = [
    { number: "99.7%", label: "Prediction Accuracy" },
    { number: "10,000+", label: "Molecules Analyzed" },
    { number: "50ms", label: "Average Response Time" },
    { number: "24/7", label: "Availability" }
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
          <div className="hero-text">
            <h1 className="hero-title">
              <span className="gradient-text">Drugzello</span>
              <span className="subtitle">Revolutionizing Drug Discovery</span>
            </h1>
            <p className="hero-description">
              Harness the power of AI and machine learning to predict molecular properties, 
              analyze drug compounds, and accelerate pharmaceutical research with cutting-edge technology.
            </p>
            <div className="cta-buttons">
              <Link to="/editor" className="primary-cta">
                <span>Start Analyzing</span>
                <div className="button-shine"></div>
              </Link>
              <button className="secondary-cta">
                <span>Watch Demo</span>
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
        <div className="container">
          <h2 className="section-title">
            <span className="title-accent">Powerful Features</span>
            <span className="title-main">Built for Modern Research</span>
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
            <div className="tech-text">
              <h2>
                <span className="tech-accent">Advanced Technology</span>
                <span className="tech-main">At Your Fingertips</span>
              </h2>
              <div className="tech-features">
                <div className="tech-feature">
                  <div className="tech-icon">🤖</div>
                  <div>
                    <h4>Machine Learning Models</h4>
                    <p>State-of-the-art neural networks trained on extensive molecular databases</p>
                  </div>
                </div>
                <div className="tech-feature">
                  <div className="tech-icon">⚡</div>
                  <div>
                    <h4>Real-time Processing</h4>
                    <p>Lightning-fast analysis with instant results and interactive feedback</p>
                  </div>
                </div>
                <div className="tech-feature">
                  <div className="tech-icon">🔒</div>
                  <div>
                    <h4>Secure & Reliable</h4>
                    <p>Enterprise-grade security with 99.9% uptime guarantee</p>
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
          <div className="cta-content">
            <h2>Ready to Transform Your Research?</h2>
            <p>Join thousands of researchers already using Drugzello to accelerate their drug discovery process.</p>
            <div className="cta-actions">
              <Link to="/editor" className="cta-primary">
                Start Free Analysis
                <div className="button-ripple"></div>
              </Link>
              <button className="cta-secondary">
                Schedule Demo
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
              <p>Accelerating drug discovery through AI innovation</p>
            </div>
            <div className="footer-links">
              <div className="link-group">
                <h4>Product</h4>
                <a href="#features">Features</a>
                <a href="#tech">Technology</a>
                <a href="/editor">Try Now</a>
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
