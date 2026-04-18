import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Cpu, Zap, Shield, BarChart2, ArrowRight, CheckCircle } from 'lucide-react';
import { api } from '../api/client';
import { useAppStore } from '../store/useAppStore';

// Reuse the same AI demo animation
function AIDemo() {
  const lines = [
    { text: '# Product Designer — Mid Level', delay: 0.2, bold: true },
    { text: 'Location: Hybrid · Full Time', delay: 0.6 },
    { text: '', delay: 1.0 },
    { text: '## Responsibilities', delay: 1.2, bold: true },
    { text: '• Design end-to-end user flows', delay: 1.7 },
    { text: '• Collaborate with engineering', delay: 2.1 },
    { text: '• Conduct user research sessions', delay: 2.5 },
    { text: '', delay: 2.8 },
    { text: '## Requirements', delay: 3.0, bold: true },
    { text: '• 3+ years of product design experience', delay: 3.5 },
    { text: '• Proficiency in Figma & design systems', delay: 4.0 },
  ];
  return (
    <div className="ai-demo-card">
      <div className="ai-demo-topbar">
        <div className="ai-demo-dots"><span /><span /><span /></div>
        <div className="ai-demo-title">
          <Cpu size={10} style={{ opacity: 0.7 }} />
          HR SLM · Generating…
        </div>
      </div>
      <div className="ai-demo-body">
        {lines.map((l, i) =>
          l.text === '' ? (
            <div key={i} style={{ height: '0.5rem' }} />
          ) : (
            <div
              key={i}
              className="ai-demo-line"
              style={{ animationDelay: `${l.delay}s`, fontWeight: l.bold ? 700 : 400 }}
            >
              {l.text}
            </div>
          )
        )}
        <span className="ai-cursor" />
      </div>
    </div>
  );
}

export default function Signup() {
  const [name,     setName]     = useState('');
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState('');
  const navigate = useNavigate();
  const setUser  = useAppStore(s => s.setUser);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const response = await api.signup({ name, email, password });
      setUser(response.user);
      navigate('/');
    } catch (err) {
      setError(err.message || 'Signup failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      {/* Navbar */}
      <nav className="auth-nav">
        <div className="auth-nav-brand">
          <div className="auth-nav-icon"><Cpu size={15} /></div>
          <span>HR SLM</span>
        </div>
        <div className="auth-nav-links">
          <span className="auth-nav-text">Already have an account?</span>
          <Link to="/login" className="auth-nav-cta">Sign in <ArrowRight size={12} /></Link>
        </div>
      </nav>

      {/* Body */}
      <div className="auth-body">
        {/* Left */}
        <div className="auth-left">
          <div className="auth-left-content">
            <div className="auth-hero-badge">
              <Zap size={12} /> AI-Powered HR Platform
            </div>
            <h1 className="auth-hero-title">
              Your AI writing<br />
              <span className="auth-hero-accent">co-pilot</span><br />
              for hiring
            </h1>
            <p className="auth-hero-sub">
              Generate, refine, and export professional job descriptions
              powered by a private SLM — fine-tuned on your feedback.
            </p>
            <ul className="auth-features">
              {[
                [CheckCircle, 'Free to start — no credit card needed'],
                [Shield,      'Your data stays private & secure'],
                [Zap,         'Generate your first JD in under 30s'],
              ].map(([Icon, text], i) => (
                <li key={i}>
                  <Icon size={13} className="auth-feature-icon" />
                  {text}
                </li>
              ))}
            </ul>
            <AIDemo />
          </div>
        </div>

        {/* Right — form */}
        <div className="auth-right">
          <div className="auth-form-card">
            <div className="auth-brand">
              <div className="auth-brand-icon"><Cpu size={16} /></div>
              <span className="auth-brand-name">HR SLM</span>
            </div>

            <h2 className="auth-title">Create account</h2>
            <p className="auth-subtitle">Start generating job descriptions for free</p>

            {error && <div className="auth-error">{error}</div>}

            <form onSubmit={handleSubmit}>
              <div className="input-group">
                <label>Full name</label>
                <input
                  id="signup-name"
                  type="text"
                  className="input"
                  placeholder="Jane Smith"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  required
                />
              </div>
              <div className="input-group">
                <label>Email address</label>
                <input
                  id="signup-email"
                  type="email"
                  className="input"
                  placeholder="you@company.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                />
              </div>
              <div className="input-group">
                <label>Password</label>
                <input
                  id="signup-password"
                  type="password"
                  className="input"
                  placeholder="At least 8 characters"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  minLength={8}
                  required
                />
              </div>
              <button
                id="signup-submit"
                type="submit"
                className="btn-primary auth-submit-btn"
                disabled={loading}
              >
                {loading ? 'Creating account…' : 'Create free account'}
                {!loading && <ArrowRight size={14} />}
              </button>
            </form>

            <div className="auth-divider"><span>or</span></div>
            <div className="auth-switch">
              Already have an account?{' '}
              <Link to="/login">Sign in</Link>
            </div>

            <p className="auth-legal">
              By creating an account you agree to our{' '}
              <a href="#">Terms of Service</a> and{' '}
              <a href="#">Privacy Policy</a>.
            </p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="auth-footer-bar">
        <span>© 2025 HR SLM · AI-Powered Hiring Tools</span>
        <span className="auth-footer-links">
          <a href="#">Privacy</a>
          <a href="#">Terms</a>
          <a href="#">Contact</a>
        </span>
      </footer>
    </div>
  );
}
