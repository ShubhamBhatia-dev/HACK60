import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Cpu, Zap, Shield, BarChart2, ArrowRight, CheckCircle } from 'lucide-react';
import { api } from '../api/client';
import { useAppStore } from '../store/useAppStore';

// Animated mock document shown on the left panel
function AIDemo() {
  const lines = [
    { text: '# Senior Backend Engineer', delay: 0.2, bold: true },
    { text: 'Location: Remote · Full Time', delay: 0.6 },
    { text: 'Experience: 5+ years', delay: 1.0 },
    { text: '', delay: 1.3 },
    { text: '## About the Role', delay: 1.5, bold: true },
    { text: 'We are looking for a passionate engineer', delay: 2.0 },
    { text: 'to lead our platform infrastructure...', delay: 2.5 },
    { text: '', delay: 2.8 },
    { text: '• Build scalable microservices', delay: 3.0 },
    { text: '• Drive technical architecture', delay: 3.4 },
    { text: '• Mentor junior engineers', delay: 3.8 },
  ];
  return (
    <div className="ai-demo-card">
      <div className="ai-demo-topbar">
        <div className="ai-demo-dots">
          <span /><span /><span />
        </div>
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

export default function Login() {
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
      const response = await api.login({ email, password });
      setUser(response.user);
      navigate('/');
    } catch (err) {
      setError(err.message || 'Invalid credentials. Please try again.');
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
          <span className="auth-nav-text">New here?</span>
          <Link to="/signup" className="auth-nav-cta">Create account <ArrowRight size={12} /></Link>
        </div>
      </nav>

      {/* Body */}
      <div className="auth-body">
        {/* Left — hero + demo */}
        <div className="auth-left">
          <div className="auth-left-content">
            <div className="auth-hero-badge">
              <Zap size={12} /> AI-Powered HR Platform
            </div>
            <h1 className="auth-hero-title">
              Draft perfect<br />
              <span className="auth-hero-accent">Job Descriptions</span><br />
              in seconds
            </h1>
            <p className="auth-hero-sub">
              Our SLM writes tailored, professional job descriptions.
              Refine with LLM, edit, export — all in one workspace.
            </p>
            <ul className="auth-features">
              {[
                [Shield,    'Secure, private workspace per user'],
                [BarChart2, 'Version history & change tracking'],
                [Zap,       'SLM + LLM dual-model generation'],
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

            <h2 className="auth-title">Welcome back</h2>
            <p className="auth-subtitle">Sign in to your workspace</p>

            {error && <div className="auth-error">{error}</div>}

            <form onSubmit={handleSubmit}>
              <div className="input-group">
                <label>Email address</label>
                <input
                  id="login-email"
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
                  id="login-password"
                  type="password"
                  className="input"
                  placeholder="••••••••"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                />
              </div>
              <button
                id="login-submit"
                type="submit"
                className="btn-primary auth-submit-btn"
                disabled={loading}
              >
                {loading ? 'Signing in…' : 'Sign in'}
                {!loading && <ArrowRight size={14} />}
              </button>
            </form>

            <div className="auth-divider"><span>or</span></div>
            <div className="auth-switch">
              Don't have an account?{' '}
              <Link to="/signup">Create one free</Link>
            </div>
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
