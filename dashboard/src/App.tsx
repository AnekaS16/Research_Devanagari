import React, { useState, useEffect } from 'react';
import {
  Layers,
  Database,
  Activity,
  GitCompare,
  Flame,
  FileCheck2,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Code2,
} from 'lucide-react';
import { loadAllDashboardData } from './data/dataLoader';
import type { DashboardState } from './types';
import { ResearchOverview } from './components/ResearchOverview';
import { DatasetExplorer } from './components/DatasetExplorer';
import { TokenizationScoring } from './components/TokenizationScoring';
import { ExplanationStability } from './components/ExplanationStability';
import { FaithfulnessLab } from './components/FaithfulnessLab';
import { FindingsLimitations } from './components/FindingsLimitations';

type Tab = 'overview' | 'explorer' | 'tokenization' | 'stability' | 'faithfulness' | 'findings';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [state, setState] = useState<DashboardState>({
    overview: null,
    tracks: [],
    tokenization: [],
    scoring: [],
    stability: [],
    faithfulness: [],
    pairMaster: [],
    eligibility: [],
    auditReport: '',
    loading: true,
    error: null,
  });

  useEffect(() => {
    loadAllDashboardData()
      .then((data) => {
        setState({
          ...data,
          loading: false,
          error: null,
        });
      })
      .catch((err) => {
        console.error('Failed to load dashboard data:', err);
        setState((prev) => ({
          ...prev,
          loading: false,
          error: `Error loading research data files: ${err.message || String(err)}`,
        }));
      });
  }, []);

  const navItems = [
    { id: 'overview', label: 'Research Overview', icon: Layers },
    { id: 'explorer', label: 'Dataset Explorer', icon: Database },
    { id: 'tokenization', label: 'Tokenization & Scoring', icon: Activity },
    { id: 'stability', label: 'Explanation Stability', icon: GitCompare },
    { id: 'faithfulness', label: 'Faithfulness Laboratory', icon: Flame },
    { id: 'findings', label: 'Findings & Limitations', icon: FileCheck2 },
  ];

  if (state.loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', gap: '16px' }}>
        <Loader2 className="animate-spin" size={36} color="var(--accent-indigo)" />
        <div style={{ fontSize: '15px', color: 'var(--text-secondary)', fontWeight: 500 }}>
          Loading audited Hindi NLP research data layer...
        </div>
      </div>
    );
  }

  if (state.error) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', padding: '24px' }}>
        <div className="alert-box alert-warning" style={{ maxWidth: '600px' }}>
          <AlertCircle size={24} />
          <div>
            <strong>Dashboard Data Loading Error:</strong>
            <p style={{ marginTop: '6px' }}>{state.error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="brand-header">
          <div className="brand-title">
            <Code2 size={22} color="var(--accent-cyan)" />
            Hindi Lexical Sensitivity
          </div>
          <div className="brand-subtitle">
            Explanation Stability vs Faithfulness
          </div>
        </div>

        <nav className="nav-menu">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => setActiveTab(item.id as Tab)}
              >
                <Icon size={17} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#34d399', fontWeight: 600 }}>
            <CheckCircle2 size={13} /> Stages 1–8 Audited
          </div>
          <div>Windows 11 CPU Environment</div>
          <div style={{ fontFamily: 'var(--font-mono)' }}>SHA-256 Verified</div>
        </div>
      </aside>

      {/* Main Viewport */}
      <div className="main-viewport">
        {/* Top Sticky Bar */}
        <header className="top-bar">
          <div className="top-title">
            <span>
              {navItems.find((n) => n.id === activeTab)?.label}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span className="badge badge-muril">google/muril-base-cased</span>
            <span className="badge badge-xlmr">FacebookAI/xlm-roberta-base</span>
            <div className="provenance-pill">
              <CheckCircle2 size={13} /> Audited Data Layer
            </div>
          </div>
        </header>

        {/* Content View */}
        <main className="content-container">
          {activeTab === 'overview' && (
            <ResearchOverview overview={state.overview} tracks={state.tracks} />
          )}
          {activeTab === 'explorer' && (
            <DatasetExplorer data={state.pairMaster} />
          )}
          {activeTab === 'tokenization' && (
            <TokenizationScoring tokenization={state.tokenization} scoring={state.scoring} />
          )}
          {activeTab === 'stability' && (
            <ExplanationStability stability={state.stability} pairMaster={state.pairMaster} />
          )}
          {activeTab === 'faithfulness' && (
            <FaithfulnessLab faithfulness={state.faithfulness} />
          )}
          {activeTab === 'findings' && (
            <FindingsLimitations />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
