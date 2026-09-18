'use client';
import React from 'react';
import { AlertTriangle, Ship } from 'lucide-react';
import { InvestigationCandidate } from '@/types/investigation';

interface CandidateTableProps {
  candidates: InvestigationCandidate[];
  onSelectCandidate?: (c: InvestigationCandidate) => void;
  activeMMSI?: string | null;
  dataMode: 'DEMO' | 'LIVE';
  disclaimer: string;
}

function ScoreBar({ value, max = 1.0 }: { value: number; max?: number }) {
  const pct = Math.round((value / max) * 100);
  const color = pct >= 70 ? '#f43f5e' : pct >= 40 ? '#fbbf24' : '#00d7b2';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <div style={{ width: '60px', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: '2px' }} />
      </div>
      <span style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color }}>{pct}%</span>
    </div>
  );
}

export default function CandidateTable({
  candidates,
  onSelectCandidate,
  activeMMSI,
  dataMode,
  disclaimer,
}: CandidateTableProps) {
  if (candidates.length === 0) {
    return (
      <div style={{
        background: 'rgba(11,23,35,0.95)',
        border: '1px solid rgba(0,215,178,0.12)',
        borderRadius: '6px',
        padding: '24px',
        textAlign: 'center',
        color: 'var(--text-muted)',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.75rem',
      }}>
        <Ship size={24} style={{ opacity: 0.3, marginBottom: '8px' }} />
        <div>NO CANDIDATE VESSELS FOUND</div>
        <div style={{ fontSize: '0.62rem', marginTop: '4px', opacity: 0.6 }}>
          No AIS records matched the trajectory corridor.
          This may indicate no vessels were tracked in this area/time window.
        </div>
      </div>
    );
  }

  return (
    <div style={{
      background: 'rgba(11,23,35,0.97)',
      border: '1px solid rgba(0,215,178,0.15)',
      borderRadius: '6px',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '10px 16px',
        borderBottom: '1px solid rgba(0,215,178,0.12)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(0,215,178,0.04)',
      }}>
        <div>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }}>ATTRIBUTION RANKING</div>
          <div style={{ color: '#00d7b2', fontSize: '0.8rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
            {candidates.length} CANDIDATE VESSEL{candidates.length !== 1 ? 'S' : ''}
          </div>
        </div>
        {dataMode === 'DEMO' && (
          <div style={{
            background: 'rgba(251,191,36,0.12)',
            border: '1px solid rgba(251,191,36,0.3)',
            borderRadius: '4px',
            padding: '3px 8px',
            fontSize: '0.62rem',
            color: '#fbbf24',
            fontFamily: 'var(--font-mono)',
          }}>
            DEMO DATA
          </div>
        )}
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              {['Rank', 'Vessel', 'MMSI', 'Min Dist', 'Traj Overlap', 'Time Match', 'Dist Score', 'Src Region', 'AIS Anomaly', 'Score'].map((h) => (
                <th key={h} style={{
                  padding: '8px 12px',
                  textAlign: 'left',
                  color: 'var(--text-muted)',
                  fontSize: '0.6rem',
                  fontFamily: 'var(--font-mono)',
                  textTransform: 'uppercase',
                  whiteSpace: 'nowrap',
                  fontWeight: 600,
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {candidates.map((c) => {
              const isActive = activeMMSI === c.mmsi;
              return (
                <tr
                  key={c.mmsi}
                  onClick={() => onSelectCandidate?.(c)}
                  style={{
                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                    cursor: 'pointer',
                    background: isActive ? 'rgba(0,215,178,0.06)' : 'transparent',
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(0,215,178,0.04)'; }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = isActive ? 'rgba(0,215,178,0.06)' : 'transparent'; }}
                >
                  <td style={{ padding: '8px 12px', color: c.rank === 1 ? '#fbbf24' : 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontWeight: c.rank === 1 ? 700 : 400 }}>
                    #{c.rank}
                  </td>
                  <td style={{ padding: '8px 12px', color: '#fff', fontWeight: 600 }}>
                    <div>{c.vesselName}</div>
                    {c.imo && <div style={{ color: 'var(--text-muted)', fontSize: '0.62rem' }}>IMO: {c.imo}</div>}
                  </td>
                  <td style={{ padding: '8px 12px', color: '#00d7b2', fontFamily: 'var(--font-mono)' }}>{c.mmsi}</td>
                  <td style={{ padding: '8px 12px', color: '#fff', fontFamily: 'var(--font-mono)' }}>
                    {c.minDistanceKm < 999 ? `${c.minDistanceKm.toFixed(1)} km` : '—'}
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <ScoreBar value={c.trajectoryOverlapScore} />
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <ScoreBar value={c.timeMatchScore} />
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <ScoreBar value={c.distanceScore ?? 0} />
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <ScoreBar value={c.sourceRegionScore ?? 0} />
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <ScoreBar value={c.aisAnomalyScore} />
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <div style={{
                      background: c.attributionScore >= 70 ? 'rgba(244,63,94,0.15)' : c.attributionScore >= 40 ? 'rgba(251,191,36,0.12)' : 'rgba(0,215,178,0.1)',
                      border: `1px solid ${c.attributionScore >= 70 ? 'rgba(244,63,94,0.4)' : c.attributionScore >= 40 ? 'rgba(251,191,36,0.3)' : 'rgba(0,215,178,0.3)'}`,
                      borderRadius: '3px',
                      padding: '2px 8px',
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 700,
                      fontSize: '0.78rem',
                      color: c.attributionScore >= 70 ? '#f43f5e' : c.attributionScore >= 40 ? '#fbbf24' : '#00d7b2',
                      textAlign: 'center',
                    }}>
                      {c.attributionScore.toFixed(0)}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Disclaimer */}
      <div style={{
        padding: '10px 16px',
        borderTop: '1px solid rgba(255,255,255,0.06)',
        display: 'flex',
        gap: '8px',
        background: 'rgba(251,191,36,0.03)',
      }}>
        <AlertTriangle size={12} style={{ color: '#fbbf24', flexShrink: 0, marginTop: '1px' }} />
        <div style={{ fontSize: '0.62rem', color: 'rgba(255,255,255,0.4)', fontFamily: 'var(--font-mono)', lineHeight: 1.5 }}>
          {disclaimer}
        </div>
      </div>
    </div>
  );
}
