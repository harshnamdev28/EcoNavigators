'use client';

import React, { useState } from 'react';
import { AlertTriangle, WifiOff, FlaskConical, Cpu } from 'lucide-react';
import InputPanel from '@/components/investigation/InputPanel';
import InvestigationMapCanvas from '@/components/investigation/InvestigationMapCanvas';
import CandidateTable from '@/components/investigation/CandidateTable';
import ImageAnalysisPanel from '@/components/investigation/ImageAnalysisPanel';
import { InvestigationResult, InvestigationCandidate } from '@/types/investigation';
import { runHistoricalInvestigation } from '@/services/api';

export default function HistoricalInvestigationPage() {
  const [result, setResult] = useState<InvestigationResult | null>(null);
  const [activeCandidate, setActiveCandidate] = useState<InvestigationCandidate | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Track the submitted spill coordinates for map centering before result arrives
  const [pendingLat, setPendingLat] = useState<number | null>(null);
  const [pendingLon, setPendingLon] = useState<number | null>(null);

  async function handleSubmit(params: {
    latitude: number;
    longitude: number;
    timestamp: string;
    durationHours: number;
    windage: number;
    nParticles: number;
    uncertaintyRadiusM?: number;
    dataMode?: 'DEMO' | 'LIVE';
    file?: File;
  }) {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setActiveCandidate(null);
    setPendingLat(params.latitude);
    setPendingLon(params.longitude);
    if (params.file) setUploadedFile(params.file);

    try {
      const res = await runHistoricalInvestigation(params);
      setResult(res);
      if (res.candidates.length > 0) {
        setActiveCandidate(res.candidates[0]);
      }
    } catch (err: any) {
      setError(err?.message || 'Investigation failed. Check that the backend is running.');
    } finally {
      setIsLoading(false);
    }
  }

  const spillLat = result?.spillObservation.latitude ?? pendingLat;
  const spillLon = result?.spillObservation.longitude ?? pendingLon;


  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', overflow: 'hidden' }}>

      {/* Top bar */}
      <div style={{
        padding: '10px 20px',
        borderBottom: '1px solid rgba(0,215,178,0.12)',
        background: 'rgba(11,23,35,0.97)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <div>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }}>
            HISTORICAL INVESTIGATION {'>'} <span style={{ color: '#00d7b2' }}>LAGRANGIAN PARTICLE BACKTRACKING</span>
          </div>
          <div style={{ color: '#fff', fontSize: '0.9rem', fontWeight: 700, marginTop: '2px' }}>
            Historical Oil Spill Investigation
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {result?.dataMode === 'DEMO' && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              background: 'rgba(251,191,36,0.1)',
              border: '1px solid rgba(251,191,36,0.3)',
              borderRadius: '4px',
              padding: '4px 10px',
              fontSize: '0.7rem',
              color: '#fbbf24',
              fontFamily: 'var(--font-mono)',
            }}>
              <FlaskConical size={12} />
              DEMO MODE — Synthetic environmental data
            </div>
          )}
          <div style={{
            background: 'rgba(0,215,178,0.08)',
            border: '1px solid rgba(0,215,178,0.2)',
            borderRadius: '4px',
            padding: '4px 10px',
            fontSize: '0.68rem',
            color: '#00d7b2',
            fontFamily: 'var(--font-mono)',
          }}>
            V_oil = V_current + W × V_wind + V_diffusion
          </div>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div style={{
          background: 'rgba(244,63,94,0.1)',
          borderBottom: '1px solid rgba(244,63,94,0.3)',
          color: '#f43f5e',
          padding: '6px 20px',
          fontSize: '0.72rem',
          fontFamily: 'var(--font-mono)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          flexShrink: 0,
        }}>
          <WifiOff size={13} />
          {error}
        </div>
      )}

      {/* Main layout: Input panel + Map */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left: Input panel + image analysis */}
        <div style={{ display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
          <InputPanel onSubmit={handleSubmit} isLoading={isLoading} />
          {/* MIT-B2 segmentation result panel */}
          {(result?.imageAnalysis || (isLoading && uploadedFile)) && (
            <div style={{ padding: '0 16px 16px' }}>
              <ImageAnalysisPanel
                imageAnalysis={result?.imageAnalysis ?? null}
                uploadedFile={uploadedFile}
              />
            </div>
          )}
        </div>

        {/* Right: Map + results */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* Map */}
          <div style={{ flex: 1, position: 'relative', minHeight: '300px' }}>
            <InvestigationMapCanvas
              result={result}
              spillLat={spillLat}
              spillLon={spillLon}
              activeCandidate={activeCandidate}
            />

            {/* Legend overlay */}
            {result && (
              <div style={{
                position: 'absolute',
                bottom: '16px',
                left: '16px',
                zIndex: 1000,
                background: 'rgba(11,23,35,0.92)',
                border: '1px solid rgba(0,215,178,0.2)',
                borderRadius: '6px',
                padding: '10px 14px',
                fontSize: '0.67rem',
                fontFamily: 'var(--font-mono)',
                lineHeight: 1.8,
              }}>
                <div style={{ color: 'var(--text-muted)', marginBottom: '4px', fontSize: '0.58rem' }}>MAP LEGEND</div>
                {[
                  { color: '#f43f5e', label: '● Observed Spill Location' },
                  { color: '#f97316', label: '▬ Backward Trajectory Corridor' },
                  { color: '#c026d3', label: '◆ Candidate Source Region' },
                  { color: '#fbbf24', label: '# Candidate Vessel (Rank 1)' },
                  { color: '#00d7b2', label: '# Candidate Vessel (Other)' },
                ].map(({ color, label }) => (
                  <div key={label} style={{ color, display: 'flex', gap: '6px' }}>
                    <span>{label}</span>
                  </div>
                ))}
                {result.sourceRegion.windageConsistency && (
                  <div style={{
                    marginTop: '6px',
                    padding: '4px 6px',
                    background: 'rgba(0,215,178,0.08)',
                    borderRadius: '3px',
                    color: '#00d7b2',
                    fontSize: '0.58rem',
                    maxWidth: '220px',
                    lineHeight: 1.5,
                  }}>
                    {result.sourceRegion.windageConsistency.startsWith('Trajectory corridor consistent')
                      ? '✓ Windage-consistent trajectory'
                      : '⚠ Windage divergence — higher uncertainty'}
                  </div>
                )}
              </div>
            )}

            {/* Loading overlay */}
            {isLoading && (
              <div style={{
                position: 'absolute',
                inset: 0,
                background: 'rgba(11,23,35,0.75)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 1001,
                fontFamily: 'var(--font-mono)',
                gap: '12px',
              }}>
                <div style={{ color: '#00d7b2', fontSize: '0.9rem', fontWeight: 700 }}>
                  ⟳ RUNNING LAGRANGIAN SIMULATION
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>
                  Integrating {pendingLat?.toFixed(4)}°N, {pendingLon?.toFixed(4)}°E backward in time...
                </div>
                <div style={{ color: 'rgba(0,215,178,0.5)', fontSize: '0.62rem' }}>
                  V_oil = V_current + W × V_wind + V_diffusion
                </div>
              </div>
            )}

            {/* Empty state */}
            {!result && !isLoading && (
              <div style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                fontFamily: 'var(--font-mono)',
                gap: '10px',
                pointerEvents: 'none',
              }}>
                <div style={{ fontSize: '2rem', opacity: 0.15 }}>🛢️</div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                  Enter spill location and timestamp, then click Run
                </div>
                <div style={{ color: 'rgba(0,215,178,0.3)', fontSize: '0.62rem' }}>
                  Lagrangian particle backtracking · 500 particles · 15-min timestep
                </div>
              </div>
            )}
          </div>

          {/* Results panel */}
          {result && (
            <div style={{
              flexShrink: 0,
              maxHeight: '260px',
              overflowY: 'auto',
              padding: '12px 16px',
              background: 'rgba(8,18,28,0.98)',
              borderTop: '1px solid rgba(0,215,178,0.12)',
            }}>
              {/* Investigation summary strip */}
              <div style={{
                display: 'flex',
                gap: '20px',
                marginBottom: '10px',
                flexWrap: 'wrap',
              }}>
                {[
                  { label: 'INVESTIGATION ID', value: result.investigationId.slice(0, 8).toUpperCase() },
                  { label: 'PARTICLES', value: result.model.particles },
                  { label: 'DURATION', value: `${result.model.durationHours}h` },
                  { label: 'TIMESTEP', value: `${result.model.timestepMinutes} min` },
                  { label: 'TRAJECTORY STEPS', value: result.trajectory.timesteps.length },
                  { label: 'CANDIDATE VESSELS', value: result.candidates.length },
                  { label: 'ENV DATA', value: result.environment.dataMode },
                  { label: 'ENV PROVIDER', value: result.environment.provider },
                  { label: 'GEOMETRY SOURCE', value: result.spillObservation.geometrySource ?? 'POINT_ONLY' },
                ].map(({ label, value }) => (
                  <div key={label}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.58rem', fontFamily: 'var(--font-mono)' }}>{label}</div>
                    <div style={{ color: '#00d7b2', fontSize: '0.82rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{value}</div>
                  </div>
                ))}
              </div>

              <CandidateTable
                candidates={result.candidates}
                onSelectCandidate={setActiveCandidate}
                activeMMSI={activeCandidate?.mmsi}
                dataMode={result.dataMode}
                disclaimer={result.disclaimer}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
