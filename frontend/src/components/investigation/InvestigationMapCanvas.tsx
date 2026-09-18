'use client';
/**
 * SSR-safe wrapper for InvestigationMap.
 * Leaflet requires a DOM environment, so we use Next.js dynamic() with ssr:false.
 * Pattern mirrors existing BacktrackMapCanvas.tsx.
 */
import dynamic from 'next/dynamic';
import React from 'react';
import { InvestigationResult, InvestigationCandidate } from '@/types/investigation';

const InvestigationMap = dynamic(
  () => import('./InvestigationMap'),
  {
    ssr: false,
    loading: () => (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#0b1723',
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.8rem',
        }}
      >
        ⟳ INITIALISING LAGRANGIAN MAP...
      </div>
    ),
  }
);

interface InvestigationMapCanvasProps {
  result: InvestigationResult | null;
  spillLat: number | null;
  spillLon: number | null;
  activeCandidate: InvestigationCandidate | null;
}

export default function InvestigationMapCanvas(props: InvestigationMapCanvasProps) {
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <InvestigationMap {...props} />
    </div>
  );
}
