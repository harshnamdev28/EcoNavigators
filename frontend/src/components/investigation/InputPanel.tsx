'use client';
import React, { useRef, useState } from 'react';
import { Search, Upload, Clock, MapPin, Wind, Layers, X, Shield, FlaskConical } from 'lucide-react';

interface InputPanelProps {
  onSubmit: (data: {
    latitude: number;
    longitude: number;
    timestamp: string;
    durationHours: number;
    windage: number;
    nParticles: number;
    uncertaintyRadiusM: number;
    dataMode: 'DEMO' | 'LIVE';
    file?: File;
  }) => void;
  isLoading: boolean;
}

export default function InputPanel({ onSubmit, isLoading }: InputPanelProps) {
  const [lat, setLat] = useState('');
  const [lon, setLon] = useState('');
  const [date, setDate] = useState('');
  const [time, setTime] = useState('');
  const [durationHours, setDurationHours] = useState(24);
  const [windage, setWindage] = useState(0.03);
  const [nParticles, setNParticles] = useState(500);
  const [uncertaintyRadiusM, setUncertaintyRadiusM] = useState(5000);
  const [dataMode, setDataMode] = useState<'DEMO' | 'LIVE'>('DEMO');
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const fileInputRef = useRef<HTMLInputElement>(null);

  function validate(): boolean {
    const errs: Record<string, string> = {};
    const latN = parseFloat(lat);
    const lonN = parseFloat(lon);
    if (!lat.trim() || isNaN(latN) || latN < -90 || latN > 90)
      errs.lat = 'Latitude must be -90 to 90';
    if (!lon.trim() || isNaN(lonN) || lonN < -180 || lonN > 180)
      errs.lon = 'Longitude must be -180 to 180';
    if (!date) errs.date = 'Date is required';
    if (!time) errs.time = 'Time is required';
    setErrors(errs);
    return Object.keys(errs).length === 0;
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!validate()) return;
    const timestamp = `${date}T${time}:00Z`;
    onSubmit({
      latitude: parseFloat(lat),
      longitude: parseFloat(lon),
      timestamp,
      durationHours,
      windage,
      nParticles,
      uncertaintyRadiusM,
      dataMode,
      file: file || undefined,
    });
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }

  const inputStyle: React.CSSProperties = {
    background: 'rgba(0,0,0,0.4)',
    border: '1px solid rgba(0,215,178,0.3)',
    color: '#fff',
    borderRadius: '4px',
    padding: '6px 10px',
    fontSize: '0.82rem',
    fontFamily: 'var(--font-mono)',
    width: '100%',
    boxSizing: 'border-box',
  };

  const labelStyle: React.CSSProperties = {
    color: 'var(--text-muted)',
    fontSize: '0.65rem',
    fontFamily: 'var(--font-mono)',
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
    marginBottom: '3px',
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  };

  const fieldStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    marginBottom: '10px',
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        width: '300px',
        minWidth: '280px',
        background: 'rgba(11,23,35,0.97)',
        borderRight: '1px solid rgba(0,215,178,0.15)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div style={{
        padding: '14px 16px 10px',
        borderBottom: '1px solid rgba(0,215,178,0.12)',
        background: 'rgba(0,215,178,0.04)',
      }}>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', marginBottom: '2px' }}>
          NEW INVESTIGATION
        </div>
        <div style={{ color: '#00d7b2', fontSize: '0.85rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
          LAGRANGIAN BACKTRACKING
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.65rem', marginTop: '2px' }}>
          Physics-based particle trajectory model
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '14px 16px' }}>

        {/* Image Upload */}
        <div style={fieldStyle}>
          <div style={labelStyle}><Upload size={11} /> Oil Spill Image (Optional)</div>
          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `1px dashed ${dragOver ? '#00d7b2' : 'rgba(0,215,178,0.25)'}`,
              borderRadius: '4px',
              padding: '10px',
              textAlign: 'center',
              cursor: 'pointer',
              background: dragOver ? 'rgba(0,215,178,0.06)' : 'rgba(0,0,0,0.2)',
              fontSize: '0.72rem',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
              transition: 'all 0.2s',
            }}
          >
            {file ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                <span style={{ color: '#00d7b2' }}>{file.name}</span>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setFile(null); }}
                  style={{ background: 'none', border: 'none', color: '#f43f5e', cursor: 'pointer', padding: 0 }}
                >
                  <X size={12} />
                </button>
              </div>
            ) : (
              <>Drop image or click to browse<br /><span style={{ fontSize: '0.6rem' }}>PNG, JPG, TIF up to 25MB</span></>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff"
            style={{ display: 'none' }}
            onChange={(e) => { if (e.target.files?.[0]) setFile(e.target.files[0]); }}
          />
        </div>

        {/* Location */}
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '10px', marginBottom: '4px' }}>
          <div style={{ ...labelStyle, color: '#00d7b2', marginBottom: '8px' }}><MapPin size={11} /> Spill Location</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <div style={fieldStyle}>
            <div style={labelStyle}>Latitude *</div>
            <input
              style={{ ...inputStyle, borderColor: errors.lat ? '#f43f5e' : undefined }}
              value={lat}
              onChange={(e) => setLat(e.target.value)}
              placeholder="e.g. 25.7715"
            />
            {errors.lat && <div style={{ color: '#f43f5e', fontSize: '0.6rem', marginTop: '2px' }}>{errors.lat}</div>}
          </div>
          <div style={fieldStyle}>
            <div style={labelStyle}>Longitude *</div>
            <input
              style={{ ...inputStyle, borderColor: errors.lon ? '#f43f5e' : undefined }}
              value={lon}
              onChange={(e) => setLon(e.target.value)}
              placeholder="e.g. -80.1518"
            />
            {errors.lon && <div style={{ color: '#f43f5e', fontSize: '0.6rem', marginTop: '2px' }}>{errors.lon}</div>}
          </div>
        </div>

        {/* Timestamp */}
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '10px', marginBottom: '4px' }}>
          <div style={{ ...labelStyle, color: '#00d7b2', marginBottom: '8px' }}><Clock size={11} /> Spill Timestamp (UTC)</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <div style={fieldStyle}>
            <div style={labelStyle}>Date *</div>
            <input
              type="date"
              style={{ ...inputStyle, borderColor: errors.date ? '#f43f5e' : undefined, colorScheme: 'dark' }}
              value={date}
              onChange={(e) => setDate(e.target.value)}
            />
          </div>
          <div style={fieldStyle}>
            <div style={labelStyle}>Time (UTC) *</div>
            <input
              type="time"
              style={{ ...inputStyle, borderColor: errors.time ? '#f43f5e' : undefined, colorScheme: 'dark' }}
              value={time}
              onChange={(e) => setTime(e.target.value)}
            />
          </div>
        </div>

        {/* Model Parameters */}
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '10px', marginBottom: '4px' }}>
          <div style={{ ...labelStyle, color: '#00d7b2', marginBottom: '8px' }}><Layers size={11} /> Model Parameters</div>
        </div>

        <div style={fieldStyle}>
          <div style={{ ...labelStyle, justifyContent: 'space-between' }}>
            <span>Duration (hours)</span>
            <span style={{ color: '#00d7b2' }}>{durationHours}h</span>
          </div>
          <input
            type="range" min={6} max={48} step={6}
            value={durationHours}
            onChange={(e) => setDurationHours(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#00d7b2' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            <span>6h</span><span>48h</span>
          </div>
        </div>

        <div style={fieldStyle}>
          <div style={{ ...labelStyle, justifyContent: 'space-between' }}>
            <span><Wind size={10} style={{ display: 'inline' }} /> Windage coefficient (W)</span>
            <span style={{ color: '#00d7b2' }}>{(windage * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range" min={1} max={4} step={1}
            value={Math.round(windage * 100)}
            onChange={(e) => setWindage(Number(e.target.value) / 100)}
            style={{ width: '100%', accentColor: '#00d7b2' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            <span>W=1% (low)</span><span>W=4% (high)</span>
          </div>
        </div>

        <div style={fieldStyle}>
          <div style={{ ...labelStyle, justifyContent: 'space-between' }}>
            <span>Particles</span>
            <span style={{ color: '#00d7b2' }}>{nParticles}</span>
          </div>
          <input
            type="range" min={100} max={1000} step={100}
            value={nParticles}
            onChange={(e) => setNParticles(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#00d7b2' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            <span>100</span><span>1000</span>
          </div>
        </div>

        {/* Uncertainty radius */}
        <div style={fieldStyle}>
          <div style={{ ...labelStyle, justifyContent: 'space-between' }}>
            <span>Uncertainty radius</span>
            <span style={{ color: '#00d7b2' }}>
              {uncertaintyRadiusM >= 1000
                ? `${(uncertaintyRadiusM / 1000).toFixed(0)} km`
                : `${uncertaintyRadiusM} m`}
            </span>
          </div>
          <input
            type="range" min={500} max={50000} step={500}
            value={uncertaintyRadiusM}
            onChange={(e) => setUncertaintyRadiusM(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#00d7b2' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            <span>500 m</span><span>50 km</span>
          </div>
        </div>

        {/* Data mode */}
        <div style={fieldStyle}>
          <div style={labelStyle}><Shield size={10} /> Environmental Data Mode</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
            {(['DEMO', 'LIVE'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setDataMode(mode)}
                style={{
                  padding: '6px',
                  borderRadius: '4px',
                  border: `1px solid ${dataMode === mode
                    ? (mode === 'DEMO' ? 'rgba(251,191,36,0.6)' : 'rgba(0,215,178,0.6)')
                    : 'rgba(255,255,255,0.1)'}`,
                  background: dataMode === mode
                    ? (mode === 'DEMO' ? 'rgba(251,191,36,0.1)' : 'rgba(0,215,178,0.1)')
                    : 'rgba(0,0,0,0.2)',
                  color: dataMode === mode
                    ? (mode === 'DEMO' ? '#fbbf24' : '#00d7b2')
                    : 'var(--text-muted)',
                  fontSize: '0.68rem',
                  fontFamily: 'var(--font-mono)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '4px',
                }}
              >
                {mode === 'DEMO' ? <FlaskConical size={10} /> : <Shield size={10} />}
                {mode}
              </button>
            ))}
          </div>
          <div style={{ fontSize: '0.58rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            {dataMode === 'DEMO'
              ? 'DEMO: synthetic environmental data — clearly labeled'
              : 'LIVE: real NOAA ERDDAP HYCOM data — requires internet'}
          </div>
        </div>

        {/* Windage ensemble note */}
        <div style={{
          background: 'rgba(251,191,36,0.06)',
          border: '1px solid rgba(251,191,36,0.2)',
          borderRadius: '4px',
          padding: '8px',
          fontSize: '0.62rem',
          color: 'rgba(251,191,36,0.8)',
          fontFamily: 'var(--font-mono)',
          marginBottom: '10px',
        }}>
          Ensemble runs W={'{'}1%, {Math.round(windage * 100)}%, {Math.min(4, Math.round(windage * 100) + 1)}%{'}'} to quantify windage uncertainty.
        </div>
      </div>

      {/* Submit */}
      <div style={{ padding: '12px 16px', borderTop: '1px solid rgba(0,215,178,0.12)' }}>
        <button
          type="submit"
          disabled={isLoading}
          className="btn-primary-cyan"
          style={{
            width: '100%',
            height: '40px',
            fontSize: '0.8rem',
            opacity: isLoading ? 0.6 : 1,
            cursor: isLoading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
          }}
        >
          <Search size={15} />
          {isLoading ? '⟳ RUNNING SIMULATION...' : 'RUN HISTORICAL INVESTIGATION'}
        </button>
        <div style={{ color: 'rgba(255,255,255,0.3)', fontSize: '0.58rem', textAlign: 'center', marginTop: '6px', fontFamily: 'var(--font-mono)' }}>
          15-min timestep · Physics-based · Not AI
        </div>
      </div>
    </form>
  );
}
