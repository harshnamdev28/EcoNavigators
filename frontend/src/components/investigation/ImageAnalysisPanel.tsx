'use client';

import React, { useRef, useEffect } from 'react';
import { AlertTriangle, CheckCircle2, XCircle, Info, Cpu } from 'lucide-react';

interface ImageAnalysisData {
  model: string;
  detected: boolean;
  geometrySource: 'IMAGE_DERIVED' | 'POINT_ONLY';
  spillPixelCount?: number;
  totalPixels?: number;
  spillFraction?: number;
  spillAreaKm2?: number | null;
  centroidRel?: [number, number] | null;
  boundingBoxRel?: [number, number, number, number] | null;
  segmentationConfidence?: number | null;
  maskPngBase64?: string | null;
  reason?: string;
  error?: string | null;
}

interface ImageAnalysisPanelProps {
  /** Raw analysis result from API */
  imageAnalysis: ImageAnalysisData | null | undefined;
  /** The File object that was uploaded (for preview) */
  uploadedFile?: File | null;
}

/**
 * Displays the MIT-B2 U-Net segmentation result for an uploaded SAR/optical image.
 *
 * Shows:
 *  - Detected / Not Detected status badge
 *  - Original image preview with semi-transparent orange mask overlay
 *  - Spill area, pixel count, confidence
 *  - Geometry source label (IMAGE_DERIVED / POINT_ONLY)
 */
export default function ImageAnalysisPanel({ imageAnalysis, uploadedFile }: ImageAnalysisPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef    = useRef<HTMLImageElement>(null);

  // Draw original image + mask overlay onto canvas once both are available
  useEffect(() => {
    if (!imageAnalysis?.maskPngBase64 || !uploadedFile || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx    = canvas.getContext('2d');
    if (!ctx) return;

    const originalImg = new Image();
    const maskImg     = new Image();

    let origLoaded = false;
    let maskLoaded = false;

    const draw = () => {
      if (!origLoaded || !maskLoaded) return;
      // Size canvas to original image
      canvas.width  = originalImg.naturalWidth;
      canvas.height = originalImg.naturalHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      // Draw original image
      ctx.drawImage(originalImg, 0, 0, canvas.width, canvas.height);
      // Draw mask on top (mask PNG has alpha=180 for spill pixels, 0 for background)
      ctx.globalAlpha = 1.0;
      ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
    };

    const objUrl = URL.createObjectURL(uploadedFile);
    originalImg.onload = () => { origLoaded = true; draw(); };
    originalImg.src    = objUrl;

    maskImg.onload = () => { maskLoaded = true; draw(); };
    maskImg.src    = `data:image/png;base64,${imageAnalysis.maskPngBase64}`;

    return () => { URL.revokeObjectURL(objUrl); };
  }, [imageAnalysis?.maskPngBase64, uploadedFile]);

  if (!imageAnalysis) return null;

  const { detected, model, geometrySource, spillPixelCount, totalPixels,
          spillFraction, spillAreaKm2, segmentationConfidence, reason, error } = imageAnalysis;

  // ── Error state ──────────────────────────────────────────────────────────
  if (error && !detected) {
    return (
      <div className="bg-gray-900/80 border border-yellow-600/40 rounded-lg p-4 mt-4">
        <div className="flex items-center gap-2 mb-2">
          <AlertTriangle size={16} className="text-yellow-400" />
          <span className="text-yellow-300 text-sm font-semibold">Image Analysis — Warning</span>
        </div>
        <p className="text-xs text-yellow-200/70">
          MIT-B2 U-Net encountered an error. Lagrangian simulation used supplied lat/lon (POINT_ONLY).
        </p>
        <p className="text-xs text-red-400 mt-1 font-mono">{error}</p>
      </div>
    );
  }

  // ── Not detected ─────────────────────────────────────────────────────────
  const detectionBadge = detected ? (
    <div className="flex items-center gap-2 px-3 py-1 bg-green-900/40 border border-green-500/40 rounded-full">
      <CheckCircle2 size={13} className="text-green-400" />
      <span className="text-green-300 text-xs font-semibold">Spill Detected</span>
    </div>
  ) : (
    <div className="flex items-center gap-2 px-3 py-1 bg-gray-800/60 border border-gray-600/40 rounded-full">
      <XCircle size={13} className="text-gray-400" />
      <span className="text-gray-300 text-xs font-semibold">No Spill Detected</span>
    </div>
  );

  const geometryBadge = (
    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono ${
      geometrySource === 'IMAGE_DERIVED'
        ? 'bg-cyan-900/30 border border-cyan-500/30 text-cyan-300'
        : 'bg-gray-800/40 border border-gray-600/30 text-gray-400'
    }`}>
      {geometrySource === 'IMAGE_DERIVED' ? '⬡ IMAGE_DERIVED' : '· POINT_ONLY'}
    </div>
  );

  return (
    <div className="bg-gray-900/80 border border-blue-900/40 rounded-lg p-4 mt-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Cpu size={15} className="text-blue-400" />
          <span className="text-blue-300 text-sm font-semibold">{model}</span>
        </div>
        <div className="flex items-center gap-2">
          {detectionBadge}
          {geometryBadge}
        </div>
      </div>

      {/* Canvas overlay */}
      {uploadedFile && imageAnalysis.maskPngBase64 && (
        <div className="relative w-full overflow-hidden rounded border border-gray-700">
          <canvas
            ref={canvasRef}
            className="w-full h-auto block"
            style={{ maxHeight: 320, objectFit: 'contain' }}
          />
          <div className="absolute bottom-2 left-2 bg-black/70 text-xs text-orange-300 px-2 py-1 rounded">
            <span className="inline-block w-3 h-3 rounded-sm mr-1 align-middle" style={{ background: 'rgba(255,140,0,0.7)' }} />
            Segmented spill region
          </div>
        </div>
      )}

      {/* Show uploaded image preview without mask if no mask available */}
      {uploadedFile && !imageAnalysis.maskPngBase64 && (
        <div className="w-full rounded border border-gray-700 overflow-hidden">
          <img
            ref={imgRef}
            src={URL.createObjectURL(uploadedFile)}
            alt="Uploaded satellite image"
            className="w-full h-auto block"
            style={{ maxHeight: 320, objectFit: 'contain' }}
          />
        </div>
      )}

      {/* Stats grid */}
      {detected && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {spillAreaKm2 != null && (
            <StatCard label="Spill Area" value={`${spillAreaKm2.toFixed(2)} km²`} />
          )}
          {spillPixelCount !== undefined && totalPixels !== undefined && (
            <StatCard label="Spill Pixels" value={`${spillPixelCount.toLocaleString()} / ${totalPixels.toLocaleString()}`} />
          )}
          {spillFraction !== undefined && (
            <StatCard label="Coverage" value={`${(spillFraction * 100).toFixed(2)}%`} />
          )}
          {segmentationConfidence != null && (
            <StatCard label="Confidence" value={`${(segmentationConfidence * 100).toFixed(1)}%`} />
          )}
        </div>
      )}

      {/* Reason / info */}
      {reason && !detected && (
        <div className="flex items-start gap-2 text-xs text-gray-400">
          <Info size={12} className="mt-0.5 shrink-0 text-gray-500" />
          <span>{reason}</span>
        </div>
      )}

      {/* Disclaimer */}
      <p className="text-[10px] text-gray-500 border-t border-gray-800 pt-2">
        Segmentation by {model}. Spill area is an approximation from pixel counts.
        Results should be verified with domain expert analysis.
        Geometry source: <span className="text-gray-400 font-mono">{geometrySource}</span>.
      </p>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-800/50 rounded p-2 text-center">
      <p className="text-[10px] text-gray-400 uppercase tracking-wider">{label}</p>
      <p className="text-xs text-white font-semibold mt-0.5">{value}</p>
    </div>
  );
}
