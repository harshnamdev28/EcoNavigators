'use client';
import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Ship,
  Droplets,
  Scan,
  History,
  Bell,
  AlertTriangle,
  Activity,
  HelpCircle,
  Search,
} from 'lucide-react';

const NAV_ITEMS = [
  { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
  { name: 'Vessel Tracking', path: '/vessels', icon: Ship },
  { name: 'Oil Spill Detection', path: '/spills', icon: Droplets },
  { name: 'SAR Oil Spill Detection', path: '/sar-detection', icon: Scan },
  { name: 'Backtracking Analysis', path: '/backtracking', icon: History },
  { name: 'Historical Investigation', path: '/historical-investigation', icon: Search },
  { name: 'Alerts', path: '/alerts', icon: Bell },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="dashboard-sidebar">
      {/* Brand / Logo */}
      <div className="sidebar-header">
        <Link href="/" className="sidebar-logo">
          <div className="logo-glow-ring">
            <div className="logo-glow-dot" />
          </div>
          <div>
            <div className="sidebar-title">EcoNavigators</div>
            <div className="sidebar-subtitle">MARINE SURVEILLANCE</div>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <ul className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.path;
          const Icon = item.icon;
          return (
            <li key={item.path} className="sidebar-nav-item">
              <Link
                href={item.path}
                className={`sidebar-link ${isActive ? 'active' : ''}`}
              >
                <Icon size={18} />
                <span>{item.name}</span>
              </Link>
            </li>
          );
        })}
      </ul>

      {/* Footer / Report Spill CTA */}
      <div className="sidebar-footer">
        <button
          className="btn-report-spill"
          onClick={() => alert('Report Incident modal initiated.')}
        >
          <AlertTriangle size={15} />
          <span>Report Oil Spill</span>
        </button>

        <div className="sidebar-footer-links">
          <button className="sidebar-sub-btn">
            <Activity size={15} />
            <span>System Health</span>
          </button>
          <button className="sidebar-sub-btn">
            <HelpCircle size={15} />
            <span>Help</span>
          </button>
        </div>
      </div>
    </aside>
  );
}
