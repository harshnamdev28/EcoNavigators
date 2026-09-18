/** @type {import('next').NextConfig} */
const nextConfig = {
  // Note: do NOT set output:'standalone' — Vercel deploys Next.js natively
  images: {
    unoptimized: true,
  },
};
module.exports = nextConfig;
