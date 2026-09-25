/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Allow a remote-backed UI alongside the normal local development server.
  distDir: process.env.NEXT_DIST_DIR || '.next',
  // Hostnames, not origins: Next matches these against the request host, so
  // 'http://127.0.0.1:3000' never matches and the page loads without its JS.
  allowedDevOrigins: ['localhost', '127.0.0.1'],
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': require('path').resolve(__dirname),
    }
    return config
  },
}

module.exports = nextConfig
