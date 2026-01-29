/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Allow dev server requests from localhost/127.0.0.1 (silences cross-origin warning)
  allowedDevOrigins: ['http://localhost:3000', 'http://127.0.0.1:3000'],
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': require('path').resolve(__dirname),
    }
    return config
  },
}

module.exports = nextConfig
