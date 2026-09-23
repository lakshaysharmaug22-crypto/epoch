/** Static export: the dashboard is a client app that reads either the replay bundle or the live API. */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  reactStrictMode: true,
  // EPOCH_RELATIVE=1 builds with relative asset URLs so `out/` can be hosted from any sub-path
  assetPrefix: process.env.EPOCH_RELATIVE ? "." : undefined,
};

export default nextConfig;
