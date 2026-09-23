// Builds the dashboard as a framework-free static page: standalone/index.html + app.js + replay/bundle.js.
// Useful for hosts that serve a single page (previews, S3, GitHub Pages) — Vercel can use `next build` instead.
import { build } from "esbuild";
import { execSync } from "node:child_process";
import { copyFileSync, mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";

const out = "standalone";
mkdirSync(`${out}/replay`, { recursive: true });

await build({
  entryPoints: ["artifact/entry.tsx"],
  bundle: true,
  minify: true,
  format: "iife",
  target: "es2020",
  outfile: `${out}/app.js`,
  jsx: "automatic",
  define: { "process.env.NODE_ENV": '"production"', "process.env.NEXT_PUBLIC_EPOCH_API": '""' },
  loader: { ".css": "css" },
  logLevel: "warning",
  tsconfig: "tsconfig.json",
});

execSync(`npx @tailwindcss/cli -i app/globals.css -o ${out}/tw.css --minify`, { stdio: "inherit" });
const css = readFileSync(`${out}/tw.css`, "utf8") + (existsSync(`${out}/app.css`) ? readFileSync(`${out}/app.css`, "utf8") : "");

const page = `<title>EPOCH Mission Control</title>
<meta name="description" content="Autonomous AI engineering & evolution platform — Pareto search, digital twin and self-healing for AI pipelines.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap">
<style>${css}</style>
<div id="root"></div>
<script src="app.js"></script>
`;
writeFileSync(`${out}/index.html`, page);
for (const f of ["bundle.js", "bundle.json"]) if (existsSync(`public/replay/${f}`)) copyFileSync(`public/replay/${f}`, `${out}/replay/${f}`);
console.log(`standalone build → ${out}/`);
