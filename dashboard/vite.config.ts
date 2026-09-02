import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static site (deployed to GitHub Pages) -- dashboard data is bundled at
// build time from ../output/api/* into src/data/dashboard_data.json via
// `npm run aggregate` (scripts/aggregate.mjs), not fetched live.
export default defineConfig({
  base: "/eval-awareness/",
  plugins: [react()],
});
