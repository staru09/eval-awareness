import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { readdirSync, statSync, existsSync, createReadStream } from "node:fs";
import { join } from "node:path";

const API_DIR = join(__dirname, "..", "output", "api");

// Serves ../output/api/* live, so the dashboard always reflects what's on
// disk (including a model finishing mid-session) instead of a frozen
// bundled snapshot from `npm run aggregate`.
function apiDataPlugin(): Plugin {
  return {
    name: "eval-awareness-api-data",
    configureServer(server) {
      server.middlewares.use("/api-data", (req, res, next) => {
        const url = new URL(req.url ?? "/", "http://localhost");
        const parts = url.pathname.split("/").filter(Boolean);

        if (parts.length === 0) {
          // GET /api-data/ -> list of model dirs that have reasoning_traces.json
          const models = readdirSync(API_DIR).filter((name) => {
            const dir = join(API_DIR, name);
            return statSync(dir).isDirectory() && existsSync(join(dir, "reasoning_traces.json"));
          });
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(models));
          return;
        }

        const [modelDir, ...rest] = parts;
        const relFile = rest.join("/");
        const allowed = new Set([
          "reasoning_traces.json",
          "verbalized_responses.csv",
          "results/verbalized_auroc.json",
        ]);
        if (!allowed.has(relFile)) {
          next();
          return;
        }
        const filePath = join(API_DIR, modelDir, relFile);
        if (!existsSync(filePath)) {
          res.statusCode = 404;
          res.end("not found");
          return;
        }
        res.setHeader(
          "Content-Type",
          relFile.endsWith(".csv") ? "text/csv" : "application/json"
        );
        createReadStream(filePath).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), apiDataPlugin()],
});
