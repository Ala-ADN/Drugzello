# Drugzello demo

The React/Vite frontend uses `src/components/EditorComponent.jsx` for molecular
drawing and calls the trained aqueous-solubility API. `App.jsx` is its entry wrapper.
There is one maintained screen; model results, learned masks and MC dropout spread
come from the API. MolT5 generation is not supported.

```bash
npm ci
npm run dev
```

Start the backend separately using the [root instructions](../README.md). Vite
proxies local API calls; set `VITE_API_URL` for a separate API host.

```bash
npm run lint
npm run build
```

The molecular editor brings substantial browser dependencies. `.npmrc` retains
legacy peer resolution for the existing Ketcher dependency tree. `node_modules/`
and `dist/` are generated and ignored; reinstall with `npm ci` after cleanup.
