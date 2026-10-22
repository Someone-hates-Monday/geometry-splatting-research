const fs = require('fs');
const path = require('path');
const compilerPath = 'C:/Users/Someone/AppData/Local/Programs/Microsoft VS Code/110a328ea5/resources/app/extensions/node_modules/typescript/lib/typescript.js';
const ts = require(compilerPath);
const result = ts.transpileModule(fs.readFileSync(path.join(__dirname, 'canvas_history_gap_pilot_stage_v1.tsx'), 'utf8'), {
  fileName: 'GGGS-research-audit.canvas.tsx', reportDiagnostics: true,
  compilerOptions: { jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ESNext }
});
const diagnostics = (result.diagnostics || []).map(d => ({ code: d.code, message: ts.flattenDiagnosticMessageText(d.messageText, '\n'), category: d.category }));
const errors = diagnostics.filter(d => d.category === ts.DiagnosticCategory.Error);
const output = { checked_utc: new Date().toISOString(), errors, diagnostics, compilerPath, compilerVersion: ts.version, scope: 'TypeScript transpile syntax only; no SDK typing or app rendering proof.' };
fs.writeFileSync(path.join(__dirname, 'results/canvas_history_gap_pilot_syntax_v1.json'), JSON.stringify(output, null, 2)+'\n');
console.log(JSON.stringify(output));
if(errors.length) process.exitCode = 1;
