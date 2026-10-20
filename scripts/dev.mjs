/** Start the existing Python pipeline API and Vite together. */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { createServer } from 'vite';

const projectRoot = fileURLToPath(new URL('../', import.meta.url));
const localPythons = [
  resolve(projectRoot, '.venv/Scripts/python.exe'),
  resolve(projectRoot, '.venv/bin/python'),
];
const python = localPythons.find(existsSync) || process.env.PYTHON || 'python';
const dependencyCheck = spawnSync(
  python,
  ['-c', 'import requests, nltk, sklearn, numpy, joblib, pyarrow, threadpoolctl'],
  { cwd: projectRoot, encoding: 'utf8', windowsHide: true },
);

if (dependencyCheck.error || dependencyCheck.status !== 0) {
  console.error('The Python backend could not load its dependencies.');
  console.error(dependencyCheck.error?.message || dependencyCheck.stderr.trim());
  console.error(`\nUsing Python: ${python}`);
  console.error('Create a Python environment and install the project dependencies:');
  console.error('  python -m venv .venv');
  console.error('  .venv\\Scripts\\python.exe -m pip install -r requirements.txt   (Windows)');
  console.error('  .venv/bin/python -m pip install -r requirements.txt          (macOS/Linux)');
  console.error('Then install the POS tagger with the same Python:');
  console.error('  python -m nltk.downloader averaged_perceptron_tagger_eng');
  process.exit(1);
}

let stopping = false;
let vite;
const backend = spawn(python, ['-u', 'src/web_api.py'], {
  cwd: projectRoot,
  stdio: 'inherit',
  windowsHide: true,
  detached: process.platform !== 'win32',
  env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONUTF8: '1' },
});

async function stop(exitCode = 0) {
  if (stopping) return;
  stopping = true;
  await vite?.close();

  if (backend.pid && backend.exitCode === null) {
    if (process.platform === 'win32') {
      // Python may own a training subprocess: stop that tree on Ctrl+C too.
      spawnSync('taskkill', ['/PID', String(backend.pid), '/T', '/F'], {
        windowsHide: true,
        stdio: 'ignore',
      });
    } else {
      try {
        process.kill(-backend.pid, 'SIGTERM');
      } catch (error) {
        if (error.code !== 'ESRCH') console.error(error.message);
      }
    }
  }
  process.exit(exitCode);
}

process.on('SIGINT', () => void stop(0));
process.on('SIGTERM', () => void stop(0));
backend.on('error', (error) => {
  console.error(`Could not start the Python backend: ${error.message}`);
  void stop(1);
});
backend.on('exit', (code, signal) => {
  if (!stopping) {
    console.error(`The Python backend stopped (${signal || `exit ${code}`}).`);
    void stop(code || 1);
  }
});

try {
  vite = await createServer({ configFile: resolve(projectRoot, 'vite.config.ts') });
  await vite.listen();
  console.log('\nAI Content Breaker — frontend and backend are starting together.');
  console.log('Python API: http://127.0.0.1:8000');
  vite.printUrls();
  console.log('Press Ctrl+C to stop both servers.\n');
} catch (error) {
  console.error(`Could not start the frontend: ${error.message}`);
  await stop(1);
}
