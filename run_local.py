"""CPU-friendly launcher; upstream model and privacy implementation stay intact."""
import argparse
import contextlib
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    local, extra = parser.parse_known_args()
    if local.threads < 1:
        parser.error('--threads must be positive')
    if any(arg == '--log' or arg.startswith('--log=') for arg in extra):
        parser.error('Log paths are generated automatically; omit --log.')
    if local.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = str(local.threads)
    os.environ['MKL_NUM_THREADS'] = str(local.threads)
    root = Path(__file__).resolve().parent
    os.chdir(root)
    run_dir = root / 'logs' / ('local-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    sys.argv = [str(root / 'main.py'), '--epochs', '1', '--batch', '16',
                '--emb_size', '32', '--eps', '0', '--sp', '0.8',
                '--align', '100', '--et', '0.3', '--mode', 'JLT',
                '--log', str(run_dir), *extra]
    # main.py parses arguments on import. For --help it exits without training.
    upstream = importlib.import_module('main')
    upstream.torch.set_num_threads(local.threads)
    if upstream.args.epochs < 1:
        parser.error('--epochs must be positive')
    if not 0 <= upstream.args.eps < 9:
        parser.error('--eps must be an index from 0 through 8')
    if upstream.args.mode not in ('JLT', 'FJLT'):
        parser.error('--mode must be JLT or FJLT')
    if upstream.args.et <= 0:
        parser.error('--et must be positive')
    random.seed(upstream.args.seed)
    upstream.np.random.seed(upstream.args.seed)
    upstream.torch.manual_seed(upstream.args.seed)
    if upstream.args.cuda:
        upstream.torch.cuda.manual_seed_all(upstream.args.seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'python': sys.version, 'executable': sys.executable,
        'torch': upstream.torch.__version__, 'numpy': upstream.np.__version__,
        'arguments': vars(upstream.args), 'threads': local.threads,
        'upstream_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'Upstream sample execution; not a verified paper reproduction.',
        'launcher_changes': 'CPU default, thread limit, unique logs, seed initialization',
    }
    (run_dir / 'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    with (run_dir / 'console.txt').open('w', encoding='utf-8') as output:
        with contextlib.redirect_stdout(Tee(sys.stdout, output)), contextlib.redirect_stderr(Tee(sys.stderr, output)):
            print('Run directory:', run_dir)
            print('Arguments:', upstream.args)
            print('Sample execution only; see README_LOCAL.md for paper/code differences.')
            upstream.main()
            print('Completed. Full console output and configuration:', run_dir)


if __name__ == '__main__':
    run()
