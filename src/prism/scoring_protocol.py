"""Explicit reproducible IFEval rescoring of immutable, already generated answers."""
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import random
from threading import RLock

from .benchmarks import score_benchmark
from .io import digest, dump, receipt, rows, verify_receipt

PROTOCOL = 'ifeval_fixed_rng_v1'
METRICS = 'benchmark.fixed.metrics.json'
RECEIPT = 'benchmark.fixed.receipt.json'
_LOCK = RLock()


@contextmanager
def fixed_checker_rng():
    # The official checker uses random defaults as well as langdetect's own RNG.
    # Restore caller state, including on exceptions; never affect model sampling.
    from langdetect import DetectorFactory
    with _LOCK:
        state, seed = random.getstate(), DetectorFactory.seed
        try:
            random.seed(0)
            DetectorFactory.seed = 0
            yield
        finally:
            random.setstate(state)
            DetectorFactory.seed = seed


def fixed_score(c, generated):
    with fixed_checker_rng():
        return score_benchmark(c, generated)


def flag_changes(original, revised):
    changes = []
    for left, right in zip(original['details'], revised['details'], strict=True):
        if left['id'] != right['id']:
            raise ValueError('Rescoring changed response order')
        for mode in ('strict', 'loose'):
            for index, (a, b) in enumerate(zip(left[mode]['follow_instruction_list'], right[mode]['follow_instruction_list'], strict=True)):
                if a != b:
                    changes.append({'id': left['id'], 'key': left['key'], 'mode': mode,
                                    'instruction': left[mode]['instruction_id_list'][index], 'original': a, 'fixed': b})
    return changes


def materialize(c, directory):
    """Publish a new score sidecar; never modify generation or original metrics."""
    directory = Path(directory)
    marker = directory/('reference.receipt.json' if directory.name == 'reference' else 'evaluation.generate.receipt.json')
    verify_receipt(marker)
    if (directory/RECEIPT).exists():
        return effective_benchmark(c, directory, required=True)
    if (directory/METRICS).exists():
        raise FileExistsError('Preserve and inspect an incomplete rescoring attempt')
    original_path, responses = directory/'benchmark.metrics.json', directory/'benchmark.responses.jsonl'
    original_hash = digest(original_path)
    original = json.loads(original_path.read_text(encoding='utf-8'))
    generated = rows(responses)
    scored = fixed_score(c, generated)
    if fixed_score(c, generated) != scored:
        raise RuntimeError('Fixed-RNG official flags are not reproducible')
    assert digest(original_path) == original_hash
    evidence = {'protocol': PROTOCOL, 'python_random_seed': 0, 'langdetect_seed': 0,
                'langdetect_version': importlib.metadata.version('langdetect'),
                'official_source_revision': c['choices']['benchmark']['official_source_revision'],
                'original_metrics_sha256': original_hash, 'original_metrics': original['metrics'],
                'changed_flags': flag_changes(original, scored), 'independent_repeat_equal': True}
    dump(directory/METRICS, {**scored, 'scoring_protocol': evidence})
    receipt(directory/RECEIPT, [directory/METRICS, original_path, responses, marker, Path(__file__)], scoring_protocol=PROTOCOL)
    return effective_benchmark(c, directory, required=True)


def effective_benchmark(c, directory, *, required=False):
    directory = Path(directory)
    if not (directory/RECEIPT).exists():
        if required and c['choices']['benchmark'].get('kind') == 'ifeval':
            raise RuntimeError('Explicit V2 IFEval evaluation requires certified fixed-RNG scoring')
        return json.loads((directory/'benchmark.metrics.json').read_text(encoding='utf-8'))
    verified = verify_receipt(directory/RECEIPT)
    scored = json.loads((directory/METRICS).read_text(encoding='utf-8'))
    evidence = scored['scoring_protocol']
    if (verified.get('scoring_protocol') != PROTOCOL or evidence['protocol'] != PROTOCOL
            or evidence['python_random_seed'] != 0 or evidence['langdetect_seed'] != 0
            or not evidence['independent_repeat_equal']
            or evidence['original_metrics_sha256'] != digest(directory/'benchmark.metrics.json')
            or evidence['official_source_revision'] != c['choices']['benchmark']['official_source_revision']):
        raise ValueError('Fixed-RNG scoring evidence changed')
    return scored
