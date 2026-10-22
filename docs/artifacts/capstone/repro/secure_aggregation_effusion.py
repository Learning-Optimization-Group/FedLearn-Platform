"""Effusion protocol v1 amendment A5: secure-aggregation equivalence and overhead on the head path.

Swaps only the engine's `mean` reduction, inside a scoped context, for the existing
LightSecAgg protocol run in-process: the real SecureAggregationClient (X25519, Shamir
shares of an os.urandom mask, sealed relay through the real servicer handlers) and the
real SecureAggregationSession recovery. Summed shares are filed with the servicer's
holder-index check but bypass its DeComFL-specific completion hook. No network is
involved. Rules that need individual updates are refused, not silently made plain.
"""
import argparse
from collections import Counter, OrderedDict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_effusion_federated as federated  # noqa: E402  (also pins this worktree's fedlearn)
import torch  # noqa: E402

import fedlearn.server.composed_strategy as composed  # noqa: E402
from fedlearn.client.secure_agg_client import _QUANTIZATION_SCALE, FrozenSurvivors, SecureAggregationClient  # noqa: E402
from fedlearn.communication.generated import fedlearn_pb2 as pb  # noqa: E402
from fedlearn.security.lightsecagg import PRIME  # noqa: E402
from fedlearn.server.coordinator import FLCoordinator  # noqa: E402
from fedlearn.server.decomfl_strategy import DeComFL  # noqa: E402
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer  # noqa: E402

Refusal = federated.Refusal
SEEDS = federated.SEEDS
PLAIN_UPLOAD_BYTES = 8348
COMPARISON_SEED = 20261025


class _Context:
    def set_code(self, code): self.code = code
    def set_details(self, details): self.details = details
    def invocation_metadata(self): return ()


class CountingStub:
    """Calls the real servicer in-process as one partition and counts serialised bytes."""

    def __init__(self, servicer, partition, ledger):
        self._servicer, self.partition, self._ledger = servicer, partition, ledger

    def _count(self, name, request, response):
        self._ledger[(self.partition, 'up', name)] += request.ByteSize()
        self._ledger[(self.partition, 'down', name)] += response.ByteSize()
        return response

    def _call(self, name, request):
        self._servicer._partition_extractor = lambda ctx, p=self.partition: p
        return self._count(name, request, getattr(self._servicer, name)(request, _Context()))

    def PublishPublicKey(self, request): return self._call('PublishPublicKey', request)
    def SubmitSecureShares(self, request): return self._call('SubmitSecureShares', request)
    def GetDeComFLConfig(self, request): return self._call('GetDeComFLConfig', request)

    def SubmitAggregatedShare(self, request):
        """The servicer's holder-index check and filing, without its DeComFL completion hook."""
        session = self._servicer._secure_session(request.round)
        holder = session.holder_index_for(self.partition)
        if request.holder_index != holder:
            raise Refusal('holder_index_mismatch')
        remaining = session.submit_summed_share(holder_index=holder, share=list(request.summed_share))
        return self._count('SubmitAggregatedShare', request,
                           pb.SubmitAggregatedShareResponse(received=True, shares_still_needed=remaining))


class SecureMeanReducer:
    """A drop-in for `reduce_deltas` that computes the mean with the real LightSecAgg protocol."""

    def __init__(self):
        self.runs, self.reuses, self._cache = [], 0, {}

    def reduce(self, deltas, weights, keys, rule, rule_params):
        if rule != 'mean':
            raise Refusal(f'secure_aggregation_unavailable:{rule} needs individual updates; the server learns only the sum')
        flat = torch.stack([torch.cat([d[k].detach().float().reshape(-1) for k in keys]) for d in deltas])
        key = hashlib.sha256(flat.numpy().tobytes() + repr(keys).encode()).hexdigest()
        if key in self._cache:
            self.reuses += 1
            mean = self._cache[key]
        else:
            mean, run = self._protocol(flat)
            self._cache[key] = mean
            self.runs.append(run)
        out, offset = OrderedDict(), 0
        for k in keys:
            size = deltas[0][k].numel()
            out[k] = mean[offset:offset + size].reshape(deltas[0][k].shape).to(torch.float32)
            offset += size
        return out, None

    def _protocol(self, flat):
        n, scalars = flat.shape
        threshold = n // 2 + 1
        strategy = DeComFL(initial_parameters=OrderedDict(w=torch.zeros(4)), num_local_steps=1, num_perturbations=scalars)
        coordinator = FLCoordinator(strategy, min_clients_for_aggregation=1, clients_per_round=n)
        coordinator.bind_or_check_identity = lambda *args, **kwargs: True
        servicer = FederatedLearningServiceServicer(coordinator, secure_agg_threshold=threshold)
        ledger, partitions, timings = Counter(), list(range(1, n + 1)), {}
        clients = {p: SecureAggregationClient(CountingStub(servicer, p, ledger), client_id=f'c{p}') for p in partitions}
        clock = time.perf_counter()
        for p in partitions:
            clients[p].begin_round(round_num=1, threshold=threshold, num_scalars=scalars, cohort_size=n)
        timings['keys'] = time.perf_counter() - clock; clock = time.perf_counter()
        for p in partitions:
            clients[p].distribute_shares(round_num=1)
        timings['shareAndSeal'] = time.perf_counter() - clock; clock = time.perf_counter()
        for p in partitions:
            clients[p].collect_shares(round_num=1)
        timings['collectAndOpen'] = time.perf_counter() - clock; clock = time.perf_counter()
        session = servicer._secure_session(1)
        for p in partitions:
            elements = clients[p].mask(flat[p - 1].tolist())
            request = pb.SubmitGradientScalarsRequest(
                client_id=f'c{p}', trained_on_round=1, num_examples=1,
                masked_gradients=pb.MaskedGradientScalars(elements=elements, modulus=PRIME, num_local_steps=1,
                                                         num_perturbations=scalars))
            ledger[(p, 'up', 'SubmitGradientScalars(masked)')] += request.ByteSize()
            session.submit_masked(partition=p, elements=elements)
        session.close_submissions()
        timings['maskAndSubmit'] = time.perf_counter() - clock; clock = time.perf_counter()
        survivors = FrozenSurvivors(tuple(session.survivors))
        for p in partitions[:threshold]:
            clients[p].finish_round(round_num=1, survivors=survivors)
        timings['summedShares'] = time.perf_counter() - clock; clock = time.perf_counter()
        recovered = session.recover()
        timings['recover'] = time.perf_counter() - clock
        mean = (recovered / n).to(torch.float32)
        plain = flat.mean(0)
        error = float((mean - plain).abs().max())
        bound = 0.5 / _QUANTIZATION_SCALE + 3 * 2 ** -24 * (float(plain.abs().max()) + float(recovered.abs().max()) / n)
        uploads = [sum(v for (p, d, _), v in ledger.items() if p == q and d == 'up') for q in partitions]
        downloads = [sum(v for (p, d, _), v in ledger.items() if p == q and d == 'down') for q in partitions]
        masked = [ledger[(q, 'up', 'SubmitGradientScalars(masked)')] for q in partitions]
        return mean, {
            'clients': n, 'threshold': threshold, 'scalars': scalars, 'survivors': len(session.survivors),
            'maxAbsErrorVsFloatMean': error, 'fixedPointBoundOnMean': bound, 'withinBound': error <= bound,
            'timings': timings, 'protocolSeconds': sum(timings.values()),
            'bytes': {'perClientUpload': float(np.mean(uploads)), 'perClientUploadMax': int(max(uploads)),
                      'perClientDownload': float(np.mean(downloads)), 'maskedSubmission': float(np.mean(masked)),
                      'byRpcUp': {name: sum(v for (p, d, nm), v in ledger.items() if d == 'up' and nm == name) / n
                                  for name in sorted({nm for (_, _, nm) in ledger})}},
        }


@contextmanager
def secure_mean(reducer):
    """Route ComposedStrategy's reduction through `reducer` for the duration of the block only."""
    original = composed.reduce_deltas
    composed.reduce_deltas = reducer.reduce
    try:
        yield reducer
    finally:
        composed.reduce_deltas = original


def main(args):
    import effusion_labels
    import fit_effusion_gate
    from sklearn.metrics import roc_auc_score
    output = Path(args.output)
    if output.exists():
        raise Refusal('output_exists')
    warnings.filterwarnings('ignore', message=r'.* encountered in matmul', category=RuntimeWarning)
    raw_gate = (Path(args.gate_dir) / 'gate.json').read_bytes()
    if hashlib.sha256(raw_gate).hexdigest() != args.gate_sha256:
        raise Refusal('gate_hash_mismatch')
    authority = federated.check_gate(json.loads(raw_gate), args.amendment)
    raw_baseline = (Path(args.baseline_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_baseline).hexdigest() != args.baseline_sha256:
        raise Refusal('baseline_hash_mismatch')
    baseline = np.load(Path(args.baseline_dir) / 'selection-margins.npz')
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    features, _ = fit_effusion_gate._load_representations(args.historical_cache, args.xrv_cache, args.xrv_report_sha256, labels)
    train_x, selection_x = features['R1']
    heads = np.load(Path(args.gate_dir) / 'selected-heads.npz')
    mean, std = fit_effusion_gate.standardiser(train_x)
    if not (np.array_equal(mean, heads['R1_mean']) and np.array_equal(std, heads['R1_std'])):
        raise Refusal('standardiser_mismatch')
    y_train, y_selection = labels['train']['y'], labels['selection']['y']
    train_patients = [r['patient_id'] for r in labels['train']['rows']]
    selection_patients = np.array([r['patient_id'] for r in labels['selection']['rows']])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.secure-', dir=output.parent))
    started = time.perf_counter()
    margins, per_seed = {}, []
    try:
        for seed in SEEDS:
            reducer = SecureMeanReducer()
            with secure_mean(reducer):
                record, margin = federated.run_one(f'secure_{seed}', seed, train_x, y_train, train_patients, selection_x,
                                                   y_selection, mean, std, clients=federated.CLIENTS,
                                                   features_path=args.xrv_cache, report_sha256=args.xrv_report_sha256,
                                                   info={'representation': 'R1', 'amendment': 'A5', 'aggregation': 'LightSecAgg mean'})
            plain_record = json.loads((Path(args.baseline_dir) / f'fed_{seed}.json').read_text())
            plain = baseline[f'fed_{seed}']
            margins[f'secure_{seed}'] = margin
            runs = reducer.runs
            per_seed.append({
                'seed': seed, 'protocolRuns': len(runs), 'previewReuses': reducer.reuses,
                'allWithinFixedPointBound': all(r['withinBound'] for r in runs),
                'maxAggregateError': max(r['maxAbsErrorVsFloatMean'] for r in runs),
                'maxFixedPointBound': max(r['fixedPointBoundOnMean'] for r in runs),
                'finalMaxAbsMarginDifference': float(np.abs(margin - plain).max()),
                'secureAuroc': float(roc_auc_score(y_selection, margin)), 'plainAuroc': float(roc_auc_score(y_selection, plain)),
                'maxAbsRoundAurocDifference': max(abs(a['auc'] - b['auc']) for a, b in zip(record['rounds'], plain_record['rounds'])),
                'meanProtocolSecondsPerRound': float(np.mean([r['protocolSeconds'] for r in runs])),
                'meanPhaseSeconds': {k: float(np.mean([r['timings'][k] for r in runs])) for k in runs[0]['timings']},
                'bytesPerClientPerRound': runs[0]['bytes'], 'clients': runs[0]['clients'], 'threshold': runs[0]['threshold'],
                'scalars': runs[0]['scalars'],
            })
            (staging / f'secure_{seed}.json').write_text(json.dumps({'record': record, 'protocolRuns': runs}, sort_keys=True, default=str) + '\n')
            print(json.dumps({k: per_seed[-1][k] for k in ('seed', 'protocolRuns', 'allWithinFixedPointBound')}), flush=True)
        from fit_effusion_gate import cluster_bootstrap
        draws = []
        for index in cluster_bootstrap(selection_patients, reps=2000, seed=COMPARISON_SEED):
            yy = y_selection[index]
            if yy.min() == yy.max():
                continue
            draws.append(float(np.mean([roc_auc_score(yy, margins[f'secure_{s}'][index]) - roc_auc_score(yy, baseline[f'fed_{s}'][index])
                                        for s in SEEDS])))
        point = float(np.mean([p['secureAuroc'] - p['plainAuroc'] for p in per_seed]))
        upload = float(np.mean([p['bytesPerClientPerRound']['perClientUpload'] for p in per_seed]))
        summary = {
            'schema': 'capstone.effusion.secure_aggregation.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A5',
            'gateAuthority': authority,
            'scope': 'LightSecAgg in-process (no network) replacing only the mean reduction in the section 6 R1 runs; selection half only.',
            'trustChange': 'Clients self-clip before masking; the server cannot verify clipping under secure aggregation.',
            'unavailable': {'trimmed_mean': 'needs per-coordinate order of individual updates; server learns only the sum',
                            'median': 'same reason', 'krum/multi_krum/bulyan': 'need pairwise distances between individual updates',
                            'weighted_mean': 'not supported by this masked-sum path as built (uniform sum only)'},
            'perSeed': per_seed,
            'seedMeanAurocDifferenceSecureMinusPlain': {'point': point, 'interval95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                                                         'bootstrap': {'reps': 2000, 'seed': COMPARISON_SEED}},
            'uploadOverhead': {'plainBytesPerClientPerRound': PLAIN_UPLOAD_BYTES, 'secureBytesPerClientPerRound': upload,
                               'ratio': upload / PLAIN_UPLOAD_BYTES},
            'inputs': {'labelInputs': provenance, 'baselineSummarySha256': args.baseline_sha256, 'xrvCacheReportSha256': args.xrv_report_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in ('secure_aggregation_effusion.py', 'run_effusion_federated.py')},
            'secureSourceSha256': {name: federated._sha256_file(federated.FRAMEWORK_SRC / 'fedlearn' / name) for name in
                                   ('client/secure_agg_client.py', 'server/secure_agg_session.py', 'security/lightsecagg.py',
                                    'server/grpc_servicer.py', 'server/composed_strategy.py')},
            'elapsedSeconds': round(time.perf_counter() - started, 1),
        }
        (staging / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        np.savez(staging / 'selection-margins.npz', **margins)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(output)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--baseline-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--baseline-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'elapsedSeconds': summary['elapsedSeconds']}))
