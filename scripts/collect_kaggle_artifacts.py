"""Collect allowlisted research artifacts into a NEW repository copy; no training."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_checked(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    if destination.exists():
        if digest(source) != digest(destination):
            raise ValueError('Refusing to replace a different file: ' + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = digest(source)
    temporary = destination.with_name(destination.name + '.writing')
    shutil.copy2(source, temporary)
    if digest(temporary) != before or digest(source) != before:
        raise ValueError('Source changed during copy: ' + str(source))
    temporary.rename(destination)


def write_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', required=True, help='Fresh extracted template directory')
    parser.add_argument('--working', default='/kaggle/working')
    parser.add_argument('--original-csv', default='/kaggle/input/datasets/feras2004/spliteoriginal/splitsOriginal.csv')
    parser.add_argument('--results-dir')
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    repo = Path(args.repo_root).resolve()
    work = Path(args.working).resolve()
    if repo == work or work in repo.parents and repo.name in {'qr_three_systems_random_fusion_v1', 'qr_three_systems_frozen_v1'}:
        raise ValueError('Use the fresh repository template, not an experiment directory')
    if (repo / 'PACKAGE_STATUS.json').exists():
        status = json.loads((repo / 'PACKAGE_STATUS.json').read_text())
        if status.get('kind') == 'complete_kaggle_package':
            raise ValueError('Already assembled; use a fresh template copy')
    new = work / 'qr_three_systems_random_fusion_v1'
    frozen = work / 'qr_three_systems_frozen_v1'
    visual_run = work / 'mobilenet_continue/robust_aug_phase2_v1'
    prepared = work / 'qr_visual_fresh'
    selected = new / 'selected_three_systems_epoch_01_b06b88974bbe.pt'
    receipt = json.loads(selected.with_suffix('.json').read_text())
    assert receipt['file'] == selected.name and digest(selected) == receipt['sha256']
    cp = torch.load(selected, map_location='cpu', weights_only=False)
    expected = 'd4509181f834a16144ad45e24b8ce686a4e608e07a9846a880757c705739d1ff'
    assert cp['epoch'] == 1 and cp['identity']['data_signature'] == expected
    assert cp['cfg']['initialization'] == 'random'
    for value in cp['model_state'].values():
        if value.is_floating_point():
            assert torch.isfinite(value).all()
    for key, value in cp['fusion_head_state'].items():
        assert torch.equal(value, cp['model_state']['fusion.' + key])
    source = cp['source_model_identity']
    warnings = []
    model_records = []

    def add_model(path, name, role, expected_sha=None):
        path = Path(path)
        if expected_sha is not None:
            assert digest(path) == expected_sha, str(path)
        destination = repo / 'models' / name
        copy_checked(path, destination)
        record = dict(file=name, sha256=digest(destination), bytes=destination.stat().st_size,
                      original_name=path.name, role=role)
        write_json(record, destination.with_suffix('.json'))
        model_records.append(record)
        print('Verified model:', name, flush=True)

    add_model(selected, 'selected_three_systems.pt', 'Selected complete visual/text/learned-fusion export', receipt['sha256'])
    add_model(frozen / 'source_backup/visual.pt', 'source_visual.pt', 'Frozen robust visual source', source['visual_sha256'])
    add_model(frozen / 'source_backup/text.pt', 'source_text.pt', 'Frozen legacy text source', source['text_sha256'])
    visual_cp = torch.load(frozen / 'source_backup/visual.pt', map_location='cpu', weights_only=False)
    phase1_sha = visual_cp.get('source_sha256')
    phase1_candidates = [visual_run / 'source_original.pt', work / 'mobilenet_continue/full_train_phase1_fp32/best_epoch_04_10a6296afd7d.pt']
    phase1 = next((p for p in phase1_candidates if p.is_file() and digest(p) == phase1_sha), None)
    if phase1 is not None:
        add_model(phase1, 'visual_phase1.pt', 'Visual before geometric augmentation', phase1_sha)
    else:
        warnings.append('Optional pre-augmentation visual checkpoint unavailable; its full reproduction needs the original backup.')
    legacy = work / 'runs/abl_visual_n200000_seed42.pt'
    legacy_sha = '9551f2c9e7e3a5b1d0a0d31264e0d67be612995d196d09c76fd190b23494008c'
    if legacy.is_file() and digest(legacy) == legacy_sha:
        add_model(legacy, 'legacy_visual_200k.pt', 'Original 200K visual warm-start source', legacy_sha)
    else:
        warnings.append('Optional legacy 200K visual checkpoint unavailable. Initial training is not reproducible from this package alone.')
    tokenizer = new / 'tokenizer'
    expected_tokenizer = source.get('tokenizer_files_sha256', {})
    assert expected_tokenizer
    for name, sha in expected_tokenizer.items():
        assert Path(name).name == name and digest(tokenizer / name) == sha
        copy_checked(tokenizer / name, repo / 'models/tokenizer' / name)
    write_json(model_records, repo / 'models/weights_manifest.json')

    # Collect authoritative Test reports selected by matching WEIGHTS, not performance.
    candidates = [Path(args.results_dir)] if args.results_dir else sorted(
        (p for p in new.glob('final_test_*') if p.is_dir()), reverse=True
    )
    final = None
    for directory in candidates:
        try:
            contract = json.loads((directory / 'evaluation_contract.json').read_text())
            completed = json.loads((directory / 'completion_record.json').read_text())
            assert contract['weights_sha256'] == receipt['sha256']
            assert contract['data_signature'] == expected and contract['threshold'] == 0.5
            assert completed['status'] == 'completed'
            for entry in completed['files']:
                assert Path(entry['file']).name == entry['file']
                assert digest(directory / entry['file']) == entry['sha256']
            final = directory
            break
        except Exception as exc:
            print('Skipping results folder:', directory.name, type(exc).__name__, flush=True)
    if final is None:
        raise ValueError('No complete final Test results matching the selected weights. Keep the original results folder.')
    summary = pd.read_csv(final / 'test_all_systems_summary.csv')
    assert set(summary.system) == {'visual', 'text', 'feature_fusion', 'text_with_visual_fallback'}
    assert summary.samples.eq(146425).all()
    required_results = ['test_all_systems_summary.csv', 'test_three_systems_comparison.csv', 'test_metrics.json',
                        'paired_correctness_counts.csv', 'evaluation_contract.json', 'completion_record.json', 'test_roc_pr.png']
    for name in required_results:
        copy_checked(final / name, repo / 'reports/final_test' / name)
    for pattern in ['*_classification_report.txt', '*_confusion_matrix.png']:
        for path in final.glob(pattern):
            copy_checked(path, repo / 'reports/final_test' / path.name)
    predictions = pd.read_csv(final / 'test_predictions.csv.gz', keep_default_na=False)
    assert len(predictions) == 146425
    compact = pd.DataFrame()
    compact['image_id'] = predictions.path.map(lambda p: p[p.index('QR_All_'):] if 'QR_All_' in p else Path(p).name)
    compact['label'] = predictions.label
    compact['ok'] = predictions.ok
    compact['available'] = predictions.available
    compact['image_sha256'] = predictions.sha256
    compact['domain_group_sha256'] = predictions.domain.astype(str).map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    for column in predictions.columns:
        if column.endswith('_probability') or column.endswith('_prediction'):
            compact[column] = predictions[column]
    compact.to_csv(repo / 'reports/final_test/predictions_compact.csv.gz', index=False, compression='gzip')

    # Preserve experiment histories/configs; never include image/feature caches.
    runs = {'visual_phase1': work / 'mobilenet_continue/full_train_phase1_fp32',
            'visual_robust': visual_run, 'text_initialized_fusion': frozen, 'random_fusion': new}
    names = ['config.json', 'history.json', 'completion.json', 'validation_comparison.csv',
             'validation_selected_comparison.csv', 'validation_selected_metrics.json',
             'validation_modality_ablations.csv', 'validation_modality_ablations.json']
    for label, directory in runs.items():
        for name in names:
            if (directory / name).is_file():
                copy_checked(directory / name, repo / 'reports/training' / label / name)
        for path in directory.glob('runtime_resume_*.json'):
            copy_checked(path, repo / 'reports/training' / label / path.name)
    # Compact optional optimizer checkpoint; keep outside normal git tracking.
    chosen_resume = None
    for path in sorted(new.glob('epoch_01_*.pt')):
        try:
            r = json.loads(path.with_suffix('.json').read_text())
            assert digest(path) == r['sha256']
            c = torch.load(path, map_location='cpu', weights_only=False)
            assert all(torch.equal(c['fusion_head_state'][key], value) for key, value in cp['fusion_head_state'].items())
            chosen_resume = path
            break
        except Exception:
            continue
    if chosen_resume is not None:
        copy_checked(chosen_resume, repo / 'checkpoints/fusion_resume.pt')
        write_json(dict(file='fusion_resume.pt', sha256=digest(chosen_resume)), repo / 'checkpoints/fusion_resume.json')

    # Exact retained source-row indices are small; no raw URLs/images are copied.
    audit = json.loads((prepared / 'audit.json').read_text())
    assert audit['data_signature'] == expected
    original_csv = Path(args.original_csv)
    assert digest(original_csv) == audit['source_splits_sha256']
    manifest = None
    for path in [prepared / 'verified_mobilenet_inputs/splits_verified.parquet', prepared / 'splits_clean.parquet']:
        if path.is_file():
            try:
                data = pd.read_parquet(path)
                assert len(data) == 1005700 and data.path.is_unique
                h = hashlib.sha256()
                for row in data[['path', 'label', 'split', 'url', 'ok', 'sha256']].itertuples(index=False, name=None):
                    h.update((json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
                assert h.hexdigest() == expected
                manifest = data
                break
            except Exception:
                continue
    if manifest is None:
        raise ValueError('A verified clean manifest is required to publish exact split membership indices')
    raw = pd.read_csv(original_csv, keep_default_na=False)
    assert len(raw) == 1005738 and raw.path.is_unique
    raw['source_row_index'] = np.arange(len(raw), dtype=np.uint32)
    keys = manifest.source_path if 'source_path' in manifest else manifest.path
    positions = raw.set_index('path').loc[keys, 'source_row_index'].to_numpy(dtype=np.uint32)
    assert len(np.unique(positions)) == len(manifest)
    aligned = raw.iloc[positions.astype(np.int64)]
    for column in ['label', 'ok', 'split', 'url', 'domain', 'fold']:
        assert np.array_equal(aligned[column].astype(str).to_numpy(), manifest[column].astype(str).to_numpy())
    indices_path = repo / 'data/retained_source_row_indices.npy'
    np.save(indices_path, positions, allow_pickle=False)
    write_json(dict(original_rows=len(raw), retained_rows=len(manifest), source_csv_sha256=digest(original_csv),
                    indices_sha256=digest(indices_path), data_signature=expected,
                    order='Exact saved clean-manifest order; positions are zero-based rows in the immutable source CSV'),
               repo / 'data/membership_record.json')
    copy_checked(prepared / 'audit.json', repo / 'data/audit.json')
    if (prepared / 'split_summary.csv').is_file():
        copy_checked(prepared / 'split_summary.csv', repo / 'data/split_summary.csv')
    del raw, manifest, predictions, compact

    environment = {'python': sys.version, 'packages': {}, 'torch_cuda_build': torch.version.cuda,
                   'cuda_available_during_collection': torch.cuda.is_available()}
    for name in ['torch', 'torchvision', 'transformers', 'tokenizers', 'numpy', 'pandas', 'scikit-learn',
                 'Pillow', 'pyarrow', 'matplotlib', 'tqdm', 'opencv-python', 'opencv-python-headless']:
        try:
            environment['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(environment, repo / 'environment.recorded.json')
    (repo / 'requirements-recorded.txt').write_text('\n'.join(
        name + '==' + version for name, version in environment['packages'].items()
    ) + '\n')
    # Validate portable code against the ACTUAL weights inside Kaggle before claiming complete.
    print('Checking strict model loading + synthetic CPU forward; no dataset training...', flush=True)
    run = subprocess.run([sys.executable, str(repo / 'scripts/smoke_model.py')], cwd=repo,
                         capture_output=True, text=True, timeout=180)
    (repo / 'reports/model_smoke_check.txt').write_text(run.stdout + '\n' + run.stderr)
    if run.returncode != 0:
        raise RuntimeError('Portable model check failed. Read reports/model_smoke_check.txt; sources are untouched.')
    for record in model_records:
        assert digest(repo / 'models' / record['file']) == record['sha256']
    write_json(dict(kind='complete_kaggle_package', collected_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    model_forward_smoke=True, new_real_image_evaluation=False,
                    final_test_source=str(final), optional_missing=warnings,
                    dataset_images_included=False, raw_urls_included=False, feature_cache_included=False,
                    notebooks_tested_on_gpu_by_packager=False), repo / 'PACKAGE_STATUS.json')
    # Remove the obsolete template manifest only in this NEW generated copy.
    template_manifest = repo / 'MANIFEST.json'
    entries = []
    for path in sorted(repo.rglob('*')):
        if not path.is_file() or path == template_manifest or '__pycache__' in path.parts:
            continue
        if path.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError('Single file too large for normal GitHub Git tracking: ' + str(path.relative_to(repo)))
        entries.append(dict(path=str(path.relative_to(repo)), bytes=path.stat().st_size, sha256=digest(path)))
    write_json(dict(files=entries), template_manifest)
    subprocess.run([sys.executable, str(repo / 'scripts/verify_package.py')], cwd=repo, check=True)
    archive_path = work / ('Quishing_GitHub_COMPLETE_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6] + '.zip')
    temporary = archive_path.with_suffix('.writing')
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for entry in entries:
            archive.write(repo / entry['path'], arcname=repo.name + '/' + entry['path'])
        archive.write(template_manifest, arcname=repo.name + '/MANIFEST.json')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for entry in entries:
            h = hashlib.sha256()
            with archive.open(repo.name + '/' + entry['path']) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            assert h.hexdigest() == entry['sha256']
    temporary.rename(archive_path)
    print('\nCOMPLETE:', archive_path, flush=True)
    print('Repository files:', len(entries) + 1, '| ZIP MiB:', round(archive_path.stat().st_size / 1024**2, 2), flush=True)
    print('Extract ZIP; upload source files to GitHub. checkpoints/ is optional private resumption data.', flush=True)
    for warning in warnings:
        print('Note:', warning, flush=True)


if __name__ == '__main__':
    main()
