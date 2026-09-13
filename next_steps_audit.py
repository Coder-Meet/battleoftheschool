"""Bounded follow-up to CURRENT_E2E_REVIEW: current-label filter fits, roots, and review queue.

New model fits exclude all five reference patients; only original validation patients select thresholds.
Experiments do not change production settings or create new annotation verdicts.
"""
import argparse
from collections import Counter
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path

import SimpleITK as sitk

from current_e2e_fusion import fuse
from detector import DetectorConfig
from final_eval_topology import infer
from final_evaluation import CASES, ROOT, case_paths, digest, read_json, score_variant, write_json
from learning import CandidateModel, FEATURE_NAMES, load_reviews, train
from nifti_io import read_nifti
from pipeline import DEFAULT_MODEL
from review_analytics import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    audit = read_json(args.audit_dir / 'provenance.json')
    for name in ('detector.py', 'learning.py'):
        if digest(ROOT / name) != audit['source_sha256'][name]:
            raise ValueError('Re-extract current candidate identities before this follow-up.')
    rows = list(csv.DictReader((args.audit_dir / 'review-queue.csv').open()))
    pool = [r for r in rows if r['profile'] == 'pool']
    original = load_reviews([ROOT / 'labels/reviews.json'])
    lookup = {(r['case_id'], identity(r['fingerprint'], r['features'])): r for r in original}
    current, queue = [], []
    old_split = read_json(ROOT / 'labels/split.json')
    for row in pool:
        vector = json.loads(row['fingerprint'])[-1]
        previous = lookup.get((row['case'], identity(row['fingerprint'], vector)))
        score = float(row['synthetic_score'])
        label = previous['label'] if previous else 'unreviewed'
        reason = ('prior_verdict_filter_disagreement' if previous and (label == 'confirmed') != (score >= .15)
                  else 'unknown_filter_survivor' if not previous and score >= .15
                  else 'unknown_below_threshold' if not previous else 'previous_verdict_agrees')
        queue.append({
            'case_id': row['case'], 'instance_id': row['instance_id'], 'reason': reason,
            'priority': {'prior_verdict_filter_disagreement': 0, 'unknown_filter_survivor': 1,
                         'unknown_below_threshold': 2, 'previous_verdict_agrees': 3}[reason],
            'synthetic_score': score, 'previous_exact_label': label,
            'patient_partition': 'reference_development' if row['case'] in CASES else next(
                (p for p, ids in old_split.items() if row['case'] in ids), 'unassigned'),
            'fingerprint': row['fingerprint'], 'new_verdict': '', 'reviewer': '', 'notes': '',
        })
        if previous and row['case'] not in CASES:
            current.append({**previous, 'instance_id': row['instance_id'], 'fingerprint': row['fingerprint'],
                            'features': vector})
    queue.sort(key=lambda r: (r['priority'], r['case_id'], r['instance_id']))
    with (output / 'adjudication-queue.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(queue[0]));writer.writeheader();writer.writerows(queue)
    present = {r['case_id'] for r in current}
    split = {p: [c for c in ids if c in present and c not in CASES] for p, ids in old_split.items()}
    write_json(output / 'current-exact-reviews.json', {'schema_version':1,'scope':'candidate_reviews_only',
               'feature_names':FEATURE_NAMES,'records':current})
    write_json(output / 'split.json', split)
    # Freeze the finite family before inspecting any new fit or tracing outcome.
    configurations = {
        'alternate_strict_roots': DetectorConfig(roots_per_contact=6),
        'alternate_review_roots': DetectorConfig.review(minimum_origin_diameter_mm=2),
        'alternate_review_parallel': DetectorConfig.review(minimum_origin_diameter_mm=2, parallel_clearance_mm=1.5),
    }
    write_json(output / 'plan.json', {
        'source_sha256': {p:digest(ROOT / p) for p in ['next_steps_audit.py','pipeline.py','detector.py','learning.py','final_eval_topology.py','current_e2e_fusion.py']},
        'audit_provenance_sha256':digest(args.audit_dir / 'provenance.json'),
        'models': ['current_exact', 'ignore_contact_volume', 'current_exact_recall095', 'ignore_contact_volume_recall095'],
        'threshold_selection':'Original validation patients only; reference cases excluded from fitting and calibration.',
        'root_configurations':{k:asdict(v) for k,v in configurations.items()},
        'root_separation_mm':2., 'root_experiment_filter_threshold':.15,
        'limitations':'Post-reference development. Exact old AI reviews only; no new adjudication or independent test cohort.',
    })
    fits = {}
    for ignore in (False, True):
        for recall in (0., .95):
            name = ('ignore_contact_volume' if ignore else 'current_exact') + ('_recall095' if recall else '')
            fitting = copy.deepcopy(current)
            if ignore:
                # Feature-selection ablation: constant training column guarantees a zero coefficient.
                # Real inference still receives the unchanged original 13-feature vector.
                for row in fitting: row['features'][-1] = 0.
            model, training_report = train(fitting, split, minimum_training_recall=recall)
            if ignore and model.weights[-1] != 0: raise AssertionError('Excluded feature learned a weight')
            model.save(output / 'models' / f'{name}.json')
            predictions = {}
            for case in CASES:
                strict = read_json(args.audit_dir / 'cases' / case / 'strict-diagnostics.json')
                review = read_json(args.audit_dir / 'cases' / case / 'review_origin2-diagnostics.json')
                prediction, _ = fuse(strict, review, model, model.threshold)
                prediction['case_id'] = case; predictions[case] = prediction
                write_json(output / name / f'{case}.json', prediction)
            fits[name] = {'training':training_report,'reference_scores':score_variant(predictions)}
            s = fits[name]['reference_scores']['3']['summary']
            print(name, model.threshold, s['true_positives'],s['false_positives'],s['false_negatives'],flush=True)
    write_json(output / 'filter-fits.json', fits)
    model = CandidateModel.load(DEFAULT_MODEL)
    recovery = {name:{} for name in configurations}
    for case in CASES:
        image_path, mask_path = case_paths(case)
        image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
        strict = read_json(args.audit_dir / 'cases' / case / 'strict-diagnostics.json')
        for name, config in configurations.items():
            _, diagnostics, scan, evidence = infer(image, mask, config, root_separation_mm=2.)
            del scan, evidence
            branches = [r['branch'] for r in diagnostics['candidates'] if r['reason']=='accepted']
            branches.sort(key=lambda b:b['instance_id'])
            predictions, _ = fuse(strict, {'branches':branches}, model, .15)
            predictions['case_id'] = case;recovery[name][case] = predictions
            write_json(output / name / f'{case}.json', predictions)
            write_json(output / name / f'{case}-diagnostics.json', diagnostics)
            print(case,name,len(predictions['daughters']),flush=True)
    root_scores = {name:score_variant(p) for name,p in recovery.items()}
    write_json(output / 'root-experiments.json', root_scores)
    write_json(output / 'summary.json', {
        'current_exact_rows_excluding_references':len(current),
        'label_counts':dict(Counter(r['label'] for r in current)),
        'split':split,'queue_counts':dict(Counter(r['reason'] for r in queue)),
        'new_adjudications':0,'production_changed':False,
        'filter_scores_3mm':{n:r['reference_scores']['3']['summary'] for n,r in fits.items()},
        'root_scores_3mm':{n:r['3']['summary'] for n,r in root_scores.items()},
    })


if __name__ == '__main__':
    main()
