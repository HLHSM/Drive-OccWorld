#!/usr/bin/env python3
"""Count every semantic voxel, before/after remapping and inside/outside task ROI."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ADAPTER = REPO / 'projects/mmdet3d_plugin/datasets/farmsim_world_dataset.py'
ROI = np.array([0., -7.5, -2., 15., 7.5, 3.])
METADATA_FIELDS = ('sequence_path', 'frame_id', 'crop_type', 'growth_stage',
                   'time_of_day', 'weather', 'split', 'used_by_current_schedule')


def active_taxonomy():
    """Read the actual adapter constants without importing the training stack."""
    names = {'FARMSIM_CLASSES', 'FARMSIM_LABEL_REMAP'}
    nodes = []
    for node in ast.parse(ADAPTER.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        root = target.value if isinstance(target, ast.Subscript) else target
        if isinstance(root, ast.Name) and root.id in names:
            nodes.append(node)
    scope = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ADAPTER), 'exec'), scope)
    return list(scope['FARMSIM_CLASSES']), scope['FARMSIM_LABEL_REMAP']


def roi_indices(schema):
    region = next(x for x in schema['regions'] if x['id'] == schema['region_mode'])
    low, high = np.array(region['min_m_xyz']), np.array(region['max_m_xyz'])
    shape = np.array(region['dimensions_xyz'], dtype=int)
    pitch = (high-low)/shape
    if np.any(ROI[:3] < low-1e-5) or np.any(ROI[3:] > high+1e-5):
        raise ValueError('Requested task ROI exceeds source region')
    target_float = (ROI[3:]-ROI[:3])/pitch
    target = np.rint(target_float).astype(int)
    if not np.allclose(target_float, target, atol=1e-4):
        raise ValueError('Task ROI does not contain an integer number of voxels')
    indices = [np.floor((ROI[a]+(np.arange(target[a])+.5)*pitch[a]-low[a])/pitch[a]
                        + 1e-6).astype(int) for a in range(3)]
    if any(index.min() < 0 or index.max() >= n for index, n in zip(indices, shape)):
        raise ValueError('ROI index outside source grid')
    return tuple(shape[::-1]), np.ix_(indices[2], indices[1], indices[0]), target.tolist()


def scan_sequence(job):
    """Only decompress occupancy/occupancy_valid; instance arrays are unnecessary."""
    path, original_ids = job
    path = Path(path)
    schema = json.loads((path/'occupancy_schema.json').read_text())
    expected_shape, index, roi_shape = roi_indices(schema)
    files = sorted((path/'occupancy').glob('*.npz'))
    exported, full, roi, valid_counts = [], [], [], []
    permitted = set(original_ids)
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            raw, valid = data['occupancy'], data['occupancy_valid']
        if (raw.shape != expected_shape or valid.shape != expected_shape or
                raw.dtype != np.uint8 or valid.dtype != np.uint8):
            raise ValueError(f'{file}: unexpected occupancy/mask shape or dtype')
        if np.any(valid > 1):
            raise ValueError(f'{file}: nonbinary occupancy_valid')
        unmasked = np.bincount(raw.ravel(), minlength=256)
        unknown = set(np.flatnonzero(unmasked))-permitted
        if unknown:
            raise ValueError(f'{file}: labels not declared by the schema: {unknown}')
        masked = np.bincount(raw[valid != 0], minlength=256)
        selected, selected_valid = raw[index], valid[index]
        cropped = np.bincount(selected[selected_valid != 0], minlength=256)
        # Effective ignore combines mask=0 with raw unknown/ignored classes.
        masked[255] += raw.size-int(valid.sum())
        cropped[255] += selected.size-int(selected_valid.sum())
        exported.append(unmasked[original_ids])
        full.append(masked[original_ids])
        roi.append(cropped[original_ids])
        valid_counts.append((int(valid.sum()), int(selected_valid.sum())))
    return dict(sequence_path=str(path), frame_ids=[p.stem for p in files],
                raw=np.asarray(exported, dtype=np.int64),
                full=np.asarray(full, dtype=np.int64), roi=np.asarray(roi, dtype=np.int64),
                valid_mask_counts=np.asarray(valid_counts, dtype=np.int64),
                source_shape_zyx=[int(value) for value in expected_shape], roi_shape_xyz=roi_shape)


def remap_counts(counts, original_ids, mapping, target_ids):
    result = np.zeros((len(counts), len(target_ids)), dtype=np.int64)
    for column, label in enumerate(target_ids):
        result[:, column] = counts[:, mapping[original_ids] == label].sum(axis=1)
    return result


def percent(numerator, denominator):
    return f'{100*numerator/denominator:.10f}' if denominator else ''


def write_table(path, records, counts, ids, names, exported=False, mask_counts=None):
    class_fields = []
    for label, name in zip(ids, names):
        prefix = f'class_{label}_{name}'
        class_fields += [prefix+'_count', prefix+'_pct_all']
        if not exported:
            class_fields += [prefix+'_pct_valid', prefix+'_pct_occupied']
    base = ['sample_index', *METADATA_FIELDS, 'total_voxels', 'valid_voxels',
            'occupied_voxels', 'ignored_voxels']
    if exported:
        base = ['sample_index', *METADATA_FIELDS, 'total_voxels', 'occupancy_valid_mask_voxels']
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(base+class_fields)
        for index, (record, row) in enumerate(zip(records, counts)):
            total, ignored = int(row.sum()), int(row[-1])
            valid = total-ignored
            occupied = valid-int(row[0])
            output = [index, *(record[field] for field in METADATA_FIELDS), total]
            output += [int(mask_counts[index])] if exported else [valid, occupied, ignored]
            for label, value in zip(ids, row):
                output += [int(value), percent(value, total)]
                if not exported:
                    output += [percent(value, valid) if label != 255 else '',
                               percent(value, occupied) if label not in (0, 255) else '']
            writer.writerow(output)


def aggregate(counts, ids, names):
    totals = counts.sum(1)
    valid = totals-counts[:, -1]
    occupied = valid-counts[:, 0]
    total_count, valid_count, occupied_count = map(int, (totals.sum(), valid.sum(), occupied.sum()))
    classes = []
    for column, (label, name) in enumerate(zip(ids, names)):
        values = counts[:, column]
        ratio = np.divide(values*100., valid, out=np.full(len(valid), np.nan), where=valid > 0)
        population = ratio[np.isfinite(ratio)] if label != 255 else np.array([])
        count = int(values.sum())
        classes.append(dict(id=int(label), name=name, voxel_count=count,
            pct_all=100*count/total_count if total_count else None,
            pct_valid=100*count/valid_count if label != 255 and valid_count else None,
            pct_occupied=100*count/occupied_count if label not in (0,255) and occupied_count else None,
            frames_present=int(np.count_nonzero(values)),
            frame_mean_pct_valid=float(population.mean()) if population.size else None,
            frame_std_pct_valid=float(population.std()) if population.size else None,
            frame_median_pct_valid=float(np.median(population)) if population.size else None,
            frame_min_pct_valid=float(population.min()) if population.size else None,
            frame_max_pct_valid=float(population.max()) if population.size else None))
    return dict(samples=len(counts), total_voxels=total_count, valid_voxels=valid_count,
                occupied_voxels=occupied_count, ignored_voxels=total_count-valid_count, classes=classes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path('/data/HL/SimData-panoptic/SimData-panoptic'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, default=REPO/'data/simdata_panoptic/splits')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root, output = args.data_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists():
        raise FileExistsError('Choose a new output directory; completed statistics already exist')
    sequences = [p for p in sorted(root.glob('*/sequence_*')) if (p/'occupancy_schema.json').is_file()]
    first_schema = json.loads((sequences[0]/'occupancy_schema.json').read_text())
    original_ids = sorted(int(k) for k in first_schema['classes'])
    if original_ids[0] != 0 or original_ids[-1] != 255:
        raise ValueError('Expected free=0 and ignore=255 in the source taxonomy')
    original_names = [first_schema['classes'][str(k)] for k in original_ids]
    effective_names = original_names[:-1]+['ignore']
    class_names, mapping = active_taxonomy()
    target_ids, target_names = list(range(len(class_names)))+[255], class_names+['ignore']
    if any(mapping[i] == 255 for i in original_ids if i != 255):
        raise ValueError('A declared source class is ignored by the current task mapping')
    split_lookup, hashes = {}, {}
    for split in ('train','val','test'):
        path = args.split_dir/f'{split}.json'
        if not path.exists():
            continue
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        for seq in json.loads(path.read_text())['sequences']:
            for index, frame in enumerate(seq['frame_ids']):
                key = (seq['path'], frame)
                if key in split_lookup:
                    raise ValueError(f'Split overlap: {key}')
                split_lookup[key] = (split, split == 'test' or index % 2 == 0)
    records, raw, full, roi, valid_masks, sequence_info = [], [], [], [], [], []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for sequence_number, item in enumerate(executor.map(scan_sequence,
                ((str(p), original_ids) for p in sequences)), 1):
            seq = Path(item['sequence_path'])
            schema = json.loads((seq/'occupancy_schema.json').read_text())
            if schema['classes'] != first_schema['classes']:
                raise ValueError(f'{seq}: source taxonomy differs across sequences')
            meta = json.loads((seq/'scenario_manifest.json').read_text())
            relative = str(seq.relative_to(root))
            sequence_info.append(dict(path=relative, source_shape_zyx=item['source_shape_zyx'],
                roi_shape_xyz=item['roi_shape_xyz'], frame_count=len(item['frame_ids'])))
            for frame in item['frame_ids']:
                split, selected = split_lookup.get((relative,frame), ('unassigned',False))
                records.append(dict(sequence_path=relative, frame_id=frame, split=split,
                    used_by_current_schedule=selected,
                    **{k:meta[k] for k in ('crop_type','growth_stage','time_of_day','weather')}))
            raw.append(item['raw']); full.append(item['full']); roi.append(item['roi'])
            valid_masks.append(item['valid_mask_counts'])
            if sequence_number % 25 == 0 or sequence_number == len(sequences):
                print(f'Scanned {sequence_number}/{len(sequences)} sequences; '
                      f'{len(records)} samples; {time.monotonic()-start:.1f}s', flush=True)
    raw, full, roi, valid_masks = (np.concatenate(x) for x in (raw,full,roi,valid_masks))
    full6 = remap_counts(full, original_ids, mapping, target_ids)
    roi6 = remap_counts(roi, original_ids, mapping, target_ids)
    assert np.array_equal(raw.sum(1), full.sum(1))
    assert np.array_equal(full.sum(1), full6.sum(1))
    assert np.array_equal(roi.sum(1), roi6.sum(1))
    assert np.array_equal(full[:,-1], full6[:,-1]) and np.array_equal(roi[:,-1], roi6[:,-1])
    assert all(len(x) == len(records) for x in (raw,full,roi,full6,roi6))
    views = dict(source_grid_original=(full,original_ids,effective_names),
                 source_grid_task6=(full6,target_ids,target_names),
                 task_roi_original=(roi,original_ids,effective_names),
                 task_roi_task6=(roi6,target_ids,target_names))
    split_array = np.array([r['split'] for r in records])
    selected = np.array([r['used_by_current_schedule'] for r in records])
    summary = dict(data_root=str(root), sequences=len(sequences), samples=len(records),
        created_at=datetime.now(timezone.utc).isoformat(),
        task_roi_m_xyz=ROI.tolist(), original_class_definitions=first_schema['classes'],
        task_class_definitions=dict(zip(target_ids,target_names)),
        original_to_task_mapping={str(i):int(mapping[i]) for i in original_ids},
        adapter_path=str(ADAPTER), adapter_sha256=hashlib.sha256(ADAPTER.read_bytes()).hexdigest(),
        split_manifest_sha256=hashes, split_counts=dict(Counter(split_array.tolist())),
        current_schedule_counts=dict(Counter(split_array[selected].tolist())),
        source_shapes_zyx=[list(x) for x in sorted({tuple(s['source_shape_zyx']) for s in sequence_info})],
        roi_shapes_xyz=[list(x) for x in sorted({tuple(s['roi_shape_xyz']) for s in sequence_info})],
        excluded_nonsequence_directories=[str(p.relative_to(root)) for p in sorted(root.iterdir())
                                          if p.is_dir() and not list(p.glob('sequence_*'))],
        denominators=dict(pct_all='all grid voxels, including ignored voxels',
                          pct_valid='mask-valid voxels with recognized class, including free',
                          pct_occupied='recognized valid voxels excluding free',
                          frame_mean_pct_valid='unweighted mean of per-frame percentages; differs from pooled voxel percentage'),
        notes=['Every occupancy NPZ is counted; no RGB subsampling, CropSample or rendering point cap is applied.',
               'Both taxonomies are counted over both source grid and the exact task ROI.',
               'Source export CSV records on-disk labels before masking; the other CSVs combine mask=0 and raw255 into ignore.',
               'Task ROI follows native loader target-center nearest-neighbor sampling.',
               'Instance IDs are not semantic classes. occupancy_visible is not an additional training mask.',
               'Current schedule means train/val stride2, test stride1; global statistics always use all samples.'],
        views={}, by_split={}, current_schedule={})
    for name, (counts,ids,names) in views.items():
        summary['views'][name] = aggregate(counts,ids,names)
        summary['by_split'][name] = {s:aggregate(counts[split_array==s],ids,names) for s in sorted(set(split_array))}
        summary['current_schedule'][name] = {s:aggregate(counts[(split_array==s)&selected],ids,names)
                                           for s in ('train','val','test') if np.any((split_array==s)&selected)}
        write_table(output/f'per_sample_{name}.csv', records,counts,ids,names)
        print(f'Wrote per_sample_{name}.csv',flush=True)
    write_table(output/'per_sample_source_exported_original.csv', records,raw,
                original_ids,original_names,exported=True,mask_counts=valid_masks[:,0])
    with (output/'sample_index.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['sample_index',*METADATA_FIELDS])
        writer.writeheader();writer.writerows(dict(sample_index=i,**r) for i,r in enumerate(records))
    np.savez_compressed(output/'per_sample_counts.npz', original_class_ids=np.array(original_ids),
        task_class_ids=np.array(target_ids),source_exported_original=raw,
        source_grid_original=full,source_grid_task6=full6,task_roi_original=roi,task_roi_task6=roi6,
        occupancy_valid_mask_voxels=valid_masks,used_by_current_schedule=selected)
    (output/'sequences.json').write_text(json.dumps(sequence_info,indent=2)+'\n')
    summary['elapsed_seconds']=round(time.monotonic()-start,3)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (output/'overall_class_statistics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['view',*summary['views']['task_roi_task6']['classes'][0]])
        writer.writeheader()
        for name,stats in summary['views'].items():
            writer.writerows(dict(view=name,**r) for r in stats['classes'])
    lines=['# SimData-panoptic 逐样本语义体素统计','',
        f'全量 {len(records):,} 帧，{len(sequences)} 个序列。实例 ID 不作为语义类别。','',
        '全部逐帧 CSV 均可通过 `sequence_path`、`frame_id` 和 `sample_index` 对齐。比例字段单位为百分比。',
        '非 exported 文件均先应用 occupancy_valid，再统计原始类别或当前六类；ignore 包括掩码外及原始255。',
        'pct_all 的分母包含 ignore；pct_valid 的分母排除 ignore、包含 free；pct_occupied 另排除 free。',
        'ignore 没有 pct_valid；free 和 ignore 没有 pct_occupied。原始存储标签单独保存在 exported CSV。','',
        '`source_grid` 是原始完整区域；`task_roi` 是当前前方15×15米区域（z=-2至3米），按训练加载器采样体素中心。',
        '`original` 使用原始11类；`task6` 使用当前6类，人和动物及树木均并入其他障碍物。','',
        'summary.json 包含全量、按划分及当前 train/val stride2、test stride1 日程的汇总。全量统计不做隔帧或 CropSample。',
        '总体占比通过体素计数累加得到；frame_mean_pct_valid 为逐帧占比的算术平均，两者不能混用。','',
        '| 文件 | 内容 |','|---|---|',
        '| per_sample_source_grid_original.csv | 完整源网格，原始类别，有效掩码后 |',
        '| per_sample_source_grid_task6.csv | 完整源网格，当前六类 |',
        '| per_sample_task_roi_original.csv | 当前任务范围，原始类别 |',
        '| per_sample_task_roi_task6.csv | 当前任务范围，当前六类 |',
        '| per_sample_source_exported_original.csv | 原始NPZ存储标签，不应用有效掩码 |',
        '| per_sample_counts.npz | 精确整数计数；行号对应sample_index.csv |',
        '| overall_class_statistics.csv | 总体占比及逐帧分布摘要 |','']
    for name, stats in summary['views'].items():
        lines += [f'## {name}','',f"有效体素 {stats['valid_voxels']:,}；忽略体素 {stats['ignored_voxels']:,}。",'',
                  '| ID | 类别 | 体素数 | 占全网格 % | 占有效体素 % | 出现帧数 |','|---:|---|---:|---:|---:|---:|']
        for c in stats['classes']:
            valid_text=f"{c['pct_valid']:.6f}" if c['pct_valid'] is not None else '—'
            lines.append(f"| {c['id']} | {c['name']} | {c['voxel_count']:,} | {c['pct_all']:.6f} | {valid_text} | {c['frames_present']:,} |")
        lines.append('')
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Completed {len(records)} samples in {summary["elapsed_seconds"]:.1f}s: {output}',flush=True)


if __name__ == '__main__':
    main()
