"""Plan/run/report the full detection study; segmentation/CHPDNet are excluded."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from study.runner import plan,run
from study.report import report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan'); p.add_argument('--config',default='config/detection_study.yaml'); p.add_argument('--output',required=True); p.add_argument('--smoke',action='store_true'); p.add_argument('--gan-steps',type=int)
    for command in ('run','finalize-test'):
        p=sub.add_parser(command); p.add_argument('--study',required=True); p.add_argument('--device',default='auto')
        p.add_argument('--models',nargs='+'); p.add_argument('--arms',nargs='+'); p.add_argument('--limit-jobs',type=int); p.add_argument('--retry-failed',action='store_true')
    p=sub.add_parser('gan'); p.add_argument('--study',required=True); p.add_argument('--device',default='auto'); p.add_argument('--subset'); p.add_argument('--limit-subsets',type=int)
    for command in ('propose-placements','blend'):
        p=sub.add_parser(command); p.add_argument('--study',required=True); p.add_argument('--subset',required=True)
    p=sub.add_parser('report'); p.add_argument('--study',required=True)
    p=sub.add_parser('import-gan'); p.add_argument('--study',required=True); p.add_argument('--subset',required=True); p.add_argument('--checkpoint',required=True)
    p=sub.add_parser('screen',help='Re-screen a queue; clears its previous human decisions')
    p.add_argument('--study',required=True); p.add_argument('--subset',required=True)
    p.add_argument('--stage',choices=['candidates','composites'],required=True)
    p=sub.add_parser('review',help='Record one explicit human decision')
    p.add_argument('--study',required=True); p.add_argument('--subset',required=True)
    p.add_argument('--stage',choices=['candidates','placements','composites'],required=True)
    p.add_argument('--id',required=True); p.add_argument('--decision',choices=['approve','reject'],required=True)
    p.add_argument('--reviewer',required=True); p.add_argument('--note',default='')
    p.add_argument('--box',nargs=4,type=float,metavar=('X1','Y1','X2','Y2'))
    for flag in ('defect-present','class-correct','placement-valid','realistic','labels-complete'):
        p.add_argument('--'+flag,action='store_true')
    for command in ('cvat-export','cvat-import'):
        p=sub.add_parser(command)
        p.add_argument('--study',required=True); p.add_argument('--subset',required=True)
        p.add_argument('--stage',choices=['candidates','placements','composites'],required=True)
        if command=='cvat-export':
            p.add_argument('--output',required=True); p.add_argument('--limit',type=int)
        else:
            p.add_argument('--package',required=True); p.add_argument('--annotations',required=True)
            p.add_argument('--reviewer',required=True); p.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='plan':
        payload=plan(args.config,args.output,args.smoke,args.gan_steps)
        print(f"Planned {len(payload['jobs'])} detector cells and {len(payload['subsets'])} training subsets. Smoke={payload['smoke']}")
    elif args.command in ('run','finalize-test'):
        if args.limit_jobs is not None and args.limit_jobs<1: parser.error('limit-jobs must be positive')
        states=run(args.study,args.device,args.models,args.arms,args.limit_jobs,args.retry_failed,args.command=='finalize-test')
        print(dict(Counter(s['status'] for s in states.values())))
        print(f'Report: {report(args.study)}')
        if any(s['status']=='failed' or s.get('test_status')=='failed' for s in states.values()): return 1
    elif args.command=='gan':
        from study.gan import train_gans
        train_gans(args.study,args.device,args.subset,args.limit_subsets)
    elif args.command=='cvat-export':
        from study.cvat import export_review
        print(export_review(args.study,args.subset,args.stage,args.output,args.limit))
    elif args.command=='cvat-import':
        from study.cvat import import_review
        print(import_review(args.study,args.subset,args.stage,args.package,args.annotations,args.reviewer,args.dry_run))
    elif args.command=='screen':
        from study.gan import screen
        print(f'Screening complete; human decisions reset. Review previews and index: {screen(args.study,args.subset,args.stage)}')
    elif args.command=='review':
        from study.gan import review_item
        print(review_item(args.study,args.subset,args.stage,args.id,args.decision,args.reviewer,args.note,
                          args.box,args.defect_present,args.class_correct,args.placement_valid,args.realistic,args.labels_complete))
    elif args.command=='propose-placements':
        from study.gan import proposals
        print(proposals(args.study,args.subset))
    elif args.command=='blend':
        from study.gan import composites
        print(f'Created {len(composites(args.study,args.subset))} composites for final visual/label review')
    elif args.command=='import-gan':
        from study.gan import import_checkpoint
        print(import_checkpoint(args.study,args.subset,args.checkpoint))
    else: print(report(args.study))
    return 0


if __name__=='__main__': raise SystemExit(main())
