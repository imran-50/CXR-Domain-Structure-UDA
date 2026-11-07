#!/usr/bin/env python3
import sys
import subprocess

def main():
    print('=' * 80)
    print('REPRODUCING ALL MANUSCRIPT TABLES & STATISTICAL TESTS')
    print('Manuscript: Characterizing Domain Structure to Anticipate UDA Performance')
    print('Authors: Iftakhar Hossain*, Md. Imran Hasan*, Taufiq Hasan (BUET mHealth Lab)')
    print('=' * 80)

    cmd = [
        sys.executable, 'scripts/02_statistics_and_tables.py',
        '--primary', 'outputs/feature_space_analysis_nih.csv',
        '--robust', 'outputs/feature_space_analysis_chex.csv',
        '--zero-shot', 'data/transfer_results/zero_shot_report_by_target.csv',
        '--dann', 'data/transfer_results/uda_dann_results_by_target.csv'
    ]
    subprocess.check_call(cmd)

if __name__ == '__main__':
    main()
